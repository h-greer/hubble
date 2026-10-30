"""Opt-in exposure batching (`mapped=True`): the loss / Gauss-Newton diagonals use jax.lax.map over stacked exposures
instead of a Python loop that unrolls the forward model once per exposure (compile time and HLO size flat in
n_exposures, ~half the memory, but ~1.1-1.3x slower per step; agrees with the list path to ~1e-14, not bit-for-bit).

Exposures with the same type, fit treedef, filter and leaf shapes/dtypes form a Group.  Leaves that differ between its
members (data, err, bad, ... and any differing fit leaf) are stacked; everything else, including the static fields
(mjd, target, filename, ...), comes from the first member, so a fit must not read those while it runs.  Dict params
(per exposure / visit / target / global) are stacked per member and shown to the fit under the first member's key.
"""
import numpy as onp
import jax, jax.numpy as np, jax.random as jr, jax.tree_util as jtu, equinox as eqx
from jax.flatten_util import ravel_pytree

from models import ModelParams
from stats import posterior
# precond is imported lazily in gn_core_mapped (it imports fitting, which imports this module)


class Group(eqx.Module):
    template: object                          # first member
    xs: object                                # template-shaped; stacked (n, ...) leaves that differ, else None
    gidx: jax.Array                           # (n,) position of each member in the list given to make_batch
    keys: tuple = eqx.field(static=True)      # ((param name, (key of each member, ...)), ...)

    def rows(self, mdl):
        """{name: value of each member's key, stacked}"""
        return {n: jtu.tree_map(lambda *v: np.stack(v), *[mdl.params[n][k] for k in ks]) for n, ks in self.keys}

    def at(self, mdl, rows, xs):
        """(model, exposure) of one member, given its slice of `rows` and `xs`."""
        new = {**mdl.params, **{n: {ks[0]: rows[n]} for n, ks in self.keys}}
        return mdl.set("params", new), eqx.combine(xs, self.template)

    def member(self, mdl, j):
        """As `at`, for member j of the stack (j may be traced)."""
        return self.at(mdl, *jtu.tree_map(lambda a: a[j], (self.rows(mdl), self.xs)))


class ExposureBatch(eqx.Module):
    groups: tuple
    checkpoint: bool = eqx.field(static=True, default=False)  # remat each exposure's forward in the backward pass


def make_batch(exposures, model, checkpoint=False):
    """ExposureBatch of `exposures` (host side, outside jit). `model` must hold every dict param (values unused)."""
    def sig(l):
        return (onp.shape(l), str(onp.asarray(l).dtype)) if isinstance(l, (jax.Array, onp.ndarray, onp.generic, int, float, bool)) else repr(l)

    groups = {}
    for p, e in enumerate(exposures):
        groups.setdefault((type(e), jtu.tree_structure(e.fit), e.filter, tuple(map(sig, jtu.tree_leaves(e)))), []).append(p)
    out = []
    for ps in groups.values():
        es = [exposures[p] for p in ps]
        leaves = [jtu.tree_leaves(e) for e in es]
        diff = [not all(onp.array_equal(l[j], leaves[0][j]) for l in leaves[1:]) for j in range(len(leaves[0]))]
        xs = [np.stack([np.asarray(l[j]) for l in leaves]) if d else None for j, d in enumerate(diff)]
        keys = tuple((n, tuple(e.get_key(n) for e in es)) for n, v in model.params.items()
                     if isinstance(v, dict) and n in es[0].fit.PARAM_KEYS)
        out.append(Group(es[0], jtu.tree_unflatten(jtu.tree_structure(es[0]), xs), np.asarray(ps, np.int32), keys))
    return ExposureBatch(tuple(out), checkpoint)


def loss_fn_mapped(params, batch, model):
    """fitting.loss_fn for an ExposureBatch."""
    mdl = params.inject(model)
    one = lambda g: lambda a: posterior(*g.at(mdl, *a))
    ck = jax.checkpoint if batch.checkpoint else lambda f: f
    return sum(np.nansum(jax.lax.map(ck(one(g)), (g.rows(mdl), g.xs))) for g in batch.groups)


# ----------------------------------------------------------------------------- Gauss-Newton
@eqx.filter_jit
def _gn_core(params, batch, exposures, model, key, ids, ok, quad_idx, n_probes, probe_batch, chunk):
    from precond import _f64, _finish, _hessian_entries, _regulariser_curvature
    params = _f64(params)
    x, unravel = ravel_pytree(params)
    gn = np.zeros_like(x)

    def sw(g, j):  # as precond._weighted_images
        mdl, e = g.member(ModelParams(unravel(x)).inject(model), j)
        err = np.where(e.bad, 1., e.err)
        if "quadrature" in mdl.params.keys():
            err = err * 10 ** mdl.get(e.fit.map_param(e, "quadrature"))
        w = np.where(e.bad, 0., 1. / err ** 2)
        return jax.lax.stop_gradient(np.sqrt(np.where(np.isfinite(w), w, 0.)))

    def f(g, xf, j, s):
        mdl, e = g.member(ModelParams(unravel(xf)).inject(model), j)
        return np.where(s > 0, e.fit(mdl, e) * s, 0.).ravel()

    for g in batch.groups:
        n = len(g.gidx)
        if n_probes > 0:  # Hutchinson; fold_in index = position in the exposure list, as precond._gn_core
            def body(gn, a, g=g):
                j, gi = a
                s = sw(g, j)
                out, vjp = jax.vjp(lambda xf: f(g, xf, j, s), x)
                probe = lambda k: vjp(jr.rademacher(k, out.shape, dtype=x.dtype))[0] ** 2
                keys = jr.split(jr.fold_in(key, gi), n_probes)
                return gn + jax.lax.map(probe, keys, batch_size=min(probe_batch, n_probes)).mean(0), None
            gn, _ = jax.lax.scan(body, gn, (np.arange(n), g.gidx))

    if ids.shape[0] > 0:  # exact columns, `chunk` tangents per row; a row only goes through the members that read it
        def row(a):
            ids, ok = a
            V = jax.nn.one_hot(ids, x.size, dtype=x.dtype)  # from traced ids, see precond._map_tangents
            tot = np.zeros(chunk, x.dtype)
            for g in batch.groups:
                def body(acc, a, g=g):
                    j, gi = a
                    def tan():
                        s = sw(g, j)
                        return jax.vmap(lambda v: np.sum(jax.jvp(lambda xf: f(g, xf, j, s), (x,), (v,))[1] ** 2))(V)
                    return acc + jax.lax.cond(ok[gi], tan, lambda: np.zeros(chunk, x.dtype)), None
                tot = jax.lax.scan(body, tot, (np.arange(len(g.gidx)), g.gidx))[0]
            return tot
        gn = gn.at[ids.ravel()].set(jax.lax.map(row, (ids, ok)).ravel())

    if quad_idx.shape[0] > 0:  # |H_kk|, as precond._gn_core
        gn = gn.at[quad_idx].set(np.abs(_hessian_entries(x, unravel, batch, model, quad_idx, chunk)))
    reg, _ = ravel_pytree(_regulariser_curvature(params, exposures))
    return _finish(gn + reg, unravel)


def gn_core_mapped(params, exposures, model, key, exact_groups, quad_idx, n_probes, probe_batch, chunk):
    """precond._gn_core with the same arguments, mapping over stacked exposures."""
    cols = [onp.pad(idx, (0, -len(idx) % chunk), mode="edge").reshape(-1, chunk) for _, idx in exact_groups]
    ok = [onp.broadcast_to(onp.isin(onp.arange(len(exposures)), users), (len(c), len(exposures)))
          for (users, _), c in zip(exact_groups, cols)]
    return _gn_core(params, make_batch(exposures, model), exposures, model, key,
                    onp.concatenate(cols) if cols else onp.zeros((0, chunk), onp.int32),
                    onp.concatenate(ok) if ok else onp.zeros((0, len(exposures)), bool),
                    quad_idx, n_probes, probe_batch, chunk)
