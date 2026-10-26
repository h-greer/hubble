"""Diagonal curvature estimates (preconditioner diagonals) for the HST/NICMOS fits.

Loss = sum over exposures of the Gaussian NLL (+ the PointResolvedFit regulariser). Only
diagonals are ever computed; no full matrix is materialised. Three estimators, all returning a
pytree shaped like `params` (float64, non-negative, NaN where the computation was not finite):

  hessian_diag   |H_kk|, one forward-over-reverse HVP per parameter.
  gn_diag_exact  Gauss-Newton  sum_e sum_pix w (dm/dtheta_k)^2, one jvp per parameter.
  gn_diag_probe  Hutchinson estimate of the same, one vjp per exposure (small blocks exact).

`quadrature` enters the error model (w) non-quadratically, so GN is meaningless for it and
its entries use |H_kk| instead. The regulariser's curvature is added in closed form for the GN
methods (autodiff already includes it in hessian_diag). Needs jax_enable_x64.
"""
import warnings
import numpy as onp
import jax
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from jax.flatten_util import ravel_pytree

from models import ModelParams, TSV_loss
from fitting import loss_fn
from batching import make_batch, gn_core_mapped

LN10 = float(np.log(10.0))


# ----------------------------------------------------------------------------- helpers
def _f64(params):
    return jtu.tree_map(lambda a: np.asarray(a, dtype=np.float64), params)


def _weighted_images(x, unravel, exposures, model):
    """Per exposure, f(x) = sqrt(w) * model image (flat), with w = 1/err^2 held constant.

    Then sum_pix w (dm/dtheta)^2 = |df/dtheta|^2. Masked pixels are exactly 0, so NaNs there
    cannot leak into the sums. `quadrature` rescales err by 10**quadrature when fitted.
    """
    mdl = ModelParams(unravel(x)).inject(model)
    fs = []
    for e in exposures:
        err = np.where(e.bad, 1., e.err)
        if "quadrature" in mdl.params.keys():
            err = err * 10 ** mdl.get(e.fit.map_param(e, "quadrature"))
        w = np.where(e.bad, 0., 1. / err ** 2)
        sw = jax.lax.stop_gradient(np.sqrt(np.where(np.isfinite(w), w, 0.)))

        def f(xf, e=e, sw=sw):
            im = e.fit(ModelParams(unravel(xf)).inject(model), e)
            return np.where(sw > 0, im * sw, 0.).ravel()

        fs.append(f)
    return fs


def _regulariser_curvature(params, exposures):
    """Exact d2(reg)/d(resolved)^2, a pytree like params (zero unless PointResolvedFit-style).

    reg = reg[0]*L2(d) + reg[1]*TSV(d) with d = 10**resolved and TSV(d) = d^T A d, where
    A_kk = 4 for every pixel (tikhonov pads by 2), so grad_d TSV = 2 A d and
      d2 TSV/dr2 = LN10^2 (2 A_kk d^2 + d g) = LN10^2 (8 d^2 + d g),  g = grad TSV,
      d2 L2 /dr2 = LN10^2 * 4 d^2.
    Clipped at >= 0 since d*g can be negative. No n x n matrix is needed.
    """
    P = ModelParams(params)
    out = ModelParams(jtu.tree_map(np.zeros_like, params))
    if "resolved" not in params:
        return out.params
    for e in exposures:
        if hasattr(e.fit, "n_share"):  # GPResolvedFit: prior 0.5*|z|^2/n_share has curvature 1/n_share
            path = e.fit.map_param(e, "resolved")
            out = out.set(path, out.get(path) + 1/e.fit.n_share)
            continue
        reg = getattr(e.fit, "regulariser", None)
        if reg is None:
            continue
        path = e.fit.map_param(e, "resolved")
        d = 10 ** P.get(path)
        g = jax.grad(TSV_loss)(d)
        c = LN10 ** 2 * (reg[0] * 4 * d ** 2 + reg[1] * (8 * d ** 2 + d * g))
        out = out.set(path, out.get(path) + np.clip(c, 0., None))
    return out.params


def get_keys(exposures, name):
    """Per exposure, the key of dict parameter `name` it reads (None if get_key cannot say)."""
    def key(e):
        try:
            return e.fit.get_key(e, name)
        except Exception:
            return None
    return [key(e) for e in exposures]


def leaf_users(exposures, name, k):
    """Indices of the exposures that read leaf params[name][k] (k None: a non-dict parameter).
    Shared by every exposure when k is None, some get_key fails, or nobody matches."""
    everyone = tuple(range(len(exposures)))
    if k is None:
        return everyone
    keys = get_keys(exposures, name)
    users = tuple(i for i, kk in enumerate(keys) if kk == k)
    return users if users and None not in keys else everyone


def _indices(params, exposures, exact_blocks, exact_max_size):
    """(exact_groups, quad_idx): flat (ravel_pytree order) int32 indices.

    exact: every entry of the blocks in exact_blocks, plus every leaf of size <= exact_max_size,
    grouped as ((users, idx), ...) by the exposures that read them (`leaf_users`), so a column is
    only pushed through those exposures.  quadrature: the `quadrature` block (handled by the Hessian).
    """
    exact, quad, start = {}, [], 0
    for name in sorted(params):  # ravel_pytree flattens dicts in sorted-key order
        val = params[name]
        for k, v in sorted(val.items()) if isinstance(val, dict) else [(None, val)]:
            users = leaf_users(exposures, name, k)
            for leaf in jtu.tree_leaves(v):
                idx = range(start, start + int(np.size(leaf)))
                if name == "quadrature":
                    quad += idx
                elif name in exact_blocks or len(idx) <= exact_max_size:
                    exact.setdefault(users, []).extend(idx)
                start += len(idx)
    return tuple((u, onp.asarray(i, dtype=onp.int32)) for u, i in exact.items()), onp.asarray(quad, dtype=onp.int32)


def _map_tangents(fn, x, idx, batch_size):
    """[fn(e_i, i) for i in idx], `batch_size` at a time via lax.map; e_i is the one-hot tangent.

    The tangent is built from the TRACED index i inside the map. A compile-time-constant
    tangent (e.g. a jnp.ones_like or a Python-loop one-hot) lets XLA constant-fold through the
    jvp and can miscompile (see proposals/jax_jvp_const_tangent_repro.py).
    """
    def one(i):
        return fn(jax.nn.one_hot(i, x.size, dtype=x.dtype), i)

    return jax.lax.map(one, idx, batch_size=min(int(batch_size), idx.shape[0]))


def _hessian_entries(x, unravel, exposures, model, idx, batch_size):
    """H_kk at flat indices idx, from one HVP (forward-over-reverse) each."""
    grad = jax.grad(lambda xf: loss_fn(ModelParams(unravel(xf)), exposures, model))
    return _map_tangents(lambda v, i: jax.jvp(grad, (x,), (v,))[1][i], x, idx, batch_size)


def _finish(flat, unravel):
    # Non-finite entries stay NaN (never 0: C=0 would freeze that parameter and get cached);
    # precond_cache refuses to cache non-finite diagonals and precond_from_diag imputes them.
    return unravel(np.where(np.isfinite(flat), np.abs(flat), np.nan))


# ----------------------------------------------------------------------------- jitted cores
# Module level and taking data as arguments, so repeat calls with the same structure, shapes and
# static ints reuse the compiled code (closing over exposures/model would recompile every call).
@eqx.filter_jit
def _hessian_core(params, exposures, model, idx, batch_size):
    x, unravel = ravel_pytree(_f64(params))
    h = _hessian_entries(x, unravel, exposures, model, idx, batch_size)
    return _finish(np.zeros_like(x).at[idx].set(h), unravel)


@eqx.filter_jit
def _gn_core(params, exposures, model, key, exact_groups, quad_idx, n_probes, probe_batch, chunk):
    """GN diagonal; n_probes == 0 means everything is in exact_groups (no Hutchinson part).
    exact_groups: ((exposure indices, flat indices), ...), the indices are traced, the users static."""
    params = _f64(params)
    x, unravel = ravel_pytree(params)
    fs = _weighted_images(x, unravel, exposures, model)
    gn = np.zeros_like(x)
    lins = [jax.linearize(f, x) for f in fs]  # (out, jvp fn): each forward is traced once, shared by vjps and jvps

    if n_probes > 0:  # Hutchinson: GN_k = E_z[(J^T (z sqrt(w)))_k^2], z Rademacher over pixels
        for ei, (out, lin) in enumerate(lins):
            vjp = jax.linear_transpose(lin, x)

            def probe(k, vjp=vjp, shape=out.shape):
                return vjp(jr.rademacher(k, shape, dtype=x.dtype))[0] ** 2

            keys = jr.split(jr.fold_in(key, ei), n_probes)
            gn = gn + jax.lax.map(probe, keys, batch_size=min(int(probe_batch), n_probes)).mean(0)

    for users, idx in exact_groups:  # exact GN_kk: one jvp per k, only through the exposures that read it
        gn_k = lambda v, i: sum(np.sum(lins[e][1](v) ** 2) for e in users)
        gn = gn.at[idx].set(_map_tangents(gn_k, x, idx, chunk))

    if quad_idx.shape[0] > 0:  # non-quadratic in the error model: |H_kk| instead
        h = _hessian_entries(x, unravel, exposures, model, quad_idx, chunk)
        gn = gn.at[quad_idx].set(np.abs(h))

    reg, _ = ravel_pytree(_regulariser_curvature(params, exposures))
    return _finish(gn + reg, unravel)


# ----------------------------------------------------------------------------- public API
def hessian_diag(params: dict, exposures: list, model, batch_size: int = 16, mapped: bool = False) -> dict:
    """|H_kk| of the full loss (incl. regulariser): n_params HVPs, `batch_size` at a time.
    mapped: lax.map over stacked exposures (batching.py)."""
    exposures = make_batch(exposures, model) if mapped else exposures
    n = sum(int(np.size(l)) for l in jtu.tree_leaves(params))
    return _hessian_core(params, exposures, model, onp.arange(n, dtype=onp.int32), int(batch_size))


def gn_diag_exact(params: dict, exposures: list, model, chunk: int = 16, mapped: bool = False) -> dict:
    """Exact GN diagonal (+ regulariser curvature): one jvp per parameter, `chunk` at a time. mapped: see hessian_diag."""
    exact, quad = _indices(params, exposures, tuple(params), 0)
    return (gn_core_mapped if mapped else _gn_core)(params, exposures, model, jr.key(0), exact, quad, 0, 1, int(chunk))


def gn_diag_probe(params: dict, exposures: list, model, n_probes: int = 32,
                  probe_batch: int = 4, key=jr.key(0),
                  exact_blocks: tuple = ("cold_mask_opd",), exact_max_size: int = 64,
                  chunk: int = 16, mapped: bool = False) -> dict:
    """Hutchinson estimate of the GN diagonal: mean over probes of (J^T (z sqrt(w)))^2.

    One vjp per exposure, `probe_batch` probes at a time. Probes are noisy for small blocks, so
    leaves of size <= exact_max_size and all of `exact_blocks` use exact jvps (`chunk` at a time). mapped: see hessian_diag.
    """
    exact, quad = _indices(params, exposures, tuple(exact_blocks), int(exact_max_size))
    return (gn_core_mapped if mapped else _gn_core)(params, exposures, model, key, exact, quad, int(n_probes),
                                                    int(probe_batch), int(chunk))


DIAG_METHODS = {"hessian": hessian_diag, "gn_exact": gn_diag_exact, "gn_probe": gn_diag_probe}


def precond_from_diag(diag: dict, damping: float = 1e-3, eps: float = 0.) -> dict:
    """C = 1/(diag + damping*block_mean + eps), 0 where the denominator is not positive.

    Blocks are the top-level keys of `diag`; non-finite entries are replaced by the mean of the
    block's finite entries (with a warning).
    """
    out = {}
    for k, blk in diag.items():
        leaves = jtu.tree_leaves(blk)
        if len(leaves) == 0:
            out[k] = blk
            continue
        flat = np.concatenate([np.ravel(np.asarray(l, dtype=np.float64)) for l in leaves])
        fin = np.isfinite(flat)
        nbad = int(np.sum(~fin))
        mean = np.sum(np.where(fin, flat, 0.)) / np.maximum(np.sum(fin), 1)
        if nbad:
            warnings.warn(f"precond_from_diag: block '{k}' has {nbad}/{flat.size} non-finite diag entries; "
                          "replaced by the block mean of the finite entries", RuntimeWarning, stacklevel=2)

        def inv(d, mean=mean):
            d = np.where(np.isfinite(d), np.asarray(d, dtype=np.float64), mean)
            den = d + damping * mean + eps
            return np.where(den > 0, 1. / np.where(den > 0, den, 1.), 0.)

        out[k] = jtu.tree_map(inv, blk)
    return out
