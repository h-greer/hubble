import hashlib
import re
import warnings

import numpy as onp

import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax
from jax.flatten_util import ravel_pytree

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx
import optax
from zodiax import optimisation as opt
import optimistix as optx

from tqdm.auto import tqdm

from apertures import *
from detectors import *
from spectra import *
from models import *
from stats import *
from batching import ExposureBatch, make_batch, loss_fn_mapped


def loss_fn(params, exposures, model):
    if isinstance(exposures, ExposureBatch):  # mapped=True, see batching.py
        return loss_fn_mapped(params, exposures, model)
    mdl = params.inject(model)
    return np.nansum(np.stack([posterior(mdl, exposure) for exposure in exposures]))


def _as_float(params):
    return jax.tree.map(lambda x: np.asarray(x, float), params)


def set_array(pytree):
    """Cast every float leaf to float64 (float32 without x64), leaving other leaves alone."""
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jax.tree.map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)


def sgd(lr, delay, momentum=0.5):
    """SGD whose learning rate is zero for the first `delay` steps."""
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)


def adam(lr, delay):
    """Adam whose learning rate is zero for the first `delay` steps."""
    return optax.adam(zdx.optimisation.delay(lr, delay))


def get_optimiser_new(model_params, optimisers):
    """Build (optim, state) from a {name: optax optimiser} dict. Like
    zodiax.optimisation.map_optimisers, but optimisers for params that are not being
    optimised are ignored, and params without an optimiser are frozen."""
    params = _as_float(dict(model_params))
    opts = {k: optimisers.get(k, optax.set_to_zero()) for k in params}
    optim = optax.multi_transform(opts, {k: k for k in params})
    return optim, optim.init(params)


# Compiled once per (params, exposures, model) structure and independent of the
# optimiser, so changing learning rates, delays or stage never recompiles it. Only the
# small optimiser update below is specific to the optimiser.
@eqx.filter_jit
def _loss_grad(params, C, exposures, model):
    loss, grads = eqx.filter_value_and_grad(
        lambda p: loss_fn(ModelParams(p), exposures, model)
    )(params)
    # Diagonal preconditioner (natural-gradient-like) step
    return loss, jax.tree.map(lambda g, c: g * c, grads, C)


_UPDATE_CACHE = {}


def _get_update(optim, params, state):
    """Jitted (grads, state, params) -> (params, state, flat params), cached per optimiser.

    optax optimisers are fresh closures, so identical ones compare unequal and a jit
    keyed on the object would recompile on every call. They carry no spec to hash
    instead, so we key on what they compute: the jaxpr of `optim.update` (minus memory
    addresses) plus its captured constants (learning rates, delays, ...).
    """
    closed = jax.make_jaxpr(optim.update)(params, state, params)
    h = hashlib.sha256(re.sub(r" at 0x[0-9a-f]+", "", str(closed.jaxpr)).encode())
    for c in closed.consts:
        c = onp.asarray(c)
        h.update(f"{c.dtype}{c.shape}".encode() + c.tobytes())
    key = h.hexdigest()

    if key not in _UPDATE_CACHE:
        @eqx.filter_jit
        def update(grads, state, params):
            updates, state = optim.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, ravel_pytree(params)[0]

        _UPDATE_CACHE[key] = update
    return _UPDATE_CACHE[key]


class ParamsHistory:
    """List-like history of a params dict, stored as one host (n_steps, n_params) array.

    history[i] gives a params dict of jax arrays; iterating or slicing gives numpy
    dicts (cheap, and all that plotting needs). history.flat is the raw array.
    """

    def __init__(self, template, n_steps, dtype=None):
        leaves, self.treedef = jax.tree.flatten(template)
        self.shapes = [x.shape for x in leaves]
        sizes = [x.size for x in leaves]
        self.splits = onp.cumsum(sizes)[:-1]
        if dtype is None:
            dtype = np.result_type(*[x.dtype for x in leaves])
        self._buf = onp.empty((n_steps, sum(sizes)), dtype=onp.dtype(dtype))
        self._n = 0

    def append(self, flat):
        self._buf[self._n] = flat
        self._n += 1

    @property
    def flat(self):
        return self._buf[: self._n]

    def _unravel(self, row):
        leaves = [x.reshape(s) for x, s in zip(onp.split(row, self.splits), self.shapes)]
        return self.treedef.unflatten(leaves)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._unravel(row) for row in self.flat[i]]
        return jax.tree.map(np.asarray, self._unravel(self.flat[i]))

    def __iter__(self):
        return (self._unravel(row) for row in self.flat)

    def __repr__(self):
        return f"ParamsHistory(n_steps={self._n}, n_params={self._buf.shape[1]})"


def _get_C(params, model, exposures, use_c, ones, precond_method, precond_kwargs, damping=1e-3):
    """Diagonal preconditioner pytree: `use_c` if given (pytree or flat vector), else
    from precond_cache.get_precond (with `damping`), with C = 1 for the params named in `ones`."""
    if use_c is not False and use_c is not None:
        if isinstance(use_c, dict):
            return use_c
        flat, unravel = ravel_pytree(params)
        assert np.shape(use_c) == flat.shape, "use_c must be a diagonal C pytree or flat vector"
        return unravel(np.asarray(use_c))

    from precond_cache import get_precond  # lazy: only needed when C is not supplied

    ones = [k for k in ones if k in params]
    sub = {k: v for k, v in params.items() if k not in ones}
    C = dict(get_precond(sub, exposures, model, method=precond_method, damping=damping,
                         **(precond_kwargs or {}))) if sub else {}
    C.update({k: jax.tree.map(np.ones_like, params[k]) for k in ones})
    return {k: C[k] for k in params}


def _prepare(model, exposures, mapped, precond_kwargs):
    """(loss exposures, precond_kwargs): unchanged unless mapped, then an ExposureBatch and mapped=True preconditioning."""
    if not mapped:
        return exposures, precond_kwargs
    return make_batch(exposures, model), {"mapped": True, **(precond_kwargs or {})}


def _run(params, model, exposures, optimisers, epochs, C, history_dtype, progress):
    """The optimisation loop. Returns (losses, ParamsHistory)."""
    params = _as_float(dict(params))
    C = jax.tree.map(lambda c, p: np.asarray(c, p.dtype), C, params)
    optim, state = get_optimiser_new(params, optimisers)
    update = _get_update(optim, params, state)

    history = ParamsHistory(params, epochs, history_dtype)
    losses = onp.full(epochs, onp.nan)
    pbar = tqdm(range(epochs), disable=not progress)

    def read(i, loss, flat):
        """Fetch step i to the host. Returns a stop reason if it is non-finite."""
        loss, flat = float(loss), onp.asarray(flat)
        if not (onp.isfinite(loss) and onp.all(onp.isfinite(flat))):
            return f"non-finite loss/params at step {i} (loss={loss})"
        losses[i] = loss
        history.append(flat)
        pbar.set_postfix(loss=f"{loss:.6g}", refresh=False)

    # Lag-1 logging: dispatch step i, then read back step i-1 (`pending`), so the host
    # never makes the device wait and the tiny transfers overlap with compute. `loss`
    # is the loss before the step and `flat` the params after it; any NaN grad
    # propagates into the updated params, so checking these two catches every bad step
    # without an extra sync. On a bad step or Ctrl-C we keep only the finite steps.
    pending, stop = None, None
    try:
        for i in pbar:
            loss, grads = _loss_grad(params, C, exposures, model)
            params, state, flat = update(grads, state, params)
            if pending is not None:
                stop = read(*pending)
                if stop:
                    break
            pending = (i, loss, flat)
        else:  # ran to completion: read the final step
            if pending is not None:
                read(*pending)
                pbar.refresh()
    except KeyboardInterrupt:
        stop = "interrupted"
        if pending is not None:  # the last dispatched step still completes
            try:
                read(*pending)
            except KeyboardInterrupt:
                pass

    if stop:
        warnings.warn(f"optimisation stopped after {len(history)}/{epochs} steps: {stop}; "
                      "returning results up to the last finite step.")
        losses = losses[: len(history)]
    return losses, history


def optimise_new_resolved(params, model, exposures, optimisers, epochs, use_c=False,
                          return_c=False, precond_method="gn_probe", precond_kwargs=None,
                          precond_resolved=False, history_dtype=None, progress=True, mapped=False):
    """Preconditioned first-order optimisation, with C = 1 for the 'resolved' params
    unless precond_resolved=True.

    C is a diagonal preconditioner pytree (same structure as params, or a flat vector).
    Pass `use_c=C`, or leave it False to compute / load it with `precond_cache.get_precond`.
    Batching is controlled via `precond_kwargs`, e.g. dict(chunk=16) for gn_exact.
    mapped=True: lax.map over stacked exposures instead of unrolling them, for the loss and the preconditioner
    (batching.py): flat compile time, ~half the memory, ~1.1-1.3x slower per step.
    Returns (losses, params_history[, C]); params_history is a ParamsHistory.
    """
    ones = () if precond_resolved else ("resolved",)
    loss_exposures, precond_kwargs = _prepare(model, exposures, mapped, precond_kwargs)
    C = _get_C(params, model, exposures, use_c, ones, precond_method, precond_kwargs)
    losses, history = _run(params, model, loss_exposures, optimisers, epochs, C, history_dtype, progress)
    return (losses, history, C) if return_c else (losses, history)


def optimise_new(params, model, exposures, optimisers, epochs, use_c=False, return_c=False,
                 precond_method="gn_probe", precond_kwargs=None, history_dtype=None,
                 progress=True, mapped=False):
    """As optimise_new_resolved, but preconditioning every param (including 'resolved')."""
    return optimise_new_resolved(params, model, exposures, optimisers, epochs, use_c=use_c,
                                 return_c=return_c, precond_method=precond_method,
                                 precond_kwargs=precond_kwargs, precond_resolved=True,
                                 history_dtype=history_dtype, progress=progress, mapped=mapped)


def optimise_optimistix(params, model, exposures, use_c=False, precond_method="gn_exact",
                        precond_kwargs=None, precond_damping=1e-3, max_steps=1024, rtol=1e-9,
                        atol=1e-2, history_length=10, step_init=None, verbose=False,
                        return_info=False, ones=(), mapped=False):
    """optimistix L-BFGS on u, with params = x0 + sqrt(C) * u and C the diagonal
    preconditioner (see optimise_new).

    C defaults to the exact Gauss-Newton diagonal (gn_exact, one jvp per parameter, cached
    by precond_cache; chunk=1 unless precond_kwargs say otherwise), so u is in units of ~1
    conditional sigma and `atol` is the step size (in sigma) at which to stop. Use
    precond_method="gn_probe" (precond_kwargs=dict(n_probes=...)) when n_params is too large
    for exact columns. L-BFGS is matrix-free: it keeps 2 * history_length vectors and applies
    the inverse Hessian by the two-loop recursion (use_inverse=True), so memory is set by the
    gradient (use the optics' remat / wl_batch options at production size).

    A non-finite model costs +inf (see stats.gauss_log_likelihood), so blow-ups are rejected. step_init < 1
    shortens the first line-search trial (default: the full -grad step). verbose=True prints
    the loss and step size each step from inside the compiled loop.

    Returns the optimised params dict (and an info dict if return_info). If the solve ends
    non-finite or worse than the start, warns and returns the start; if it did not converge,
    warns and returns the best point found. ones : params with C = 1 (e.g. ("resolved",), as
    optimise_new_resolved). mapped: as optimise_new_resolved.
    """
    if not jax.config.x64_enabled:
        warnings.warn("optimise_optimistix: x64 is disabled; L-BFGS curvature pairs lose precision in float32")

    x0 = _as_float(dict(params))
    if precond_kwargs is None and precond_method == "gn_exact":
        precond_kwargs = dict(chunk=1)       # lowest memory, same result
    loss_exposures, precond_kwargs = _prepare(model, exposures, mapped, precond_kwargs)
    C = _get_C(x0, model, exposures, use_c, ones, precond_method, precond_kwargs, precond_damping)
    S = jax.tree.map(lambda c, x: np.sqrt(np.asarray(c, x.dtype)), C, x0)
    frozen = [k for k, s in S.items() if any(bool(np.any(l == 0)) for l in jax.tree.leaves(s))]
    if frozen:
        warnings.warn(f"optimise_optimistix: C = 0 (no curvature) for entries of {frozen}; they will not move")

    # x0 and S are passed as args (not closed over) so they are traced, not baked in
    def scale(u, x0, S):
        return jax.tree.map(lambda x, s, v: x + s * v, x0, S, u)

    def scaled_loss(u, args):
        exposures, model, x0, S = args
        return loss_fn(ModelParams(scale(u, x0, S)), exposures, model)

    def printer(**kw):
        jax.debug.print("loss {f} (last accepted {fa}), step size {s}", f=kw["loss_this_step"][1],
                        fa=kw["loss_last_accepted_step"][1], s=kw["step_size"][1])

    lbfgs = optx.LBFGS(rtol=rtol, atol=atol, history_length=history_length,
                       verbose=printer if verbose else False)
    if step_init is not None:
        lbfgs = eqx.tree_at(lambda s: s.search, lbfgs, optx.BacktrackingArmijo(step_init=step_init))
    solver = optx.BestSoFarMinimiser(lbfgs)

    args = (loss_exposures, model, x0, S)
    u0 = jax.tree.map(np.zeros_like, x0)
    f0 = float(scaled_loss(u0, args))
    sol = optx.minimise(scaled_loss, solver, u0, args, max_steps=max_steps, throw=False)
    f1 = float(scaled_loss(sol.value, args))
    out = scale(sol.value, x0, S)
    info = dict(result=str(sol.result), num_steps=int(sol.stats["num_steps"]), loss_start=f0,
                loss_end=f1, frozen=frozen)

    finite = all(bool(np.all(np.isfinite(l))) for l in jax.tree.leaves(out))
    if not (finite and onp.isfinite(f1) and f1 <= f0):
        warnings.warn(f"optimise_optimistix failed ({info}); returning the starting params")
        out = x0
    elif sol.result != optx.RESULTS.successful:
        warnings.warn(f"optimise_optimistix did not converge ({info['result']} after {info['num_steps']} "
                      "steps); returning the best point found")
    return (out, info) if return_info else out
