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


def loss_fn(params, exposures, model):
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


def _get_C(params, model, exposures, use_c, ones, precond_method, precond_kwargs):
    """Diagonal preconditioner pytree: `use_c` if given (pytree or flat vector), else
    from precond_cache.get_precond, with C = 1 for the params named in `ones`."""
    if use_c is not False and use_c is not None:
        if isinstance(use_c, dict):
            return use_c
        flat, unravel = ravel_pytree(params)
        assert np.shape(use_c) == flat.shape, "use_c must be a diagonal C pytree or flat vector"
        return unravel(np.asarray(use_c))

    from precond_cache import get_precond  # lazy: only needed when C is not supplied

    ones = [k for k in ones if k in params]
    sub = {k: v for k, v in params.items() if k not in ones}
    C = dict(get_precond(sub, exposures, model, method=precond_method, **(precond_kwargs or {}))) if sub else {}
    C.update({k: jax.tree.map(np.ones_like, params[k]) for k in ones})
    return {k: C[k] for k in params}


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
                          precond_resolved=False, history_dtype=None, progress=True):
    """Preconditioned first-order optimisation, with C = 1 for the 'resolved' params
    unless precond_resolved=True.

    C is a diagonal preconditioner pytree (same structure as params, or a flat vector).
    Pass `use_c=C`, or leave it False to compute / load it with `precond_cache.get_precond`.
    Batching is controlled via `precond_kwargs`, e.g. dict(chunk=16) for gn_exact.
    Returns (losses, params_history[, C]); params_history is a ParamsHistory.
    """
    ones = () if precond_resolved else ("resolved",)
    C = _get_C(params, model, exposures, use_c, ones, precond_method, precond_kwargs)
    losses, history = _run(params, model, exposures, optimisers, epochs, C, history_dtype, progress)
    return (losses, history, C) if return_c else (losses, history)


def optimise_new(params, model, exposures, optimisers, epochs, use_c=False, return_c=False,
                 precond_method="gn_probe", precond_kwargs=None, history_dtype=None,
                 progress=True):
    """As optimise_new_resolved, but preconditioning every param (including 'resolved')."""
    return optimise_new_resolved(params, model, exposures, optimisers, epochs, use_c=use_c,
                                 return_c=return_c, precond_method=precond_method,
                                 precond_kwargs=precond_kwargs, precond_resolved=True,
                                 history_dtype=history_dtype, progress=progress)


def optimise_optimistix(params, model, exposures, use_c=False, precond_method="gn_probe",
                        precond_kwargs=None, max_steps=1024, rtol=1e-6, atol=1e-6):
    """optimistix LBFGS on u, with params = x0 + sqrt(C) * u and C the diagonal
    preconditioner (see optimise_new). Returns the optimised params dict."""
    x0 = _as_float(dict(params))
    C = _get_C(x0, model, exposures, use_c, (), precond_method, precond_kwargs)
    S = jax.tree.map(lambda c, x: np.sqrt(np.asarray(c, x.dtype)), C, x0)
    # x0 and S are passed as args (not closed over) so they are traced, not baked in
    def scale(u, x0, S):
        return jax.tree.map(lambda x, s, v: x + s * v, x0, S, u)

    def scaled_loss(u, args):
        exposures, model, x0, S = args
        return loss_fn(ModelParams(scale(u, x0, S)), exposures, model)

    solver = optx.BestSoFarMinimiser(optx.LBFGS(rtol=rtol, atol=atol))
    sol = optx.minimise(scaled_loss, solver, jax.tree.map(np.zeros_like, x0), (exposures, model, x0, S),
                        max_steps=max_steps, throw=False)
    return scale(sol.value, x0, S)
