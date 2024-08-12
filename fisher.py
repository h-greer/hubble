import jax
import zodiax as zdx
import equinox as eqx
import jax.numpy as np
from jax import jit, grad, jvp, linearize, lax, vmap
import jax.tree_util as jtu
import os
from tqdm.auto import tqdm
from stats import posterior

import dLux as dl
import dLux.utils as dlu


def fisher_fn(model, exposure, params, new_diag=False):
    return FIM(model, params, posterior, exposure, new_diag=new_diag)


def self_fisher_fn(model, exposure, params, new_diag=False):
    return fisher_fn(model, exposure, params, new_diag=new_diag)

def calc_fisher(
    model,
    exposure,
    param,
    file_path,
    recalculate=False,
    save=True,
    overwrite=False,
    new_diag=False,
):
    # Check that the param exists - caught later
    try:
        leaf = model.get(exposure.map_param(param))
        if not isinstance(leaf, np.ndarray):
            raise ValueError(f"Leaf at path '{param}' is not an array")
        N = leaf.size
    except ValueError as e:
        # Param doesn't exist, return None
        print(e)
        return None

    # Check for cached fisher mats
    exists = os.path.exists(file_path)

    # Check if we need to recalculate
    if exists and not recalculate:
        fisher = np.load(file_path)
        if fisher.shape[0] != N:

            # Overwrite shape miss-matches
            if overwrite:
                fisher = self_fisher_fn(model, exposure, [param], new_diag=new_diag)
                if save:
                    np.save(file_path, fisher)
            else:
                raise ValueError(f"Shape mismatch for {param}")

    # Calculate and save
    else:
        fisher = self_fisher_fn(model, exposure, [param], new_diag=new_diag)
        if save:
            np.save(file_path, fisher)
    print(fisher)
    return fisher



def calc_fishers(
    model,
    exposures,
    parameters,
    param_map_fn=None,
    recalculate=False,
    overwrite=False,
    save=True,
    new_diag=False,
    cache="files/fishers",
):

    if not os.path.exists(cache):
        os.makedirs(cache)

    # Iterate over exposures
    fisher_exposures = {}
    for exp in tqdm(exposures):

        # Iterate over params
        fisher_params = {}
        looper = tqdm(range(0, len(parameters)), leave=False, desc="")
        for idx in looper:
            param = parameters[idx]
            looper.set_description(param)

            # Ensure the path to save to exists
            save_path = f"{cache}/{exp.filename}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Path to the file
            file_path = os.path.join(save_path, f"{param}.npy")

            # Get path correct for parameters
            # param_path = key_mapper(model, exp, param)
            param_path = exp.map_param(param)

            # Allows for custom mapping of parameters
            if param_map_fn is not None:
                param_path = param_map_fn(model, exp, param)

            # Calculate fisher for each exposure
            fisher = calc_fisher(
                model, exp, param_path, file_path, recalculate, save, overwrite, new_diag
            )

            # Store the fisher
            if fisher is not None:
                fisher_params[param] = fisher
            else:
                print(f"Could not calculate fisher for {param_path} - {exp.key}")

        fisher_exposures[exp.key] = fisher_params

    return fisher_exposures


def hessian_diag(fn, x):
    """Source: https://github.com/google/jax/issues/924"""
    eye = np.eye(len(x))
    return np.array(
        [jvp(lambda x: jvp(fn, (x,), (eye[i],))[1], (x,), (eye[i],))[1] for i in range(len(x))]
    )


def hessian(f, x, fast=False):
    if fast:
        # print("Running Vmapped")
        # I think this basically just returns np.eye?
        basis = np.eye(x.size).reshape(-1, *x.shape)

        _, hvp = linearize(grad(f), x)
        hvp = jit(hvp)

        # Compile on first input
        # TODO: I Think this needs to be re-worked so that we call the vmapped jit fn,
        # not the function, then the vmapped version. ie
        # hvp = vmap(jit(hvp))
        # first = hvp(np.array([(basis[0])]))
        first = np.array([hvp(basis[0])])  # Add empty dim for concatenation

        # Vmap others
        others = vmap(hvp)(basis[1:])

        # Recombine
        return np.stack(np.concatenate([first, others], axis=0)).reshape(x.shape + x.shape)
    else:
        # print("Running non-vmapped")
        _, hvp = linearize(grad(f), x)
        # Jit the sub-function here since it is called many times
        # TODO: Test effect on speed
        hvp = jit(hvp)
        # basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
        basis = np.eye(x.size).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


"""
Some code adapted from here: 
https://github.com/google/jax/issues/3801#issuecomment-662131006

More resources:
https://github.com/google/jax/discussions/8456

I believe this efficient hessian diagonal methods only works _correctly_ if the output
hessian is _naturally_ diagonal, else the results are spurious.
"""


def hvp(f, x, v):
    return jvp(grad(f), (x,), (v,))[1]


# TODO: Update this to take the matrix mapper class
def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    shape_dict={},
    save_ram=True,
    vmapped=False,
    diag=False,
    new_diag=False,
    **loglike_kwargs,
):
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    # Build function to calculate FIM and calculate
    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    if diag:
        diag = hvp(loglike_fn_vec, X, np.ones_like(X))
        return np.eye(diag.shape[0]) * diag[:, None]

    if new_diag:
        return hessian_diag(loglike_fn_vec, X)

    if save_ram:
        return hessian(loglike_fn_vec, X)

    if vmapped:
        return hessian(loglike_fn_vec, X, fast=True)

    return jax.hessian(loglike_fn_vec)(X)


def _perturb(X, pytree, parameters, shapes, lengths):
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)





def calc_lrs(model, exposures, fishers, params=None, order=1):
    # Get the parameters from the fishers
    if params is None:
        params = []
        for exp_key, fisher_dict in fishers.items():
            for param in fisher_dict.keys():
                params.append(param)
        params = list(set(params))

    # Build a filter, we need to handle parameters that are stored in dicts
    # TODO: Add this to model?
    bool_model = jtu.tree_map(lambda _: False, model)
    for param in params:
        leaf = model.get(param)
        if isinstance(leaf, dict):
            true_leaf = jtu.tree_map(lambda x: True, leaf)
        else:
            true_leaf = True
        bool_model = bool_model.set(param, true_leaf)

    # Make an empty fisher model
    # Flag and deal with large arrays
    grad_model = eqx.filter(model, bool_model)
    is_large = jtu.tree_map(lambda x: x.size > 1e4, grad_model)
    bool_model = jtu.tree_map(lambda x, y: x and not y, bool_model, is_large)
    grad_model = eqx.filter(model, bool_model)
    fisher_model = jtu.tree_map(lambda x: np.zeros((x.size, x.size)), grad_model)
    large_grad_model = eqx.filter(model, is_large)
    large_lr_model = jtu.tree_map(lambda x: np.ones(x.shape), large_grad_model)

    # Loop over exposures
    for exp in exposures:

        # Loop over parameters
        for param in params:

            # Check if the parameter is in the fisher
            if param not in fishers[exp.key].keys():
                continue

            param_path = exp.map_param(param)
            fisher_model = fisher_model.add(param_path, fishers[exp.key][param])

    # Convert fisher to lr model
    inv_fn = lambda fmat, leaf: dlu.nandiv(-1, np.diag(fmat), 1).reshape(leaf.shape)
    lr_model = jtu.tree_map(inv_fn, fisher_model, model)
    lr_model = eqx.combine(lr_model, large_lr_model)
    return lr_model