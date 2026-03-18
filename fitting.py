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

from apertures import *
from detectors import *
from spectra import *
from models import *
from stats import *
from fisher import *

def get_optimiser_new(model_params, optimisers):
    param_spec = ModelParams({param: param for param in model_params.keys()})
    optim = optax.multi_transform(optimisers, param_spec)
    return optim, optim.init(model_params)

def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

def optimise_optimistix(params, model, exposures, project=True, diag=False):
    if project:
        f = lambda params: loss_fn(params, exposures, model)
        F, unflatten = zdx.batching.hessian(f, ModelParams(params), nbatches=len(exposures)*5, checkpoint=True)
        if diag:
            F = np.diag(np.diag(F))
            

    def projected_loss_fn(u, args):
        exposures, model, project_fn = args
        params = project_fn(u)
        return loss_fn(params, exposures, model)

    # Estimate our initial parameters from the data
    params = ModelParams(params)
    X0, unravel = ravel_pytree(params)

    # Generate the projection matrix P, projection function, and initial vector
    P = zdx.optimisation.eigen_projection(fmat=F) if project else np.eye(X0.shape[0])
    project_fn = lambda u: unravel(X0 + np.dot(P, u))
    X = np.zeros(P.shape[-1])


    # Minimise algorithm
    args = (exposures, model, project_fn)
    solver = optx.BestSoFarMinimiser(optx.LBFGS(rtol=1e-6, atol=1e-6))
    sol = optx.minimise(projected_loss_fn, solver, X, args, max_steps=1024, throw=False)
    return project_fn(sol.value)

def optimise_no_norm(params, model, exposures, optimisers, epochs):
    optim, state = opt.map_optimisers(params, optimisers)

    loss_grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(lambda params, exposures, model: loss_fn(ModelParams(params), exposures, model)))

    pbar = tqdm(range(epochs))
    losses, params_history = [], []
    for step in pbar:
        loss, grads = loss_grad_fn(params, exposures, model)

        updates, state = optim.update(grads, state)
        params = optax.apply_updates(params, updates)
        pbar.set_postfix(log_loss=f"{np.log10(loss):.4f}")
        losses.append(loss)
        params_history.append(params)
    losses = np.array(losses)

    return losses, params_history


def optimise_new(params, model, exposures, optimisers, epochs, diag=True, nbatches=1):
    f = lambda params: loss_fn(ModelParams(params), exposures, model)
    F, unflatten = zdx.batching.hessian(f, params, nbatches=nbatches, checkpoint=True)

    if diag:
        C = dlu.nandiv(1, np.abs(np.diag(np.diag(F))), fill=0.)
        print(np.diag(C))
    else:
        C = np.linalg.inv(F)
        
    optim, state = opt.map_optimisers(params, optimisers)

    loss_grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(lambda params, exposures, model: loss_fn(ModelParams(params), exposures, model)))

    pbar = tqdm(range(epochs))
    losses, params_history = [], []
    for step in pbar:
        loss, grads = loss_grad_fn(params, exposures, model)

        # Normalise the gradients by the fisher matrix to get a natural gradient step
        G, unflatten = ravel_pytree(grads)
        grads = unflatten(np.dot(G, C))

        updates, state = optim.update(grads, state)
        params = optax.apply_updates(params, updates)
        pbar.set_postfix(log_loss=f"{np.log10(loss):.4f}")
        losses.append(loss)
        params_history.append(params)
    losses = np.array(losses)

    return losses, params_history


def optimise(params, model, exposures, things, niter, reduce_ram=False, recalculate=False):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    print("Calculating Fishers")

    fishers = calc_fishers(model, exposures, paths, fisher_fn, recalculate=recalculate)
    model_params = ModelParams({p: model.get(p) for p in things.keys()})
    lrs = populate_lr_model(fishers, exposures, model_params)

    print(ravel_pytree(lrs))
    

    optim, opt_state = get_optimiser_new(
        model_params, things
    )

    jit_loss = zdx.filter_value_and_grad(paths)(loss_fn)

    print("Fitting Model")

    @zdx.filter_jit
    def update(model_params, exposures, model, lrs, opt_state):
        grads = jax.tree.map(lambda x: x * 0.0, model_params)

        loss, new_grads = jit_loss(model_params,exposures, model)
        grads += new_grads
        grads = jax.tree.map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs.params))
        updates, opt_state = optim.update(grads, opt_state)
        model_params = zdx.apply_updates(model_params, updates)
        return loss, model_params, opt_state



    losses, models = [], []
    for i in tqdm(range(niter)):
        loss, model_params, opt_state = update(model_params, exposures, model, lrs, opt_state)
        models.append(model_params)
        losses.append(loss)

    
    return losses, models

def optimise_without_fisher(params, model, exposures, things, niter):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    model_params = ModelParams({p: model.get(p) for p in things.keys()})



    optim, opt_state = get_optimiser_new(
        model_params, things
    )

    jit_loss = zdx.filter_value_and_grad(paths)(loss_fn)

    print("Fitting Model")

    @zdx.filter_jit
    def update(model_params, exposures, model, opt_state):
        grads = jax.tree.map(lambda x: x * 0.0, model_params)

        loss, new_grads = jit_loss(model_params,exposures, model)
        grads += new_grads
        #grads = jax.tree.map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs.params))
        updates, opt_state = optim.update(grads, opt_state)
        model_params = zdx.apply_updates(model_params, updates)
        return loss, model_params, opt_state



    losses, models = [], []
    for i in tqdm(range(niter)):
        loss, model_params, opt_state = update(model_params, exposures, model, opt_state)
        models.append(model_params)
        losses.append(loss)


    
    return losses, models

