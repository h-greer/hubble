import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx
import optax

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

def optimise(params, model, exposures, things, niter, reduce_ram=False, recalculate=False):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    print("Calculating Fishers")

    fishers = calc_fishers(model, exposures, paths, fisher_fn, recalculate=recalculate)
    print(fishers)
    model_params = ModelParams({p: model.get(p) for p in things.keys()})
    lrs = populate_lr_model(fishers, exposures, model_params)
    

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

