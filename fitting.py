import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from apertures import *
from detectors import *
from spectra import *
from models import *
from stats import *
from fisher import *


def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

def optimise(params, model, exposures, things, niter):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    print("Calculating Fishers")

    fishers = calc_fishers(model, exposures, paths, recalculate=True)
    lrs = calc_lrs(model, exposures, fishers, paths)

    #print(fishers)

    optim, opt_state = zdx.get_optimiser(
        params, paths, optimisers
    )

    jit_loss = zdx.filter_jit(zdx.filter_value_and_grad(paths)(loss_fn))

    print("Fitting Model")

    losses, models = [], []
    for i in tqdm(range(niter)):
        loss, grads = jit_loss(params,exposures, model)
        grads = jtu.tree_map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs.params))
        updates, opt_state = optim.update(grads, opt_state)
        params = zdx.apply_updates(params, updates)

        models.append(params)
        losses.append(loss)
    
    return losses, models

