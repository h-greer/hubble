# %%
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import jax
import numpy

jax.config.update("jax_enable_x64", True)


# Optimisation imports
import zodiax as zdx
import optax

import blackjax
import pickle


# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

from detectors import *
from apertures import *
from models import *
from fisher import *
from stats import posterior
from fitting import *
from plotting import *
from spectra import *

import jax.tree_util as jtu

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

# %%
wid = 64
oversample = 4

nwavels = 3
npoly=1

n_zernikes = 12

optics = NICMOSFresnelOptics(512, wid, oversample, n_zernikes = n_zernikes, defocus=0., fnumber=80.)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))


ddir = "../data/MAST_2026-02-02T23_18_00.739Z/HST/"

files_f190N = """
n8o101a1q_cal.fits
n8o101a2q_cal.fits
n8o101a3q_cal.fits
""".split()[:1]


files_f108N = """
n8o101adq_cal.fits
n8o101atq_cal.fits
n8o101b9q_cal.fits
""".split()[:1]


files_f164N = """
n8o101a8q_cal.fits
n8o101a9q_cal.fits
n8o101aaq_cal.fits
""".split()[:1]

exposures_binary_f108N = [exposure_from_file(ddir + file, BinaryFit(spectrum_basis, "F108N"), crop=wid) for file in files_f108N]
exposures_binary_f190N = [exposure_from_file(ddir + file, BinaryFit(spectrum_basis, "F190N"), crop=wid) for file in files_f190N]
exposures_binary_f164N = [exposure_from_file(ddir + file, BinaryFit(spectrum_basis, "F164N"), crop=wid) for file in files_f164N]

exposures_binary = exposures_binary_f190N+exposures_binary_f108N+exposures_binary_f164N

exposures_single_f108N = [exposure_from_file(ddir + file, SinglePointFit(spectrum_basis, "F108N"), crop=wid) for file in files_f108N]
exposures_single_f190N = [exposure_from_file(ddir + file, SinglePointFit(spectrum_basis, "F190N"), crop=wid) for file in files_f190N]
exposures_single_f164N = [exposure_from_file(ddir + file, SinglePointFit(spectrum_basis, "F164N"), crop=wid) for file in files_f164N]

exposures_single = exposures_single_f190N+exposures_single_f108N+exposures_single_f164N

# %%
for e in exposures_binary:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)
    print(e.exptime)


# %%
params = {
    "positions": {},
    "primary_spectrum": {},
    "secondary_spectrum": {},

    "aberrations": {},

    "separation": 2.,
    "position_angle": 110., #-180.,


    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_scale": {},
    "cold_mask_shear": {},
    "primary_scale": {},
    "primary_rot": {},
    "primary_shear": {},
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
    "scale": 0.043142,

    "softening": 10.,
    "bias": {},
    "jitter": {},

    "defocus": {},
    "fnumber": 78.75,
    "quadrature": {},
}


for idx, exp in enumerate(exposures_binary):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["primary_spectrum"][exp.fit.get_key(exp, "primary_spectrum")] = np.zeros(npoly).at[0].set(np.log10(np.nansum(exp.data)/nwavels))
    params["secondary_spectrum"][exp.fit.get_key(exp, "secondary_spectrum")] = np.array([3.2-np.log10(6.3)]) if exp.filter == "F108N" else np.array([3.5-np.log10(6.9)])

    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)#+1.

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8.,8.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 0.0

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample

    params["quadrature"][exp.fit.get_key(exp, "quadrature")] = 0.


model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params_binary = ModelParams(params)



# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)


g = 5e-2


things_binary = {
    "positions": sgd(g*2.5, 0),
    "position_angle": sgd(g*10, 10),
    "separation": sgd(g*15, 20),
    "primary_spectrum": sgd(g*5, 30),
    "secondary_spectrum": sgd(g*3, 30),

    "cold_mask_shift": sgd(g*20, 50),
    
    "bias": sgd(g*3, 40),
    "aberrations": sgd(g*0.2, 90),
    #"jitter": sgd(g*1, 120),

    "defocus": sgd(g*6, 70),
    "fnumber": sgd(g*3, 110),
    "cold_mask_shear": sgd(g*2, 110),

    "quadrature": sgd(g*20, 900)
}


things_start = {
    "positions": sgd(g*5, 0),
}


# %%
orig_params = params_binary.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(opt_params, model_binary, exposures_binary, things_start, 10)

# %%
plot_comparison(model_binary, ModelParams(params_history[-1]), exposures_binary)

# %%
orig_params = params_binary.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_binary})

# %%
losses, params_history = optimise_new(opt_params, model_binary, exposures_binary, things_binary, 1000, nbatches=len(exposures_single)*5)

# %%
plot_params(params_history, list(things_binary.keys()), xw = 4, save="gl164-params")
plot_comparison(model_binary, ModelParams(params_history[-1]), exposures_binary)

# %%
final_params_binary = optimise_optimistix(params_history[-1], model_binary, exposures_binary)

print(final_params_binary.params)

# %%
plot_comparison(final_params_binary.inject((model_binary)), final_params_binary, exposures_binary)



def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))


f = lambda params: loss_fn(ModelParams(params), exposures_binary, model_binary)  
F, unflatten = zdx.batching.hessian(f, final_params_binary, nbatches=len(exposures_binary)*5, checkpoint=True)

def projected_loss_fn(u, args):
    exposures, model, project_fn = args
    params = project_fn(u)
    return loss_fn(ModelParams(params), exposures, model)

# Estimate our initial parameters from the data
params = ModelParams(final_params_binary)
X0, unravel = ravel_pytree(final_params_binary)

# Generate the projection matrix P, projection function, and initial vector
P = zdx.optimisation.eigen_projection(fmat=F)
project_fn = lambda u: unravel(X0 + np.dot(P, u))
X = np.zeros(P.shape[-1])



# wrap all the extra parameters into a dedicated log likelihood
loglike = lambda params: -projected_loss_fn(params, (exposures_binary, model_binary, project_fn))

# inference loop from Blackjax docs
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

# initialise at the centre of our transformed distribution
initial_position = np.zeros_like(X0)

rng_key = jr.key(0)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

#  perform a warmup to adapt the mass matrix
warmup = blackjax.window_adaptation(blackjax.nuts, loglike, progress_bar=True)
(state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=5000)

# run inference with the known mass matrix
kernel = blackjax.nuts(loglike, **parameters).step
states = inference_loop(sample_key, kernel, state, 5000)

# extract samples, blocking avoids lazy evaluation for timing purposes
blackjax_samples = states.position.block_until_ready()







import itertools
import pandas as pd
from chainconsumer import ChainConsumer, Chain, Truth

# Helper function to get the scalar parameter names from the tree structure of the parameters
def scalar_names_from_tree(tree):
    names = []
    for path, leaf in jax.tree.leaves_with_path(tree):
        base = "_".join([str(p.key if hasattr(p, "key") else p) for p in path])
        leaf = np.asarray(leaf)
        if leaf.ndim == 0:
            names.append(base)
        else:
            for idx in itertools.product(*[range(s) for s in leaf.shape]):
                names.append(f"{base}_{'_'.join(map(str, idx))}")
    return names


# Unpack the latent samples into a dataframe

# Project latent samples -> original parameter space
samples_dict = eqx.filter_vmap(project_fn)(blackjax_samples)
flat_samples = jax.vmap(lambda p: ravel_pytree(p)[0])(samples_dict)


# Get the parameter names for the dataframe columns and the truth dict
param_names = scalar_names_from_tree(final_params_binary)
mcmc_df = pd.DataFrame(np.asarray(flat_samples), columns=param_names)

mcmc_df.to_pickle("gl164-chains.pickle")

# Build chains
mcmc_chain = Chain(samples=mcmc_df, name="MCMC posterior")

plt.figure()
# Plot
c = ChainConsumer()
c.add_chain(mcmc_chain)
fig = c.plotter.plot()

plt.savefig("gl164-hmc.png")