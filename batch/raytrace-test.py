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
import optimistix as optx

# dLux imports
import dLux as dl
import dLux.utils as dlu

import pickle
from raytrace_jax import *


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
from newfisher import *
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
extra_bad = None
#extra_bad = np.isnan(np.zeros((64, 64)).at[35,60].set(np.nan))

#extra_bad = np.isnan(np.zeros((wid,wid))).at[wid//2-3:wid//2+3,:].set(np.nan)


# %%
wid = 64
oversample = 4

nwavels = 3
npoly=1

n_zernikes = 12#12

optics = NICMOSOptics(1024, wid, oversample, n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"

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

exposures_f108N = [exposure_from_file(ddir + file, BinaryFit(spectrum_basis, "F108N"), crop=wid) for file in files_f108N]
exposures_f190N = [exposure_from_file(ddir + file, BinaryFit(spectrum_basis, "F190N"), crop=wid) for file in files_f190N]
exposures_single = exposures_f108N+exposures_f190N

# %%
for e in exposures_single:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)

# %%


# %%
params = {
    "positions": {},
    "primary_spectrum": {},
    "secondary_spectrum": {},

    "aberrations": {},

    "separation": 2.,
    "position_angle": 110.,


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

    "softening": 2.,#0.1,
    "bias": {},
    "jitter": {},
    #"displacement": 1.#1e5#{}
}


for idx, exp in enumerate(exposures_single):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["primary_spectrum"][exp.fit.get_key(exp, "primary_spectrum")] = (np.zeros(npoly)).at[0].set(np.log10(np.nansum(exp.data)/nwavels))#np.array([3.2]) if exp.filter == "F108N" else np.array([3.5]) 
    params["secondary_spectrum"][exp.fit.get_key(exp, "secondary_spectrum")] = (np.zeros(npoly)).at[0].set(np.log10(np.nansum(exp.data)/nwavels)) - np.log10(6.5)

    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8.,8.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    #params["displacement"][exp.fit.get_key(exp, "displacement")] = 1e6

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
print(np.nansum(exp.data))

# %%
np.log10(7336/np.sum(10**spectrum_basis[:,0]))

# %%
print(params.params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%
plt.imshow(exposures_single[0].err)
plt.colorbar()

# %%
plt.imshow(np.log10(exposures_single[0].data/exposures_single[0].err))
plt.colorbar()

# %%
print(exposures_single[0].exptime)

# %%
plt.imshow(exposures_single[0].data**0.125)
plt.xticks([])
plt.yticks([])

# %%
cmap = matplotlib.colormaps['inferno']
cmap.set_bad('k',1)
plt.figure(figsize=(10,10))
plt.imshow(exposures_single[0].data**0.125, cmap=cmap)
plt.title(exposures_single[0].target)
plt.colorbar()


# %%
def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

base_sgd2 = lambda vals: optax.noisy_sgd(vals)

opts = lambda lr, start, *schedule: base_sgd2(scheduler(lr, start, *schedule))


base_adam = lambda vals: optax.adam(vals)
opta = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))





def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 5e-2

things = {
    "positions": opt(g*5, 0),
    "position_angle": opt(g*10, 10),
    "separation": opt(g*15, 20),
    "primary_spectrum": opt(g*5, 30),
    "secondary_spectrum": opt(g*3, 30),
    "cold_mask_shift": opt(g*10, 60),
    #"cold_mask_rot": opt(g*10, 100),
    "bias": opt(g*5, 50),
    "aberrations": opt(g*0.05, 80),

    #"cold_mask_scale": opt(g*1, 300),
    #"cold_mask_shear": opt(g*1, 300),
    #"primary_scale": opt(g*1, 300),
    #"primary_shear": opt(g*1, 300),
}

things_start = {
    "positions": opt(g*5, 0),
}

groups = list(things.keys())

# %%
initial_losses, initial_models = optimise(params, model_single, exposures_single, things_start, 10, recalculate=True)

# %%
plot_comparison(model_single, initial_models[-1], exposures_single)

# %%
initial_models[-1].params

# %%
losses, models = optimise(initial_models[-1].inject(params), initial_models[-1].inject(model_single), exposures_single, things, 5000, recalculate=True)

# %%
plt.plot(np.asarray(losses[-50:])/(len(exposures_single)*wid**2))

# %%
print(losses[0], losses[-1])

# %%
models_pd = [jax.tree.map(lambda x,y: (x-y)/y, models[i], models[-1]) for i in range(len(models))]

# %%
plot_params(models, groups, xw = 3)
plot_comparison(model_single, models[-1], exposures_single)

# %%
print(models[-1].params)

# %%
print(models[-1].params["primary_spectrum"])
print(models[-1].params["secondary_spectrum"])
print(models[-1].params["separation"]*42)
print(models[-1].params["position_angle"])


def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    res = np.sum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))
    return np.where(res==0.0, np.inf, res)

@eqx.filter_jit
def fun(params, args):
    exposures, model = args
    return loss_fn(params, exposures, model)

def optimise_optimistix(params, model, exposures, things, niter):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    model_params = ModelParams({p: params.get(p) for p in things.keys()})

    solver = optx.BFGS(rtol=1e-6, atol=1e-6,verbose=frozenset({"step_size", "loss"}))
    sol = optx.minimise(fun, solver, model_params, (exposures, model), throw=False, max_steps=niter)
    
    return sol

sol = optimise_optimistix(models[-1], models[-1].inject(model_single), exposures_single, things, 5000)
print(sol.value.params)
print(fun(sol.value, (exposures_single, model_single)), (losses[-1]))

best_params = sol.value

#initial_position = sol.value

def populate_fishers(fishers, exposures, model_params):

    # Build the lr model structure
    params_dict = jax.tree.map(lambda x: np.zeros((x.size, x.size)), model_params).params

    # Loop over exposures
    for exp in exposures:

        # Loop over parameters
        for param in model_params.keys():

            # Check if the fishers have values for this exposure
            key = f"{exp.key}.{param}"
            if key not in fishers.keys():
                continue

            # Add the Fisher matrices
            if isinstance(params_dict[param], dict):
                params_dict[param][exp.get_key(param)] += fishers[key]
            else:
                params_dict[param] += fishers[key]

    fisher_params = model_params.set("params", params_dict)

    return jtu.tree_map(lambda x: np.diag(x), fisher_params)

fsh = jax.flatten_util.ravel_pytree(populate_fishers(fishers, exposures_single, best_params))[0]
bpf = jax.flatten_util.ravel_pytree(best_params)[0]

initial_position = best_params.map(lambda x: np.zeros_like(x))

pms, unflat = jax.flatten_util.ravel_pytree(initial_position)

def loss_fn(params, exposures, model):
    params_transformed = params / fsh + bpf
    mdl = unflat(params_transformed).inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

loglike = lambda params: -loss_fn(params, exposures_single, models[-1].inject(model_single))

rng_key = jr.key(0)

samples = sample_raytrace(key=rng_key, params_init=pms, \
    log_prob_fn=loglike, n_steps=10000, n_leapfrog_steps=10, \
    step_size=3e-1, refresh_rate=0.0, metro_check=1, sample_hmc=False)


with open("raytrace-chains.pickle", 'wb') as file:
    pickle.dump(samples, file)

with open("raytrace-chains-unflat.pickle", 'wb') as file:
    pickle.dump([unflat(x/fsh + bpf).params for x in samples[0]], file)