# %%
import sys
sys.path.insert(0, '..')

# %%

import os
os.environ["JAX_PLATFORMS"] = "cpu"

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

import glob

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
import interpax as ipx

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

# %%
wid = 64
oversample = 4

nwavels = 3#13#6
npoly=1#10#2

n_zernikes = 20#30#12

optics = NICMOSSecondaryFresnelOptics(512, wid, oversample, mag=3.3, defocus=0., despace=0., n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))

# %%

number = int(sys.argv[1])


ddir = "../data/MAST_2025-06-24T0210/HST/"

cfiles = glob.glob(ddir+"*_cal.fits")
cfiles.sort()

if number >= len(cfiles):
    exit(0)

fname = cfiles[number]

filt = fits.getheader(fname, ext=0)['FILTER']

spectrum_basis = np.ones((1, 3))

exposures_raw = [exposure_from_file(fname, SinglePointFit(spectrum_basis, filt, time_series=True), crop=wid)]


exposures_single = []
for exp in exposures_raw:
    if exp.data.shape == (wid, wid):
        exposures_single.append(exp)

# %%
for e in exposures_single:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)
    print(e.exptime)

# %%
params = {
    #"fluxes": {},
    "positions": {},
    "spectrum": {},
    "aberrations": {},

    #"rot": 0.,

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

    "softening": 10.,
    "bias": {},
    "jitter": {},

    "defocus": {},
    "despace": {},
    "mag": 3.3,
    "quadrature": {},
}

positions = [[0.,0.,],[0.,0.,],[0.,0.,],[0.,0.,]]#[[0.43251792, 0.33013815],[ 0.49417186, -0.5629123 ]]


for idx, exp in enumerate(exposures_single):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = 1.7165396

    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.array([ 12.66013155,  -3.74036652, -20.8438651 ,   1.09282138,
           3.48634449,  -8.31898117,  22.48192372,   1.41053358,
          -2.31421321,   4.85990522,  -6.0902642 ,  14.10482417,
          -9.43575842,   1.14218497,   1.77468617,   1.81254211,
          -2.2649688 ,  -0.45356718,  -3.61981819,   0.74366754])

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8.8,8.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 0.233
    params["despace"][exp.fit.get_key(exp, "despace")] = 0.#-5.#0.
    

    params["bias"][exp.fit.get_key(exp, "bias")] = 2.85
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample

    params["quadrature"][exp.fit.get_key(exp, "quadrature")] = np.array(0.)


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)


g = 5e-2

things = {
    "positions": sgd(g*2.5, 0),
    "spectrum": sgd(g*3, 10),
    "cold_mask_shift": sgd(g*2, 30),
    
    "bias": sgd(g*3, 20),
    "aberrations": sgd(g*2, 70),
    #"jitter": opt(g*1, 120),

    "despace": sgd(g*10, 50),
    "mag": sgd(g*3, 100),

    "cold_mask_shear": sgd(g*2, 100),

    "quadrature": sgd(g*20, 400)
}

things_start = {
    "positions": sgd(g*5, 0),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 10)

# %%
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single)

# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things, 500, nbatches=len(exposures_single))

# %%
plt.plot(losses[:])

# %%
plot_params(params_history, groups, xw = 3)
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=True)

# %%
final_params = optimise_optimistix(params_history[-1], model_single, exposures_single)

# %%
plot_comparison(final_params.inject((model_single)), final_params, exposures_single, quadrature=True)

# %%
print(final_params.params)


f = lambda params: loss_fn(ModelParams(params), exposures_single, final_params.inject((model_single)))
F, unflatten = zdx.batching.hessian(f, final_params, nbatches=5*len(exposures_single))

numpy.savez(f"breathing-data/{exposures_single[0].key}.npz", mjd=exp.mjd, params=final_params.params, fisher=F)