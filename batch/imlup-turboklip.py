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

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import chainconsumer as cc


plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

from detectors import *
from apertures import *
from models import *
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
wid = 80
oversample = 4

nwavels = 20
npoly=8

n_modes = 40
n_zernikes = 50

resolved_wid = 60

optics = NICMOSCoronagraph(512, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, turboklip="../data/turboklip.npz")

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))

ddir = '../data/data/'
flatdir = '../data/NICMOS-LAPL-DD2/LAPL_HOLEFLATS_DD2/'


vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))


regulariser = np.array([0.1,0.9])

exposures_single = [
    exposure_from_file(ddir + 'n8zu11epq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid, flatcorr=flatdir),
    # exposure_from_file(ddir + 'n8zu11eqq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid), crop=wid),
    exposure_from_file(ddir + 'n8zu12exq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid, flatcorr=flatdir),
    # exposure_from_file(ddir + 'n8zu12eyq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid), crop=wid),
]

params = {
    "spectrum": {},
    "primary_klip": {},
    "primary_low": {},
    "primary_tilt": {},
    "cold_mask_tilt": {},
    "bias": {},
    "resolved": {},
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*4)

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.75, 0.05])*0.075
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([
        np.array(exp.hdr["TARSIAFX"],dtype=float) - (256-181), 
        np.array(exp.hdr["TARSIAFY"],dtype=float) - (256-44)
    ])*0.075

    params["primary_klip"][exp.fit.get_key(exp, "primary_klip")] = np.zeros(7)
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros(n_zernikes)

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((resolved_wid, resolved_wid))


params = params | np.load("../data/optical_params.npy", allow_pickle=True)[()]


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)


# %%
# plot_comparison(model_single, params, exposures_single)

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)

def adam(lr, delay):
    return optax.adam(zdx.optimisation.delay(lr, delay))


g = 5e-2


things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*1, 15.),
    "cold_mask_tilt": sgd(g*0.1, 0),

    "bias": sgd(g*3, 40),
    "cold_mask_shift": sgd(g*0.1, 50),

    "primary_low": sgd(g*1., 70),
    "cold_mask_opd": sgd(g*1., 70),
    "primary_klip": sgd(g*2, 100),

    "resolved": adam(3e-2, 150),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 400, nbatches=10)


# %%
plot_params(params_history, groups, xw = 3, save="imlup-params")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, save="imlup-comparison", percentile=99)

# %%
params_history[-1]

# %%
plt.imshow(10**(params_history[-1]["resolved"]["SZ-82_F160W"]))  
plt.colorbar()

np.save("imlup.npy", params_history[-1]["resolved"]["SZ-82_F160W"])

plt.figure(figsize=(10,10))
plt.imshow(10**(params_history[-1]["resolved"]["SZ-82_F160W"]))
plt.savefig("imlup.png")

plt.figure(figsize=(10,10))
plt.imshow((params_history[-1]["resolved"]["SZ-82_F160W"]))
plt.savefig("imlup-log.png")