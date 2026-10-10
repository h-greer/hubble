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

import interpax as ipx

# %%
wid = 80
oversample = 4

nwavels = 20
npoly=3

n_modes = 40
n_zernikes = 30

resolved_wid = 60#*2
regulariser = np.array([0.1, 1.])

optics = NICMOSCoronagraph(512, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'


vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))




exposures_single = [
    exposure_from_file(ddir + 'n93m23lmq_o_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid),
    exposure_from_file(ddir + 'n93m24lsq_o_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid),
]
params = {
    "spectrum": {},
    "primary_opd": {},
    "primary_low": {},
    "primary_tilt": {},
    "cold_mask_opd": {},
    "cold_mask_tilt": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "primary_rot": {},
    "cold_mask_shear": {},
    "cold_mask_scale": {},

    "bias": {},
    "occulter_radius": 0.7,
    "occulter_coeffs": np.zeros(2)+1,
    "fnumber": 45.7,

    "resolved": {},
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*3)

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.02535399,  0.01166608])
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([-0.14056529, -0.22439143])

    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.array([-15.86129843,  -0.71768201,  -1.42882413,  26.15460722,
         -20.11501062,  -2.76059106,  21.16134548,  10.25475808,
         -10.85443342, -11.49565824, -12.22853623,   4.40457998,
           5.28396953,  -1.2162837 ,  -8.06600783,  -5.98268676,
           4.60561808,   2.93092808,  -4.78088159,   4.79876309,
           0.52691283,   2.49798775,   5.04594252,  -0.24174374,
          -3.31476011,   3.4945766 ,  -8.59432607,   2.94284114,
           5.45766127,  -0.50770123])#np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.array([132.24425199])

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.12248021,  8.57885088])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 0.#-90.
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = 0.##-90.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 1.6

    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((resolved_wid,resolved_wid))#+3#.at[:3, :3].set(3.)-2
    


    # params["quadrature"][exp.fit.get_key(exp, "quadrature")] = np.array(0.)


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%

g = 5e-2

"""
    "spectrum": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*20, 40),
    
    "bias": sgd(g*3, 20),
    "primary_opd": sgd(g*0.1, 10),
    # "primary_opd": adam(0.1, 10),

    "cold_mask_opd": sgd(g*3, 10),

    "primary_tilt": sgd(g*1, 10),
    "cold_mask_tilt": sgd(g*1, 10),
    #"jitter": opt(g*1, 120),


    "cold_mask_shear": sgd(g*2, 200),
    "cold_mask_rot": sgd(g*3, 200),
    "cold_mask_scale": sgd(g*15, 200),

    # "quadrature": sgd(g*20, 400)

    "occulter_radius": sgd(g*10, 200),
    "fnumber": sgd(g*0.05, 220),

    "occulter_coeffs": sgd(g*20, 300),
"""

# things = {
#     "spectrum": sgd(g*3, 0),
#     "primary_opd": sgd(g*3, 20),
#     "primary_low": sgd(g*3, 10),
#     "primary_tilt": sgd(g*1., 10),
#     "cold_mask_tilt": sgd(g*1, 10),
#     "cold_mask_opd": sgd(g*1, 10),

#     "bias": sgd(g*3, 50),
#     "cold_mask_shift": sgd(g*20, 60),

#     # # "cold_mask_shear": sgd(g*2, 200),
#     # # "cold_mask_rot": sgd(g*3, 200),
#     # # "cold_mask_scale": sgd(g*15, 200),

#     # # # "quadrature": sgd(g*20, 400)

#     # # "occulter_radius": sgd(g*10, 200),
#     # # "fnumber": sgd(g*0.05, 220),

#     # "resolved": adam(3e-2, 100)
# }

# things_start = {
#     "positions": sgd(g*5, 0),
# }

things = {
    "primary_opd": sgd(g*0.02, 30),
    "spectrum": sgd(g*3, 0),
    "primary_tilt": sgd(g*3, 0),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1, 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0),
    "cold_mask_rot": sgd(g*1, 0),
    "primary_rot": sgd(g*1, 0),
    "primary_low": sgd(g*1, 0),
    "occulter_radius": sgd(g*3., 0),
    # "fnumber": sgd(g*2., 0),

    "resolved": adam(3e-2, 60)
}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*3, 15.),
    "cold_mask_tilt": sgd(g*0.7, 0),
    "cold_mask_opd": sgd(g*1, 40),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*1, 70),
    "cold_mask_rot": sgd(g*10, 70),
    "primary_rot": sgd(g*10, 70),
    "primary_low": sgd(g*0.1, 90),
    "occulter_radius": sgd(g*0.3, 130),
    # "occulter_coeffs": sgd(g*2, 200),
    # "fnumber": sgd(g*2., 150),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 500,
    precond_method="gn_exact", precond_kwargs=dict(chunk=16))

# %%
plt.plot(losses[:])

# %%
plot_params(params_history, list(things_start.keys()), xw = 3, save="aumic-intermediate-params")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, save="aumic-intermediate-comparison")


# %%
# Keep the stage-1 values of parameters that stage 2 does not fit (stage 2 injects into model_single)
model_single = ModelParams(params_history[-1]).inject(model_single)
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new_resolved(
    opt_params, model_single, exposures_single, things, 300,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32), precond_resolved=True)

# %%
plt.plot(losses[:])

# %%
losses[-1]

# %%
plot_params(params_history, groups, xw = 3, save="aumic-params")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, save="aumic-comparison", percentile=99)

# %%
params_history[-1]

# %%
plt.imshow(10**(params_history[-1]["resolved"]["GJ803_F160W"]))  
plt.colorbar()

plt.figure(figsize=(10,10))
plt.imshow(10**(params_history[-1]["resolved"]["GJ803_F160W"]))
plt.savefig("aumic.png")

plt.figure(figsize=(10,10))
plt.imshow((params_history[-1]["resolved"]["GJ803_F160W"]))
plt.savefig("aumic-log.png")



# %%



