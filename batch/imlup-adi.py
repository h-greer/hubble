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
npoly=5

n_modes = 45
n_zernikes = 30

resolved_wid = 60#*2
regulariser = np.array([0.02, 0.4])

optics = NICMOSCoronagraph(512, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))

ddir = '../data/data/'


vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

flatdir = '../data/NICMOS-LAPL-DD2/LAPL_HOLEFLATS_DD2/'



exposures_single = [
    exposure_from_file(ddir + 'n8zu11epq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid, flatcorr=flatdir),
    # exposure_from_file(ddir + 'n8zu11eqq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid),
    exposure_from_file(ddir + 'n8zu12exq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid,flatcorr=flatdir),
    # exposure_from_file(ddir + 'n8zu12eyq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid, regulariser=regulariser), crop=wid),
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
    "cold_mask_shear": {},
    "cold_mask_scale": {},
    "primary_rot": {},
    "primary_shear": {},

    "bias": {},
    "occulter_radius": 0.7,
    "occulter_coeffs": np.zeros(2)+1,
    "fnumber": 45.7,
    "anisotropy": 1.+1e-5,

    "outer_radius": 1.2*0.9768,
    "secondary_radius": 0.357*1.2,
    "spider_width": 0.072*1.2,
    "primary_spider": 0.0256,

    "resolved": {},
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*6)

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.75, 0.05])*0.075#np.array([-0.05, -0.75])*0.075
    # params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([-0.2167105 , -0.17420508])#*0.
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([
        np.array(exp.hdr["TARSIAFX"],dtype=float) - (256-181), 
        np.array(exp.hdr["TARSIAFY"],dtype=float) - (256-44)
    ])*0.075


    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.zeros(16).at[0].set(120.)#np.array([120.])

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.,10.]) #np.asarray([-13.,-7.])#
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 2.5#-90.
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -0.6##-90.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.06,-0.06])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.

    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((resolved_wid,resolved_wid))#+3#.at[:3, :3].set(3.)-2
    


    # params["quadrature"][exp.fit.get_key(exp, "quadrature")] = np.array(0.)


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
# plot_comparison(model_single, params, exposures_single)

# %%

g = 5e-2



things = {
    "primary_opd": sgd(g*0.02, 0),

    "spectrum": sgd(g*1, 0),
    "primary_tilt": sgd(g*1., 0),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1., 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0),
    "cold_mask_rot": sgd(g*5., 50),
    "primary_rot": sgd(g*3., 0),

    "primary_low": sgd(g*1, 0),

    "cold_mask_scale": sgd(g*1, 0),
    "cold_mask_shear": sgd(g*1, 0),

    "primary_shear": sgd(g*1, 0),
    # "primary_shear": sgd(g*5, 100),

    "occulter_radius": sgd(g*1., 0),
    "occulter_coeffs": sgd(g*1, 0),

    "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0),
    "primary_spider": sgd(g*1., 0),

    "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0),


    "resolved": adam(3e-2, 500)
}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*3., 15.),
    "cold_mask_tilt": sgd(g*0.3, 0),
    "cold_mask_opd": sgd(g*0.2, 40),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*10., 70),
    "cold_mask_rot": sgd(g*3., 85),
    "primary_rot": sgd(g*20., 85),

    "primary_low": sgd(g*0.05, 150),

    "cold_mask_scale": sgd(g*5, 100),
    "cold_mask_shear": sgd(g*10, 100),

    "primary_shear": sgd(g*5, 100),
    # "primary_shear": sgd(g*5, 100),

    "occulter_radius": sgd(g*1., 120),
    "occulter_coeffs": sgd(g*1, 120),

    # "outer_radius": sgd(g*1., 120),
    "secondary_radius": sgd(g*1., 120),
    "spider_width": sgd(g*1., 120),
    "primary_spider": sgd(g*1., 120),

    "fnumber": sgd(g*1., 200),
    "anisotropy": sgd(g*2., 200),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 600,
    precond_method="gn_exact", precond_kwargs=dict(chunk=16))

# %%
# plt.plot(losses[:])

# %%
plot_params(params_history, list(things_start.keys()), xw = 3, save="imlup-intermediate-params")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, save="imlup-intermediate-comparison")


# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new_resolved(
    opt_params, model_single, exposures_single, things, 3000,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32), precond_resolved=True)

# %%
plt.plot(losses[:])

# %%
losses[-1]

# %%
plot_params(params_history, groups, xw = 4, save="imlup-params")
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



# %%



