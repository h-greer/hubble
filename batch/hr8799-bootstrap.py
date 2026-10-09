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
import os

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from plotting import _signed_panel  # underscore names are not covered by `import *`
from spectra import *

import interpax as ipx

out = "hr8799-bootstrap"
os.makedirs(out, exist_ok=True)

# %%
wf_wid = 512
wid = 80
oversample = 4

nwavels = 20
npoly=5

n_modes = 45
n_zernikes = 30

optics = NICMOSCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, amplitude=True)

detector = NICMOSDetector(oversample, wid)

vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))


ddir = '../data/NICMOS-LAPL-DD1/archive.stsci.edu/missions/hlsp/laplace/dd1/LAPL/NICMOS-LAPL-DD1/LAPL_DATA/contemp_flats/repaired/'
flatdir = '../data/NICMOS-LAPL-DD1/archive.stsci.edu/missions/hlsp/laplace/dd1/LAPL/NICMOS-LAPL-DD1/HOLEFLATS/'

# Same visit pair at two rolls (ORIENTAT -147.4 and -117.5 deg); the spectrum is keyed by target, so both share it
exposures_single = [
    exposure_from_file(ddir + 'n4qs09akq_clc_calf.fits', SinglePointFit(spectrum_basis, "F160W"), crop=wid, extra_bad=None, flatcorr=flatdir),
    exposure_from_file(ddir + 'n4qs10asq_clc_calf.fits', SinglePointFit(spectrum_basis, "F160W"), crop=wid, extra_bad=None, flatcorr=flatdir),
]

# %%
params = {
    "spectrum": {},
    "primary_opd": {},
    "primary_amp": {},
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
    "jitter": {},

    "bias": {},
    "occulter_radius": 0.8,
    "occulter_coeffs": np.zeros(2)+1e-3,
    "fnumber": 45.7,
    "anisotropy": 1.+1e-5,

    "outer_radius": 1.2*0.9768,
    "secondary_radius": 0.365*1.2,
    "spider_width": 0.072*1.2,
    "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2,
    "primary_outer": 1.2,
    "primary_pad": 0.065*1.2,
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*100) # matches the data flux at initialisation

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.05, -0.75])*0.075
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([
        np.array(exp.hdr["TARSIAFY"],dtype=float) - (256-44),
        np.array(exp.hdr["TARSIAFX"],dtype=float) - (256-181),
    ])*0.075

    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_amp"][exp.fit.get_key(exp, "primary_amp")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.zeros(16).at[0].set(120.)

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.,10.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 3.2
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -0.6
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.06,-0.06])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7. # mas; gradient vanishes at 0


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)

# %%
plot_comparison_detailed(model_single, params, exposures_single, percentile=99, wf_size=wf_wid, save=f"{out}/initial-comparison")

# %%
g = 5e-2

things = {
    "primary_opd": adam(2e-2, 50),
    # Delayed until well after the phase (primary_opd starts at 50), so the phase is fitted first
    # and the phase/amplitude degeneracy is avoided
    "primary_amp": adam(2e-2, 150),

    "spectrum": sgd(g*1, 0),
    "primary_tilt": sgd(g*1., 0),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1., 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0),
    "cold_mask_rot": sgd(g*1., 0.),
    "primary_rot": sgd(g*3., 0),

    "primary_low": sgd(g*1, 0),
    "jitter": sgd(g*1, 100),

    "cold_mask_scale": sgd(g*1, 0),

    "occulter_radius": sgd(g*1., 0),
    "occulter_coeffs": adam(2e-2, 300),

    "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0),
    "primary_spider": sgd(g*1., 0),
    "primary_secondary": sgd(g*1., 0),

    "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0),
}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*5., 15.),
    "cold_mask_tilt": sgd(g*1., 0),
    "cold_mask_opd": sgd(g*0.1, 40),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*0.3, 70),
    "cold_mask_rot": sgd(g*0.1, 85),
    "primary_rot": sgd(g*0.03, 85),

    "primary_low": sgd(g*2., 150),

    "cold_mask_shear": sgd(g*0.1, 100),

    "primary_shear": sgd(g*1, 100),

    "occulter_radius": sgd(g*1., 120),

    "outer_radius": sgd(g*0.1, 120),
    "secondary_radius": sgd(g*0.1, 120),
    "spider_width": sgd(g*0.1, 120),
    "primary_spider": sgd(g*0.1, 120),
    "primary_secondary": sgd(g*0.1, 120),
    "primary_pad": sgd(g*0.1, 120),
    "primary_outer": sgd(g*0.1, 120),

    "fnumber": sgd(g*1., 200),
    "anisotropy": sgd(g*2., 200),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 500,
    precond_method="gn_exact", precond_kwargs=dict(chunk=16))

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/intermediate-losses.png")

plot_params(params_history, list(things_start.keys()), xw = 5, save=f"{out}/intermediate-params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False, wf_size=wf_wid, save=f"{out}/intermediate-comparison")
np.save(f"{out}/intermediate-params.npy", params_history[-1])

# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things, 2000,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32))

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/losses.png")
print("final loss", losses[-1])

plot_params(params_history, groups, xw = 5, save=f"{out}/params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, wf_size=wf_wid, percentile=99, save=f"{out}/comparison")

# %%
# Recovered amplitude where both the primary and the cold mask transmit, as in "Pupil x Cold Mask"
model = ModelParams(params_history[-1]).inject(model_single)
exp = exposures_single[0]
optics = exp.fit.update_optics(model, exp)

coords = dlu.pixel_coords(wf_wid, optics.diameter)
primary = optics.primary.transmission(coords, 2.4/wf_wid)
cold_mask = optics.cold_mask.transmission(coords, 2.4/wf_wid)
amp = 100*(np.exp(optics.primary_amp.eval_basis()) - 1)

fig, ax = plt.subplots(figsize=(10, 10), layout='compressed')
_signed_panel(ax, np.where(primary*cold_mask < .5, np.nan, amp), "Pupil x Cold Mask Amplitude", label="Amplitude (%)", percentile=99)
fig.savefig(f"{out}/amplitude.png")

# %%
# Recovered stellar spectrum (before the filter) and the detected (filtered) spectrum
exp = exposures_single[0]
spectrum = exp.fit.source.spectrum.set("basis_weights", ModelParams(params_history[-1]).get(exp.fit.map_param(exp, "spectrum")))
wavels = 1e6*spectrum.wavelengths

fig, axs = plt.subplots(1, 2, figsize=(20, 8), layout='compressed')
axs[0].plot(wavels, spectrum.spec_weights())
axs[0].set(xlabel="Wavelength (um)", ylabel="Relative flux", title="Stellar spectrum")
axs[1].plot(wavels, spectrum.weights)
axs[1].set(xlabel="Wavelength (um)", ylabel="Normalised weight", title="Detected spectrum")
fig.savefig(f"{out}/spectrum.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])
print({k: float(np.abs(v)) for k, v in ModelParams(params_history[-1]).get("jitter").items()}, "mas")
