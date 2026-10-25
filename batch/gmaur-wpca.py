# %%
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax
import numpy
import os

jax.config.update("jax_enable_x64", True)

import zodiax as zdx
import optax
import equinox as eqx

import dLux as dl
import dLux.utils as dlu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

out = "gmaur-wpca"
os.makedirs(out, exist_ok=True)

# %%
wf_wid = 512
wid = 64
oversample = 4
pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

filt = "F110W"
nwavels = 200
npoly = 8

# Wavefront PCA basis of the calibrator-joint fits (notebooks/calibrator-pca.ipynb): the mean high-order wavefront plus
# n_wpca unit-norm modes (in nm per unit coefficient), as Fourier coefficients on the fits' grid, with the primary_low span removed. The high-order
# wavefront is restricted to this basis (primary_opd stays zero and is not fitted). The basis was fitted in F160W on a
# 48 x 48 grid (k_max = 24), so in F110W it reaches only ~1.5" (0.9 um) to ~1.8" (1.1 um), short of the crop edges (~2.6")
wpca_file = "../data/wavefront_pca.npz"
wpca = np.load(wpca_file)
n_wpca = wpca["modes"].shape[0]
n_modes = int(round(wpca["mean"].size**0.5))
n_zernikes = 50

resolved_wid = 56   # 56 x 0.0864" = 4.84", the 64 px (4.84") crop
regulariser = np.array([0.02, 0.4])

# The calibrator-joint model: Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter,
# Hermite interpolation between 20 knots across the 200-wavelength spectrum, no pupil amplitude map; plus the wavefront PCA.
# The disc (one wavelength, the filter centre, propagated directly) is an InterpolatedResolvedSource: exact PSFs within
# 0.6" and 36 anchor PSFs beyond (rings 0.6-2.2", 6 angles), 184 renders instead of 56^2 = 3136
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20,
                                  wavefront_pca=wpca_file, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_{filt}_{nwavels}.npy")[:, :npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'


class WavefrontPCAResolvedFit(PointResolvedFit):
    """Star plus resolved disc, with calibrator-joint's per-exposure keys (cold mask shift, tilt and OPD, detector
    defocus); the wavefront PCA coefficients are per visit, like primary_opd in calibrator-joint. One stellar spectrum
    for all exposures (global), as is the disc (keyed by target)."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                "primary_wpca": "visit", "cold_mask_opd": "exposure",
                                                "defocus3": "exposure", "spectrum": "global"}


# GM Aur F110W, 48 s, programme 10852 (DD2): two rolls 58 deg apart, two exposures each. The disc is shared by both
# rolls, so roll diversity separates it from the pupil-fixed speckles, alongside the wavefront PCA basis
#   n9p511 2006-09-17, ORIENTAT  43.6 deg
#   n9p510 2006-11-22, ORIENTAT -14.4 deg
# (The 1998 F110W visit n4q813 (DD1, 640 s, -145.5 deg) is not used.)
exposures_single = [
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits',
                       WavefrontPCAResolvedFit(spectrum_basis, filt, wid=resolved_wid, regulariser=regulariser,
                                               resolved_source="interp"),
                       crop=wid, extra_bad=None, hole_margin=0.1)
    for root in [
        "n9p511wzq", 
        "n9p511x0q", 
        "n9p510kxq", 
        "n9p510kyq"
        ]
]

# %%
# Instrument state: the calibrator-joint medians (notebooks/calibrator-pca.ipynb). Global parameters stay fixed at
# them; the per-exposure ones start at them and are refitted. defocus3 is held at the medians (0 if they do not have it),
# as calibrator-joint holds it at 0
optical_params = np.load("../data/calibrator_optical_params.npy", allow_pickle=True)[()]
per_exposure_init = [k for k in ["cold_mask_shift", "cold_mask_opd", "defocus1", "defocus2", "defocus3"] if k in optical_params]

params = {k: {} for k in ["spectrum", "primary_wpca", "primary_low", "primary_tilt", "cold_mask_tilt", "bias",
                          "defocus3", "resolved"] + per_exposure_init}
for exp in exposures_single:
    key = lambda k: exp.fit.get_key(exp, k)
    params["spectrum"][key("spectrum")] = np.zeros(npoly).at[0].set(1.)          # rescaled below
    params["primary_tilt"][key("primary_tilt")] = np.array([-0.05, -0.75])*0.075
    params["cold_mask_tilt"][key("cold_mask_tilt")] = hole_tilt(exp)
    params["primary_wpca"][key("primary_wpca")] = np.zeros(n_wpca)              # 0: the calibrators' mean wavefront
    params["primary_low"][key("primary_low")] = np.zeros(n_zernikes)
    params["bias"][key("bias")] = 0.
    params["defocus3"][key("defocus3")] = 0.
    for k in per_exposure_init:
        params[k][key(k)] = np.asarray(optical_params[k]["global"])
    params["resolved"][key("resolved")] = np.zeros((resolved_wid, resolved_wid))

params |= {k: v for k, v in optical_params.items() if k not in per_exposure_init}

model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
params = ModelParams(params)

# %%
# Initial star flux: match the total flux of the data minus the (unchanged) initial disc over the unmasked pixels of all
# exposures (one global spectrum, so one scaling)
render = eqx.filter_jit(lambda m, e: e.fit(m, e))

def images(p):
    m = ModelParams(p).inject(model_single)
    return [render(m, e) for e in exposures_single]

p0 = params.params
star = images(p0 | {"resolved": jax.tree.map(lambda r: np.full_like(r, -30.), p0["resolved"])})   # disc off
disc = images(p0 | {"spectrum": jax.tree.map(np.zeros_like, p0["spectrum"])})                      # star off (eps flux)

c_star = sum(np.nansum(np.where(e.bad, 0., e.data - d)) for e, d in zip(exposures_single, disc)) / \
         sum(np.nansum(np.where(e.bad, 0., s)) for e, s in zip(exposures_single, star))
print(f"initial star flux scaling: x{float(c_star):.3g}")

params = ModelParams(p0 | {"spectrum": jax.tree.map(lambda s: s*c_star, p0["spectrum"])})

# %%
plot_comparison_detailed(model_single, params, exposures_single, percentile=99, wf_size=wf_wid, save=f"{out}/initial-comparison")

# %%
g = 5e-2

# Stage 1: pointing, focus, cold mask, low-order and wavefront PCA, with the disc held at its initial value
things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*5, 15.),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*0.1, 40),
    "defocus1": sgd(g*1, 60),
    "defocus2": sgd(g*10, 60),
    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*0.3, 70),
    "primary_low": sgd(g*2., 150),
    "primary_wpca": sgd(g*5, 100),
}

# Stage 2: the same, undelayed, plus the disc (log10 brightness), released once the star model has settled
things = {k: sgd(g*1, 0) for k in things_start} | {
    "bias": sgd(g*3, 0),
    "primary_wpca": sgd(g*5, 0),
    "resolved": adam(3e-2, 0),
}

# %%
opt_params = set_array({k: v for k, v in params.params.items() if k in things_start})
losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 1000,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=True))

plt.figure(figsize=(10, 10))
plt.plot(losses)
plt.savefig(f"{out}/intermediate-losses.png")
plot_params(params_history, list(things_start), xw=5, save=f"{out}/intermediate-params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False,
                         wf_size=wf_wid, save=f"{out}/intermediate-comparison")
np.save(f"{out}/intermediate-params.npy", params_history[-1])

# %%
# Keep the stage-1 values of parameters that stage 2 does not fit (stage 2 injects into model_single)
model_single = ModelParams(params_history[-1]).inject(model_single)
orig_params = params.params | params_history[-1]
opt_params = set_array({k: orig_params[k] for k in orig_params if k in things})

losses, params_history = optimise_new_resolved(
    opt_params, model_single, exposures_single, things, 500,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=True))

plt.figure(figsize=(10, 10))
plt.plot(losses)
plt.savefig(f"{out}/losses.png")
print("final loss", losses[-1])

plot_params(params_history, list(things), xw=5, save=f"{out}/params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False,
                         wf_size=wf_wid, percentile=99, save=f"{out}/comparison")

# %%
# Recovered disc (log10 surface brightness, keyed by target)
key = exposures_single[0].fit.get_key(exposures_single[0], "resolved")
disc = params_history[-1]["resolved"][key]
np.save(f"{out}/gmaur.npy", disc)

for name, im in (("gmaur", 10**disc), ("gmaur-log", disc)):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.colorbar()
    plt.savefig(f"{out}/{name}.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])


# %%
# optimistix L-BFGS polish (fitting.optimise_optimistix) of the stage-2 fit, over the same fitted parameters including the
# disc, which keeps C = 1 as in stage 2. The preconditioner is stage 2's (gn_probe), recomputed at the stage-2 point. Stops
# at steps below atol (~sigma units) or after n_optx steps; a blow-up or a worse end point falls back to the stage-2 values.
# Outputs carry an "optx" suffix, separate from those above.
n_optx = 2048
x2 = params_history[-1]
model_optx = ModelParams(x2).inject(model_single)
x3, optx_info = optimise_optimistix(x2, model_optx, exposures_single, max_steps=n_optx, return_info=True, ones=("resolved",),
                                    precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=True))
np.save(f"{out}/params_amp_optx.npy", x3)
plot_comparison_detailed(model_optx, ModelParams(x3), exposures_single, quadrature=False, wf_size=wf_wid, percentile=99,
                         save=f"{out}/comparison-optx")

line = (f"optimistix: {optx_info['result']} after {optx_info['num_steps']} steps; loss {optx_info['loss_start']:.1f} -> "
        f"{optx_info['loss_end']:.1f} (stage 2 final {losses[-1]:.1f})" + (f"; frozen {optx_info['frozen']}" if optx_info["frozen"] else ""))
print(line); open(f"{out}/optx.txt", "w").write(line + "\n")

disc_optx = x3["resolved"][key]
np.save(f"{out}/gmaur-optx.npy", disc_optx)
for name, im in (("gmaur-optx", 10**disc_optx), ("gmaur-optx-log", disc_optx)):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.colorbar()
    plt.savefig(f"{out}/{name}.png")
