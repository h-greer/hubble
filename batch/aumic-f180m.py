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

out = "aumic-f180m"
os.makedirs(out, exist_ok=True)

# %%
wf_wid = 512
wid = 64
oversample = 4
pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

nwavels = 200
npoly=1

# The 64 px crop (centre detector pixel (75, 212)) holds the July 2004 hole (TARSIAFX/Y ~ (74.96, 211.66)) within ~0.5 px
# of its centre, so its farthest edges are ~32.5 px ~ 2.46" from the star. Fourier mode k of primary_opd puts a speckle at
# k lambda / 3 m (the 3 m wavefront array), so reaching them at the blue end of F180M (1.70 um) needs k ~ 21;
# k_max = n_modes // 2 = 24 adds ~2 lambda/D for the PSF wings of speckles just outside the crop
n_modes = 49
n_zernikes = 50

resolved_wid = 56   # 56 x 0.0864" = 4.84", the 64 px (4.84") crop
regulariser = np.array([0.02, 0.4])

# Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter with an edge glint ring;
# the PSF is Hermite-interpolated between 20 knot wavelengths across the 200-wavelength spectrum. No pupil amplitude map.
# The resolved disc (one wavelength, propagated directly, not through the Hermite knots) is an InterpolatedResolvedSource:
# exact PSFs within 0.6" (148 pixels) and 36 anchor PSFs beyond (rings 0.6-2.2", 6 angles), 184 renders instead of 56^2 = 3136,
# 0.46% rms of the exact image outside the hole mask on this geometry. Keep the outer ring ~0.35" inside the window edge
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=6, pad=pad)
# optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20, amplitude=True, pad=pad)

detector = NICMOSDetector(oversample, wid, pad=pad)

filt = "F180M"
# Generate with notebooks/eigenbasis.ipynb (filtname = "F180M") if missing
vects = np.load(f"../data/iterative_spectrum_basis_{filt}_{nwavels}.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'


class ResolvedFit(PointResolvedFit):
    """Star plus resolved disc (PointResolvedFit), with hr8799-planets' PlanetFit keys: the cold mask lateral
    position (shift), its tilt (initialised from the era's approximate hole offset, models.hole_tilt), cold-mask OPD and
    defocus can differ between exposures; the high-order wavefront (primary_opd) is global (shared by both
    rolls, so roll diversity separates pupil-fixed speckles from the sky-fixed disc); low-order Zernikes stay per exposure."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                "primary_opd": "global", "cold_mask_opd": "exposure",
                                                "defocus3": "exposure"}


# AU Mic (GJ 803) F180M, 192 s exposures: one from each roll of the 2004-07-27 pair (visits n93m27 / n93m28, ORIENTAT
# 114.6 / 139.6 deg, same day, so the global wavefront need only hold across a ~25 deg roll); the spectrum and disc are
# keyed by target, so both exposures share them. Remaining F180M: n93m27hcq-heq, n93m28hzq-i1q, and the 2004-09-28
# rolls n93m57g6q-g9q (-163.4 deg) / n93m58gtq-gwq (-139.4 deg)
exposures_single = [
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits', ResolvedFit(spectrum_basis, filt, wid=resolved_wid, regulariser=regulariser,
                                                                                  resolved_source="interp"),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for root in ["n93m27hbq", "n93m28hyq"]
]

# %%
params = {
    "spectrum": {},
    "primary_opd": {},
    # "primary_amp": {},
    "primary_low": {},
    "primary_tilt": {},
    "cold_mask_opd": {},
    "cold_mask_tilt": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_shear": {},
    "cold_mask_scale": {},
    "cold_mask_distortion": {},
    "defocus1": {},
    "defocus2": {},
    "defocus3": {},
    "primary_rot": {},
    "primary_shear": {},

    "bias": {},
    "occulter_radius": 0.8,
    "occulter_ellipticity": np.zeros(2),   # (e1, e2); major axis at atan2(e2, e1)/2 from the occulter-plane x axis
    # "occulter_glint": np.zeros((5, 2)),    # (re, im) Fourier coefficients (g0, g1 cos/sin, g2 cos/sin) of the edge ring
    # "occulter_glint_width": 3.,            # um in the occulter plane
    "fnumber": 45.7,
    "anisotropy": 1.+1e-5,

    "outer_radius": 1.2*0.9768,
    "secondary_radius": 0.365*1.2,
    "spider_width": 0.072*1.2,
    "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2,
    "primary_outer": 1.2,
    "primary_pad": 0.065*1.2,
    "secondary_shift": np.zeros(2),        # cm, spider-aligned frame; ~1 cm expected from the 7.4' field offset
    "primary_distortion": np.zeros((2, 7)),  # fraction of the 1.2 m radius at the edge
    "distortion": np.zeros((2, 7)),        # detector-plane polynomial distortion

    "resolved": {},
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*100) # matches the data flux at initialisation

    # star pointing within the hole: target-specific, IM Lup's value as a starting guess (calibrator-joint uses the axes
    # swapped, [-0.05, -0.75]); check initial-comparison and adjust
    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.75, -0.05])*0.075
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = hole_tilt(exp)

    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    # params["primary_amp"][exp.fit.get_key(exp, "primary_amp")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.zeros(16).at[0].set(120.)

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.,10.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 3.2
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -0.6
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.06,-0.06])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    for k in ("defocus1", "defocus2", "defocus3"):
        params[k][exp.fit.get_key(exp, k)] = 0.
    params["cold_mask_distortion"][exp.fit.get_key(exp, "cold_mask_distortion")] = np.zeros((2, 7))

    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((resolved_wid,resolved_wid))


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)

# %%
# Initial star flux. The image is linear in the star's spectrum coefficients, so scale them to match the total flux of
# the data minus the (unchanged) initial disc over the unmasked pixels of every exposure (the spectrum is keyed by
# target, so one amplitude is shared by all exposures). Not a chi^2 fit: with the initial (wrong) speckle pattern the
# 1/err^2 weights favour the faint outer pixels and give ~x1.8 where the flux needs ~x5
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

things = {
    "primary_opd": adam(2e-2, 50),
    # Delayed until well after the phase (primary_opd starts at 50), so the phase is fitted first
    # and the phase/amplitude degeneracy is avoided
    # "primary_amp": adam(2e-2, 150),

    "spectrum": sgd(g*1, 0),
    "primary_tilt": sgd(g*1., 0),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1., 0),

    "defocus1": sgd(g*1, 0),
    "defocus2": sgd(g*1, 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0),
    "cold_mask_rot": sgd(g*1., 0.),
    "primary_rot": sgd(g*3., 0),

    "primary_low": sgd(g*1, 0),

    "cold_mask_scale": sgd(g*1, 0),
    "cold_mask_distortion": sgd(g*3, 0),

    "occulter_radius": sgd(g*1., 0),
    "occulter_ellipticity": sgd(g*3, 300),
    # "occulter_glint": sgd(g*15, 400),
    # "occulter_glint_width": sgd(g*5, 600),   # smaller: width and amplitude trade off (ring area ~ g w)

    "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0),
    "primary_spider": sgd(g*1., 0),
    "primary_secondary": sgd(g*1., 0),
    "secondary_shift": sgd(g*1, 0),
    "primary_distortion": sgd(g*3, 0),

    "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0),
    "distortion": sgd(g*1, 0),

    # Disc (log10 brightness, initially 0, i.e. 1 everywhere) held until the wavefront and instrument have converged,
    # as the planets are in hr8799-planets
    "resolved": adam(3e-2, 1200),
}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*5., 15.),
    "cold_mask_tilt": sgd(g*1., 0),
    "cold_mask_opd": sgd(g*0.1, 40),

    "defocus1": sgd(g*1, 60),
    "defocus2": sgd(g*10, 60),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*0.3, 70),
    "cold_mask_rot": sgd(g*0.1, 85),
    "primary_rot": sgd(g*0.03, 85),

    "primary_low": sgd(g*2., 150),

    "cold_mask_shear": sgd(g*0.1, 100),
    "cold_mask_distortion": sgd(g*0.1, 150),

    "primary_shear": sgd(g*1, 100),

    "occulter_radius": sgd(g*1., 120),
    # in stage 1 too, so stage 2's preconditioner (computed once) sees a nonzero glint and can move its width
    # "occulter_glint": sgd(g*0.1, 300),

    "outer_radius": sgd(g*0.1, 120),
    "secondary_radius": sgd(g*0.1, 120),
    "spider_width": sgd(g*0.1, 120),
    "primary_spider": sgd(g*0.1, 120),
    "primary_secondary": sgd(g*0.1, 120),
    "primary_pad": sgd(g*0.1, 120),
    "primary_outer": sgd(g*0.1, 120),
    "secondary_shift": sgd(g*0.3, 120),
    "primary_distortion": sgd(g*0.1, 150),

    "fnumber": sgd(g*1., 200),
    "anisotropy": sgd(g*2., 200),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 500,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=True))

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/intermediate-losses.png")

plot_params(params_history, list(things_start.keys()), xw = 5, save=f"{out}/intermediate-params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False, wf_size=wf_wid, save=f"{out}/intermediate-comparison")
np.save(f"{out}/intermediate-params.npy", params_history[-1])

# %%
# Keep the stage-1 values of parameters that stage 2 does not fit (stage 2 injects into model_single)
model_single = ModelParams(params_history[-1]).inject(model_single)
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

losses, params_history = optimise_new_resolved(
    opt_params, model_single, exposures_single, things, 2000,
    precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=True))

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/losses.png")
print("final loss", losses[-1])

plot_params(params_history, groups, xw = 5, save=f"{out}/params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, wf_size=wf_wid, percentile=99, save=f"{out}/comparison")

# %%
# Recovered disc (log10 surface brightness, keyed by target)
key = exposures_single[0].fit.get_key(exposures_single[0], "resolved")
disc = params_history[-1]["resolved"][key]
np.save(f"{out}/aumic.npy", disc)

plt.figure(figsize=(10,10))
plt.imshow(10**disc)
plt.colorbar()
plt.savefig(f"{out}/aumic.png")

plt.figure(figsize=(10,10))
plt.imshow(disc)
plt.colorbar()
plt.savefig(f"{out}/aumic-log.png")

# %%
# # Recovered amplitude where both the primary and the cold mask transmit, as in "Pupil x Cold Mask"
# model = ModelParams(params_history[-1]).inject(model_single)
# exp = exposures_single[0]
# optics = exp.fit.update_optics(model, exp)

# coords = dlu.pixel_coords(wf_wid, optics.diameter)
# primary = optics.primary.transmission(coords, 2.4/wf_wid)
# cold_mask = optics.cold_mask.transmission(coords, 2.4/wf_wid)
# amp = 100*(np.exp(optics.primary_amp.eval_basis()) - 1)

# fig, ax = plt.subplots(figsize=(10, 10), layout='compressed')
# _signed_panel(ax, np.where(primary*cold_mask < .5, np.nan, amp), "Pupil x Cold Mask Amplitude", label="Amplitude (%)", percentile=99)
# fig.savefig(f"{out}/amplitude.png")

# %%
# Recovered stellar spectrum (before the filter) and the detected (filtered) spectrum
exp = exposures_single[0]
spectrum = exp.fit.source.point.spectrum.set("basis_weights", ModelParams(params_history[-1]).get(exp.fit.map_param(exp, "spectrum")))
wavels = 1e6*spectrum.wavelengths

fig, axs = plt.subplots(1, 2, figsize=(20, 8), layout='compressed')
axs[0].plot(wavels, spectrum.spec_weights())
axs[0].set(xlabel="Wavelength (um)", ylabel="Relative flux", title="Stellar spectrum")
axs[1].plot(wavels, spectrum.weights)
axs[1].set(xlabel="Wavelength (um)", ylabel="Normalised weight", title="Detected spectrum")
fig.savefig(f"{out}/spectrum.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])
