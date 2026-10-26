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

# Persistent compilation cache (directory from JAX_COMPILATION_CACHE_DIR, set in the .sh); cache every compile
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

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
from plotting import _signed_panel  # underscore names are not covered by `import *`
from spectra import *

out = "betapic-f207m-wpca"
os.makedirs(out, exist_ok=True)

# %%
wf_wid = 512
wid = 64
oversample = 4
pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

filt = "F207M"
nwavels = 200
n_nodes = 6     # Hermite-interpolate the PSF between 6 knot wavelengths
npoly = 1       # flux only: there is no iterative spectrum basis for F207M

# Wavefront PCA basis of the calibrator-joint fits (notebooks/calibrator-pca.ipynb): the mean high-order wavefront plus
# n_wpca unit-norm modes (in nm per unit coefficient), as Fourier coefficients on the fits' grid, with the primary_low span removed. The high-order
# wavefront is restricted to this basis (primary_opd stays zero and is not fitted). The basis was fitted in F160W on a
# 48 x 48 grid (k_max = 24), so at 2.07 um it reaches ~3.4", beyond the crop edges (~2.6")
wpca_file = "../data/wavefront_pca.npz"
wpca = np.load(wpca_file)
n_wpca = wpca["modes"].shape[0]
n_modes = int(round(wpca["mean"].size**0.5))
n_zernikes = 50

resolved_wid = 56   # 56 x 0.0864" = 4.84", the 64 px (4.84") crop
regulariser = np.array([0.2, 4.0])   # (L1, TSV) weights, 10x gmaur-wpca's: the outer disc map was unconstrained pixel noise

# The calibrator-joint model: Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter,
# Hermite interpolation between 6 knots across the 200-wavelength spectrum, no pupil amplitude map; plus the wavefront PCA.
# The disc (one wavelength, the filter centre, propagated directly) is an InterpolatedResolvedSource: exact PSFs within
# 0.6" and 36 anchor PSFs beyond (rings 0.6-2.2", 6 angles), 184 renders instead of 56^2 = 3136
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=n_nodes,
                                  wavefront_pca=wpca_file, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)

vects = numpy.ones((nwavels, npoly))   # flat: the spectral shape is the filter throughput
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

# The un-repaired epochal-flat reductions: the repaired (calf) files of these visits have an empty ERR extension, which
# exposure_from_file needs. Bad pixels are masked through DQ instead of repaired
ddir = '../data/NICMOS-LAPL-DD1/archive.stsci.edu/missions/hlsp/laplace/dd1/LAPL/NICMOS-LAPL-DD1/LAPL_DATA/epochal_flats/not_repaired/'


class WavefrontPCAResolvedFit(PointResolvedFit):
    """Star plus resolved disc, with calibrator-joint's per-exposure keys (cold mask shift, tilt and OPD, detector
    defocus); the wavefront PCA coefficients are per exposure. One stellar spectrum for all exposures (global), as is
    the disc (keyed by target)."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                "primary_wpca": "exposure", "cold_mask_opd": "exposure",
                                                "defocus3": "exposure", "spectrum": "global"}


# Beta Pic (HD39060) F207M, programme 7835 (DD1), NIC2-CORON, 175.9 s each: two rolls 90 deg apart, four exposures each
# (the same visits as the F180M fit).
# The disc is shared by both rolls, so roll diversity separates it from the pupil-fixed speckles; the wavefront is fitted
# per exposure
#   n4xj03 1998-09-16, ORIENTAT  29.5 deg
#   n4xj21 1998-11-15, ORIENTAT 119.5 deg
exposures_single = [
    exposure_from_file(ddir + f'{root}_cle_cal.fits',
                       WavefrontPCAResolvedFit(spectrum_basis, filt, wid=resolved_wid, regulariser=regulariser,
                                               resolved_source="interp"),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for root in [
        "n4xj03viq",
        "n4xj03vmq",
        "n4xj03vnq",
        "n4xj03voq",
        "n4xj21e6q",
        "n4xj21eaq",
        "n4xj21ebq",
        "n4xj21ecq",
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

p0 = set_array(params.params)   # float64 arrays (no python-float/weak-type leaves): render variants share one compile
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
    mapped=True, precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32))

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
    mapped=True, precond_method="gn_probe", precond_kwargs=dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32))

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
np.save(f"{out}/betapic.npy", disc)

for name, im in (("betapic", 10**disc), ("betapic-log", disc)):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.colorbar()
    plt.savefig(f"{out}/{name}.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])


# %%
# Residuals and the disc with the predicted positions of beta Pic b, c and d marked, as in hr8799-bootstrap. Sky
# offsets (east, north; arcsec) at each visit: b and c from the Lacour et al. 2021 (A&A 654, L2) orbits (Table 2, RV + b
# and c astrometry; Keplerian, reproducing their GRAVITY astrometry to ~1.5 mas); d from the Sutlieff & Bonse et al.
# 2026 posterior modes (a 26.0 au, i 88.9, Omega 210.9, lambda 270 deg at 2010.0; circular, reproducing their
# detections to <40 mas, <1 deg; its separation is poorly constrained, its PA is not). c is inside the hole at both epochs.
# The star sits at the crop centre shifted by primary_tilt + cold_mask_tilt (arcsec) at the 40 um / (45.7 x 2.4 m) plate
# scale (checked against out-of-hole renders to <0.4 px); the planets are offset from it through the exposure WCS. The
# disc grid is in a fixed sky frame: row = centre + north / pitch, column = centre - east / pitch
mark_r = 3
planets = {
    "n4xj03": {"b": (0.263, 0.431), "c": (0.029, 0.053), "d": (0.467, 0.743)},   # 1998-09-16
    "n4xj21": {"b": (0.268, 0.439), "c": (0.043, 0.075), "d": (0.462, 0.734)},   # 1998-11-15
}
from astropy.coordinates import SkyCoord
import astropy.units as u

final = params.params | params_history[-1]
model = ModelParams(final).inject(model_single)
plate_scale = dlu.rad2arcsec(40e-6/(45.7*2.4))   # arcsec per detector pixel
for exp in exposures_single:
    star_xy = (wid - 1)/2 + (final["primary_tilt"][exp.key] + final["cold_mask_tilt"][exp.key])/plate_scale
    star = SkyCoord(exp.hdr["RA_TARG"]*u.deg, exp.hdr["DEC_TARG"]*u.deg)
    sxy = numpy.array(exp.wcs.world_to_pixel(star))
    resid = exp.data - render(model, exp)
    fig, axs = plt.subplots(1, 2, figsize=(20, 9), layout='compressed')
    for ax, im, unit in zip(axs, [resid, resid/exp.err], ["DN/s", "z"]):
        _signed_panel(ax, im, f"{exp.key} residual ({unit})", percentile=99, label=unit, bad='w')
        for name, (east, north) in planets[exp.key[:6]].items():
            x, y = star_xy + numpy.array(exp.wcs.world_to_pixel(star.spherical_offsets_by(east*u.arcsec, north*u.arcsec))) - sxy
            ax.add_patch(matplotlib.patches.Circle((x, y), mark_r, fill=False, ec='g', lw=2))
            ax.text(x + mark_r, y + mark_r, name, color='g', fontsize=24)
    fig.savefig(f"{out}/planets-residual_{exp.key}.png")

pitch = dlu.rad2arcsec(exposures_single[0].fit.source.resolved.pitch)
centre = disc.shape[0]//2
fig, ax = plt.subplots(figsize=(10, 9))
fig.colorbar(ax.imshow(disc, origin='lower'), label='log10 brightness')
for visit, col in zip(planets, ['c', 'm']):
    for name, (east, north) in planets[visit].items():
        r, cc = centre + north/pitch, centre - east/pitch
        ax.add_patch(matplotlib.patches.Circle((cc, r), 2, fill=False, ec=col, lw=2))
        ax.text(cc + 2, r + 2, f"{name} {visit}", color=col, fontsize=12)
fig.savefig(f"{out}/planets-disc.png")
