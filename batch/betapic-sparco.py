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

out = "betapic-sparco"
os.makedirs(out, exist_ok=True)

# %%
# SPARCO-style (Kluska et al. 2014) joint fit of beta Pic in F165M / F171M / F180M / F207M: one disc morphology shared
# by every filter and roll, with a per-filter contrast c = log10(f_star / f_disc). The disc has the star's spectrum within
# each filter (scattered light), so its colour relative to the star is carried by c alone. The wavefront is fitted by
# angular diversity (two rolls 90 deg apart), as in imlup-adi / hr8799-planets, not with the wavefront PCA
wf_wid = 512
wid = 64
oversample = 4
pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

nwavels = 100
npoly = 2
n_nodes = 6  # Hermite knots across the 100-wavelength spectrum, for the star and the disc

# Fourier mode k of primary_opd puts a speckle at k lambda / 3 m; k_max = 25 reaches ~3.6" at 2.07 um, beyond the crop
n_modes = 50
n_zernikes = 50

resolved_wid = 56                    # 56 x 0.0864" = 4.84", the 64 px crop
regulariser = np.array([0.2, 4.0])   # (L1, TSV) weights, from betapic-f207m-wpca
support_radius = 0.3                 # arcsec; the disc map has zero support inside it (under the star)

# Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter, no pupil amplitude map.
# The disc is an InterpolatedResolvedSource (exact PSFs within 0.6", 36 anchor PSFs beyond) propagated over the same
# 6 Hermite knots as the star
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=n_nodes, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)


def load_basis(filt):
    vects = np.load(f"../data/iterative_spectrum_basis_{filt}_{nwavels}.npy")[:, :npoly]
    assert vects.shape == (nwavels, npoly)
    return vects/np.sqrt(np.mean(vects**2, axis=0))


# The un-repaired epochal-flat reductions: the repaired (calf) files of these visits have an empty ERR extension, which
# exposure_from_file needs. Bad pixels are masked through DQ instead of repaired
ddir = '../data/NICMOS-LAPL-DD1/archive.stsci.edu/missions/hlsp/laplace/dd1/LAPL/NICMOS-LAPL-DD1/LAPL_DATA/epochal_flats/not_repaired/'


class SparcoFit(PointResolvedFit):
    """Star plus a resolved disc whose morphology is shared by all filters and rolls ("resolved", global: a log10 map,
    normalised to unit sum over its support) and whose flux in each filter is f_star 10^-c ("contrast", keyed by target
    and filter). The disc has the star's spectrum ("spectrum", keyed by target and filter). The high-order wavefront
    (primary_opd, in nm, so achromatic) and the defocus after the occulter and at the detector are per visit, i.e. shared
    by all filters within one roll; the low-order wavefront, pointing, cold mask shift, tilt and OPD, the defocus before
    the occulter and the bias are per exposure; the instrument geometry (rotations, shears, scales, distortions) is global."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {
        "primary_opd": "visit", "defocus2": "visit", "defocus3": "visit",
        "primary_low": "exposure", "primary_tilt": "exposure",
        "cold_mask_shift": "exposure", "cold_mask_tilt": "exposure", "cold_mask_opd": "exposure",
        "defocus1": "exposure",
        "resolved": "global", "contrast": "target"}
    support: Array

    def __init__(self, spectrum_basis, filter, wid, regulariser=np.zeros(2), support_radius=0.3, resolved_kwargs=None):
        super().__init__(spectrum_basis, filter, wid, regulariser=regulariser, resolved_source="interp",
                         resolved_kwargs=resolved_kwargs)
        # the disc takes the star's (100-wavelength) spectrum in place of the single filter-centre wavelength
        point_spectrum = self.source.point.spectrum
        self.source = self.source.set("resolved.spectrum", point_spectrum)
        R = dlu.pixel_coords(wid, pixel_scale=0.0432*2, polar=True)[0]   # arcsec, as the resolved source's pitch
        self.support = np.asarray(R >= support_radius, dtype=float)

    def star_spectrum(self, model, exposure):
        return self.source.point.spectrum.set("basis_weights", model.get(self.map_param(exposure, "spectrum")))

    def get_distribution(self, model, exposure):
        log_map = model.get(self.map_param(exposure, "resolved"))
        # unit sum over the support (a constant offset of the map is a no-op); max subtracted for stability
        shape = self.support*10**(log_map - jax.lax.stop_gradient(np.max(log_map)))
        shape = shape/np.sum(shape)
        f_star = self.star_spectrum(model, exposure).flux
        return f_star*10**(-model.get(self.map_param(exposure, "contrast")))*shape

    def update_source(self, model, exposure):
        spectrum_coeffs = model.get(self.map_param(exposure, "spectrum"))

        source = self.source.set("point.spectrum.basis_weights", spectrum_coeffs)
        source = source.set("point.flux", source.point.spectrum.flux)
        source = source.set("resolved.spectrum.basis_weights", spectrum_coeffs)

        source = source.set("resolved.distribution", self.get_distribution(model, exposure))
        source = source.set("resolved.roll", -np.deg2rad(exposure.orient))
        return source


# Beta Pic (HD39060), programme 7835 (DD1), NIC2-CORON, 175.9 s each: two rolls 90 deg apart
#   n4xj03 1998-09-16, ORIENTAT  29.5 deg
#   n4xj21 1998-11-15, ORIENTAT 119.5 deg
# One exposure per filter per roll to start with; the rest are commented out. (Visits n4xj02 / n4xj20 are the 1.2 s
# unocculted NIC2-FIX acquisitions and are not used.)
roots = [
    # ("n4xj03usq", "F165M"),
    # # ("n4xj03uvq", "F165M"),
    # # ("n4xj03uwq", "F165M"),
    # # ("n4xj03uxq", "F165M"),
    # ("n4xj21dhq", "F165M"),
    # # ("n4xj21dkq", "F165M"),
    # # ("n4xj21dlq", "F165M"),
    # # ("n4xj21dmq", "F165M"),

    # ("n4xj03v1q", "F171M"),
    # # ("n4xj03v4q", "F171M"),
    # # ("n4xj03v5q", "F171M"),
    # # ("n4xj03v6q", "F171M"),
    # ("n4xj21dpq", "F171M"),
    # # ("n4xj21dsq", "F171M"),
    # # ("n4xj21dtq", "F171M"),
    # # ("n4xj21duq", "F171M"),

    ("n4xj03v8q", "F180M"),
    # ("n4xj03vcq", "F180M"),
    # ("n4xj03vdq", "F180M"),
    ("n4xj21e0q", "F180M"),
    # ("n4xj21e4q", "F180M"),
    # ("n4xj21e5q", "F180M"),

    ("n4xj03viq", "F207M"),
    # ("n4xj03vmq", "F207M"),
    # ("n4xj03vnq", "F207M"),
    # ("n4xj03voq", "F207M"),
    ("n4xj21e6q", "F207M"),
    # ("n4xj21eaq", "F207M"),
    # ("n4xj21ebq", "F207M"),
    # ("n4xj21ecq", "F207M"),
]
filters = sorted({filt for _, filt in roots})
bases = {filt: load_basis(filt) for filt in filters}

exposures_single = [
    exposure_from_file(ddir + f'{root}_cle_cal.fits',
                       SparcoFit(bases[filt], filt, wid=resolved_wid, regulariser=regulariser, support_radius=support_radius),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for root, filt in roots
]
for exp, (root, filt) in zip(exposures_single, roots):
    assert exp.filter == filt, (root, exp.filter, filt)

# %%
params = {
    "spectrum": {},
    "contrast": {},
    "primary_opd": {},
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
    "fnumber": 45.7,
    "anisotropy": 1.+1e-5,

    "outer_radius": 1.2*0.9768,
    "secondary_radius": 0.365*1.2,
    "spider_width": 0.072*1.2,
    "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2,
    "primary_outer": 1.2,
    "primary_pad": 0.065*1.2,
    "secondary_shift": np.zeros(2),        # cm, spider-aligned frame
    "primary_distortion": np.zeros((2, 7)),  # fraction of the 1.2 m radius at the edge
    "distortion": np.zeros((2, 7)),        # detector-plane polynomial distortion

    "resolved": {},
}

for exp in exposures_single:
    key = lambda k: exp.fit.get_key(exp, k)
    basis = bases[exp.filter]
    # first mode with the sign that makes the spectrum positive; rescaled per filter below
    params["spectrum"][key("spectrum")] = np.zeros(npoly).at[0].set(np.sign(np.mean(basis[:, 0])))
    params["contrast"][key("contrast")] = 30.                                  # disc off; set below

    params["primary_tilt"][key("primary_tilt")] = np.array([-0.05, -0.75])*0.075
    params["cold_mask_tilt"][key("cold_mask_tilt")] = hole_tilt(exp)

    params["primary_opd"][key("primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_low"][key("primary_low")] = np.zeros(n_zernikes)
    params["cold_mask_opd"][key("cold_mask_opd")] = np.zeros(16).at[0].set(120.)

    params["cold_mask_shift"][key("cold_mask_shift")] = np.array([13., 10.])
    params["cold_mask_rot"][key("cold_mask_rot")] = 3.2
    params["primary_rot"][key("primary_rot")] = -0.6
    params["cold_mask_scale"][key("cold_mask_scale")] = np.asarray([1., 1.])
    params["cold_mask_shear"][key("cold_mask_shear")] = np.asarray([0.06, -0.06])
    params["primary_shear"][key("primary_shear")] = np.asarray([0., 0.])

    params["bias"][key("bias")] = 0.
    for k in ("defocus1", "defocus2", "defocus3"):
        params[k][key(k)] = 0.
    params["cold_mask_distortion"][key("cold_mask_distortion")] = np.zeros((2, 7))

    params["resolved"][key("resolved")] = np.zeros((resolved_wid, resolved_wid))   # uniform over the support

model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
params = ModelParams(params)

# %%
# Initial star flux per filter: match the total flux of the data over the unmasked pixels of that filter's exposures,
# with the disc off. Then set the contrast so that the disc starts at 1 per supported map pixel, as the single-filter
# fits do (10^0 everywhere), i.e. much fainter than the star
render = eqx.filter_jit(lambda m, e: e.fit(m, e))

def images(p):
    m = ModelParams(p).inject(model_single)
    return [render(m, e) for e in exposures_single]

p0 = set_array(params.params)   # float64 arrays (no python-float/weak-type leaves): render variants share one compile
star = images(p0)               # contrast 30: disc off

spectrum, contrast = dict(p0["spectrum"]), dict(p0["contrast"])
n_support = float(exposures_single[0].fit.support.sum())
for filt in filters:
    idx = [i for i, e in enumerate(exposures_single) if e.filter == filt]
    c_star = sum(np.nansum(np.where(exposures_single[i].bad, 0., exposures_single[i].data)) for i in idx) / \
             sum(np.nansum(np.where(exposures_single[i].bad, 0., star[i])) for i in idx)
    exp = exposures_single[idx[0]]
    k = exp.fit.get_key(exp, "spectrum")
    spectrum[k] = spectrum[k]*c_star
    f_star = exp.fit.source.point.spectrum.set("basis_weights", spectrum[k]).flux
    contrast[exp.fit.get_key(exp, "contrast")] = np.log10(f_star/n_support)
    print(f"{filt}: initial star flux scaling x{float(c_star):.3g}, contrast {float(contrast[k]):.3g}")

params = ModelParams(p0 | {"spectrum": spectrum, "contrast": contrast})
model_single = params.inject(model_single)

# %%
plot_comparison_detailed(model_single, params, exposures_single, percentile=99, wf_size=wf_wid, save=f"{out}/initial-comparison")

# %%
g = 5e-2

# Stage 1: star and instrument, with the disc (map and contrasts) held at its initial value
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

# Stage 2: the high-order wavefront (from angular diversity) and every instrument parameter, then the disc (shared map
# and per-filter contrasts), released once the wavefront has converged, as in imlup-adi
things = {
    "primary_opd": adam(2e-2, 50),

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

    "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0),
    "primary_spider": sgd(g*1., 0),
    "primary_secondary": sgd(g*1., 0),
    "secondary_shift": sgd(g*1, 0),
    "primary_distortion": sgd(g*3, 0),

    "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0),
    "distortion": sgd(g*1, 0),

    "resolved": adam(3e-2, 1200),
    "contrast": sgd(g*1, 1200),
}

groups = list(things.keys())
precond_kwargs = dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32)

# %%
opt_params = set_array({k: v for k, v in params.params.items() if k in things_start})
losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 500,
    mapped=True, precond_method="gn_probe", precond_kwargs=precond_kwargs)

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
    opt_params, model_single, exposures_single, things, 2500,
    mapped=True, precond_method="gn_probe", precond_kwargs=precond_kwargs)

plt.figure(figsize=(10, 10))
plt.plot(losses)
plt.savefig(f"{out}/losses.png")
print("final loss", losses[-1])

plot_params(params_history, groups, xw=5, save=f"{out}/params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False,
                         wf_size=wf_wid, percentile=99, save=f"{out}/comparison")

final = orig_params | params_history[-1]
np.save(f"{out}/params_amp.npy", final)

# %%
# Recovered shared disc morphology: the unit-sum map over its support (the per-filter flux is f_star 10^-c)
exp = exposures_single[0]
log_map = final["resolved"][exp.fit.get_key(exp, "resolved")]
shape = exp.fit.support*10**(log_map - np.max(log_map))
shape = shape/np.sum(shape)
np.save(f"{out}/betapic.npy", shape)

for name, im in (("betapic", shape), ("betapic-log", np.log10(np.where(shape > 0, shape, np.nan)))):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.colorbar()
    plt.savefig(f"{out}/{name}.png")

# %%
# Disc contrast (log10 f_star / f_disc) against the filters' mean wavelength: the disc colour relative to the star
wl_mean = {}
for filt in filters:
    wv, w = calc_throughput(filt, nwavels)
    wl_mean[filt] = float(np.sum(wv*w)/np.sum(w))*1e6

fig, ax = plt.subplots(figsize=(10, 8), layout='compressed')
for filt in filters:
    exp = next(e for e in exposures_single if e.filter == filt)
    c = float(final["contrast"][exp.fit.get_key(exp, "contrast")])
    ax.plot(wl_mean[filt], c, 'o', ms=12)
    ax.annotate(filt, (wl_mean[filt], c), xytext=(8, 8), textcoords="offset points", fontsize=16)
    print(f"{filt}: contrast log10(f_star/f_disc) = {c:.4f}")
ax.set(xlabel="Filter mean wavelength (um)", ylabel="log10(f_star / f_disc)", title="Disc contrast")
fig.savefig(f"{out}/contrast.png")

# %%
# Recovered stellar (and disc) spectrum in each filter, before the filter and as detected
fig, axs = plt.subplots(1, 2, figsize=(20, 8), layout='compressed')
for filt in filters:
    exp = next(e for e in exposures_single if e.filter == filt)
    spec = exp.fit.source.point.spectrum.set("basis_weights", final["spectrum"][exp.fit.get_key(exp, "spectrum")])
    wavels = 1e6*spec.wavelengths
    axs[0].plot(wavels, spec.spec_weights()/np.mean(spec.spec_weights()), label=filt)
    axs[1].plot(wavels, spec.weights, label=filt)
axs[0].set(xlabel="Wavelength (um)", ylabel="Relative flux", title="Stellar spectrum")
axs[1].set(xlabel="Wavelength (um)", ylabel="Normalised weight", title="Detected spectrum")
axs[0].legend()
fig.savefig(f"{out}/spectrum.png")
