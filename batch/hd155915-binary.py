# HD 155915 (HIP 84642): binary-star fit (batch export of notebooks/hd155915-binary.ipynb)
#
# NIC2 coronagraphic F160W, 2005-03-15. HD 155915 B lies behind the hole: 220 +/- 14 mas at PA 191.3 +/- 0.8 deg with
# dKs = 2.5 (contrast ~0.10), VLT/NACO 2007-06-05 (Chauvin et al. 2010, Table 6). That is ~1.6 lambda/D at F160W,
# inside the 0.3" hole, so the companion is only seen through the hole-edge light.
#
# Model: as turboklip.ipynb (wavefront PCA, calibrator optical parameters, 64x64 crop). Two models are fitted from the
# same starting point with the same schedule: a single star, and a binary. For the binary, stage 1 fits the star and
# instrument with the companion held at the literature values (a starting point, not a prior: the loss is the
# likelihood alone); stage 2, from the stage-1 result, also frees the companion's offset, contrast and spectral shape.
# Outputs: a 9-panel model comparison per exposure, chi2/BIC of both models, and Fisher-information errors on the
# companion's separation, PA and contrast, marginalised over all fitted parameters (the calibrator optics are fixed,
# so the errors are conditional on them).
#
# Usage: python hd155915-binary.py [rootname ...]   (default: n918b1e5q, one exposure at roll 35.3 deg)
import sys
sys.path.insert(0, '..')
import os

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
from jax.flatten_util import ravel_pytree
import astropy.units as u

out = "hd155915-binary"
os.makedirs(out, exist_ok=True)


# %%
# Binary-star model: the primary at the origin plus a point-source companion through the same optics. Unlike
# hr8799-planets' stop-gradient planets, the companion's PSF passes gradients to the wavefront and instrument, since a
# ~10% companion is not negligible.
#   companion_offset   : sky offset of B from A (east, north; arcsec)
#   companion_contrast : B/A ratio of in-filter detected flux (CombinedSpectrum.flux; the spectrum weights are
#                        normalised internally, so this is exactly the flux ratio)
#   companion_spectrum : B's spectral shape, the basis weights after the first, which is fixed at 1 (B's overall scale
#                        is set by the contrast, so a free first weight would be degenerate with it)
# All three are keyed by target, so every exposure of the system shares them.
def sky_to_optics(offset, orient):
    """Sky offset (east, north; arcsec) -> optics-frame source position (rad) for an exposure at ORIENTAT `orient` (deg):
    x = -east, y = north, rotated by -ORIENTAT (as hr8799-planets, whose fit recovers the HR 8799 b/c positions;
    a render check against this exposure's WCS matched 1-1.5" offsets in three directions to < 0.1 px)."""
    roll = -np.deg2rad(orient)
    rot = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    return dlu.arcsec2rad(rot @ np.array([-offset[0], offset[1]]))


class BinaryFit(SinglePointFit):
    PARAM_KEYS = SinglePointFit.PARAM_KEYS | {"companion_offset": "target", "companion_contrast": "target",
                                              "companion_spectrum": "target"}
    companion: dl.PointSource

    def __init__(self, spectrum_basis, filter, time_series=False):
        super().__init__(spectrum_basis, filter, time_series)
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.companion = dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))

    def update_source(self, model, exposure):
        star = super().update_source(model, exposure)
        shape = model.get(self.map_param(exposure, "companion_spectrum"))
        spectrum = self.companion.spectrum.set("basis_weights", np.concatenate([np.ones(1), shape]))
        flux = model.get(self.map_param(exposure, "companion_contrast")) * star.flux
        position = sky_to_optics(model.get(self.map_param(exposure, "companion_offset")), exposure.orient)
        companion = self.companion.set(["spectrum", "flux", "position"], [spectrum, flux, position])
        return dl.Scene([("star", star), ("companion", companion)])


# turboklip's keys (PCA coefficients per visit, as primary_opd in calibrator-joint)
class WavefrontPCABinaryFit(BinaryFit):
    PARAM_KEYS = BinaryFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                         "primary_wpca": "visit", "cold_mask_opd": "exposure", "defocus3": "exposure"}


class WavefrontPCAFit(SinglePointFit):
    """Single star with turboklip's keys, for the model comparison"""
    PARAM_KEYS = SinglePointFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                              "primary_wpca": "visit", "cold_mask_opd": "exposure", "defocus3": "exposure"}


# %%
wid = 64
oversample = 4
pad = 2      # as calibrator-joint: 2 extra detector pixels each side, cropped after resampling

nwavels = 200
npoly = 8

wpca_file = "../data/wavefront_pca.npz"
wpca = np.load(wpca_file)
n_wpca = wpca["modes"].shape[0]
n_modes = int(round(wpca["mean"].size**0.5))   # the fits' Fourier grid
n_zernikes = 50                                 # as calibrator-joint; the PCA removed this span (calibrator-pca)

optics = NICMOSFresnelCoronagraph(512, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20,
                                  wavefront_pca=wpca_file, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_F160W_{nwavels}.npy")[:, :npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'

# One exposure at one roll by default. The visit: roll 35.3 deg = n918b1e5q, e6q, e7q (192 s), e8q (288 s);
# roll 65.2 deg = n918b2ebq, ecq, edq (192 s), eeq (288 s)
roots = sys.argv[1:] or ["n918b1e5q"]
load = lambda fit: [exposure_from_file(ddir + f'{root}_o_clc_calf.fits', fit, crop=wid, extra_bad=None, hole_margin=0.1)
                    for root in roots]
exposures = {"binary": load(WavefrontPCABinaryFit(spectrum_basis, "F160W")),
             "single": load(WavefrontPCAFit(spectrum_basis, "F160W"))}
for exp in exposures["binary"]:
    print(exp.key, exp.target, f"ORIENTAT {exp.orient:.1f}", f"{exp.exptime:.0f} s")


# %%
# Literature values for B relative to A (Chauvin et al. 2010, Table 6; VLT/NACO Ks, 2007-06-05), used as the starting
# point only. Orbital motion between that epoch and 2005-03-15 is unknown
sep, pa, dmag = 0.220, 191.3, 2.5     # arcsec, deg (east of north), Ks
companion_offset = np.array([sep*np.sin(np.deg2rad(pa)), sep*np.cos(np.deg2rad(pa))])   # (east, north)
companion_contrast = 10**(-0.4*dmag)
print("companion start (east, north) =", companion_offset, "arcsec; contrast =", float(companion_contrast))

optical_params = np.load("../data/calibrator_optical_params.npy", allow_pickle=True)[()]
per_exposure_init = ["cold_mask_shift", "cold_mask_opd", "defocus1", "defocus2"]   # medians as starting values
render = eqx.filter_jit(lambda m, e: e.fit(m, e))


def initial_params(exps, binary):
    names = ["spectrum", "primary_wpca", "primary_low", "primary_tilt", "cold_mask_tilt", "bias"] + per_exposure_init
    if binary:
        names += ["companion_offset", "companion_contrast", "companion_spectrum"]
    params = {k: {} for k in names}
    for exp in exps:
        key = lambda k: exp.fit.get_key(exp, k)
        params["spectrum"][key("spectrum")] = np.zeros(npoly).at[0].set(1.)          # rescaled below
        params["primary_tilt"][key("primary_tilt")] = np.array([-0.05, -0.75])*0.075
        params["cold_mask_tilt"][key("cold_mask_tilt")] = hole_tilt(exp)             # as calibrator-joint
        params["primary_wpca"][key("primary_wpca")] = np.zeros(n_wpca)               # 0 = the calibrators' mean wavefront
        params["primary_low"][key("primary_low")] = np.zeros(n_zernikes)
        params["bias"][key("bias")] = 0.
        if binary:
            params["companion_offset"][key("companion_offset")] = companion_offset
            params["companion_contrast"][key("companion_contrast")] = np.asarray(companion_contrast)
            params["companion_spectrum"][key("companion_spectrum")] = np.zeros(npoly - 1)   # reset to A's fitted shape before stage 2
        for k in per_exposure_init:
            params[k][key(k)] = np.asarray(optical_params[k]["global"])
    params |= {k: v for k, v in optical_params.items() if k not in per_exposure_init}

    model = set_array(NICMOSModel(exps, params, optics, detector))
    params = ModelParams(params)
    # Star flux: match the total flux of the data over the unmasked pixels (as turboklip). The companion's flux is
    # contrast x the star's, so it scales with it
    im = [render(params.inject(model), e) for e in exps]
    c = sum(np.nansum(np.where(e.bad, 0., e.data)) for e in exps) / sum(np.nansum(np.where(e.bad, 0., s)) for e, s in zip(exps, im))
    return model, ModelParams(params.params | {"spectrum": jax.tree.map(lambda s: s*c, params.params["spectrum"])})


# %%
# Schedules. Stage 1: as turboklip (companion_* held at the literature values). Stage 2: the stage-1 parameters with
# no delay, plus (binary only) the companion. Learning rates raised x3 relative to stage 1 for the parameters that
# drift slowly in the calibrator fits (cold_mask_tilt, cold_mask_opd, defocus1) and for the companion offset; the
# drift diagnostic printed at the end shows which (if any) still need raising
g = 5e-2
n_stage1, n_stage2 = 300, 1000
precond = dict(precond_method="gn_exact", precond_kwargs=dict(chunk=1))   # every fitted leaf is small: exact GN, forward mode only

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
stage2_rates = {"spectrum": 3, "primary_tilt": 5, "cold_mask_tilt": 3, "cold_mask_opd": 0.3, "defocus1": 3,
                "defocus2": 10, "bias": 3, "cold_mask_shift": 0.3, "primary_low": 2, "primary_wpca": 5}
companion_things = {
    "companion_offset": sgd(g*1, 20),
    "companion_contrast": sgd(g*1, 20),
    "companion_spectrum": sgd(g*0.3, 50),
}


def fit(label, binary):
    exps = exposures[label]
    model, params = initial_params(exps, binary)
    plot_comparison(model, params, exps, percentile=99, wf_size=512, save=f"{out}/{label}-initial-comparison")

    opt_params = set_array({k: v for k, v in params.params.items() if k in things_start})
    losses1, hist1 = optimise_new(opt_params, model, exps, things_start, n_stage1, **precond)

    stage1 = params.params | hist1[-1]
    things = {k: sgd(g*lr, 0) for k, lr in stage2_rates.items()}
    if binary:
        # B starts from A's fitted spectral shape (A's coefficients over its first one); its flux is set by the contrast
        stage1["companion_spectrum"] = {k: stage1["spectrum"][k][1:]/stage1["spectrum"][k][0] for k in stage1["companion_spectrum"]}
        things |= companion_things
    model = ModelParams(stage1).inject(model)
    opt_params = set_array({k: v for k, v in stage1.items() if k in things})
    losses2, hist2 = optimise_new(opt_params, model, exps, things, n_stage2, **precond)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(losses1); ax[0].set(title=f"{label}: stage 1", xlabel="step", ylabel="loss")
    ax[1].plot(losses2); ax[1].set(title=f"{label}: stage 2", xlabel="step")
    fig.savefig(f"{out}/{label}-losses.png"); plt.close(fig)
    plot_params(hist2, list(things.keys()), xw=3, save=f"{out}/{label}-params")
    final = stage1 | hist2[-1]
    np.save(f"{out}/{label}-params.npy", final)
    np.save(f"{out}/{label}-losses.npy", numpy.concatenate([numpy.asarray(losses1), numpy.asarray(losses2)]))
    return dict(model=model, final=ModelParams(final), hist=hist2, losses=numpy.asarray(losses2), fitted=list(things))


results = {"single": fit("single", False), "binary": fit("binary", True)}


# %%
# Goodness of fit over the likelihood's pixels (finite error, not flagged bad)
def chi2(label):
    r = results[label]; m = r["final"].inject(r["model"])
    out_ = 0.
    for e in exposures[label]:
        z = numpy.asarray((e.data - render(m, e))/e.err)[~numpy.asarray(e.bad)]
        out_ += numpy.nansum(z**2)
    return float(out_)

n_pix = sum(int(numpy.sum(~numpy.asarray(e.bad) & numpy.isfinite(numpy.asarray(e.err)))) for e in exposures["binary"])
n_par = {l: sum(int(numpy.size(x)) for x in jax.tree.leaves({k: results[l]["final"].params[k] for k in results[l]["fitted"]})) for l in results}
stats = {l: dict(chi2=chi2(l), k=n_par[l]) for l in results}
for l, s_ in stats.items():
    s_["chi2_red"] = s_["chi2"]/(n_pix - s_["k"]); s_["bic"] = s_["chi2"] + s_["k"]*numpy.log(n_pix)
scale = stats["binary"]["chi2_red"]       # rescaled comparison: errors inflated to make the binary's chi2_red = 1
lines = [f"pixels {n_pix}"] + [f"{l}: chi2 {s_['chi2']:.1f}, k {s_['k']}, chi2_red {s_['chi2_red']:.3f}, BIC {s_['bic']:.1f}" for l, s_ in stats.items()]
lines += [f"delta chi2 (single - binary) {stats['single']['chi2'] - stats['binary']['chi2']:.1f}",
          f"delta BIC (single - binary) {stats['single']['bic'] - stats['binary']['bic']:.1f}  (positive favours the binary)",
          f"delta BIC with errors inflated by sqrt(chi2_red binary = {scale:.2f}): "
          f"{(stats['single']['chi2'] - stats['binary']['chi2'])/scale - (n_par['binary'] - n_par['single'])*numpy.log(n_pix):.1f}"]
print("\n".join(lines)); open(f"{out}/model_comparison.txt", "w").write("\n".join(lines) + "\n")


# %%
# 9-panel comparison per exposure:
#   data | single-star model | data - single-star model
#   data | binary model      | data - binary model
#   data | binary primary only (the binary fit with contrast 0) | data - primary only (B's signal + residual)
# Stretch as plotting.plot_comparison: columns 1-2 quarter-power with one shared limit, column 3 bwr with one symmetric
# limit for all rows. Masked pixels black on images, white on residuals. Dashed circle:
# the 0.3" hole about TARSIAFX/Y; cross: B's fitted offset from the hole centre via the exposure WCS (the star sits
# within ~1 px of the hole centre, so this marker is approximate)
mb = results["binary"]["final"].inject(results["binary"]["model"])
ms = results["single"]["final"].inject(results["single"]["model"])
for e_b, e_s in zip(exposures["binary"], exposures["single"]):
    data = numpy.where(numpy.asarray(e_b.bad), numpy.nan, numpy.asarray(e_b.data))
    im_s, im_b = numpy.asarray(render(ms, e_s)), numpy.asarray(render(mb, e_b))
    im_p = numpy.asarray(render(mb.set(e_b.fit.map_param(e_b, "companion_contrast"), np.asarray(0.)), e_b))
    rows = [("single star", im_s), ("binary", im_b), ("binary primary only", im_p)]
    # As plotting.plot_comparison: images on a quarter-power stretch with one colour limit (the max over the data and
    # every model image), residuals on bwr symmetric about zero; masked pixels black on images, white on residuals
    stretch = lambda a: numpy.where(numpy.isfinite(data), a, numpy.nan)**0.25
    smax = max(numpy.nanmax(stretch(data)), *[numpy.nanmax(stretch(m_)) for _, m_ in rows])
    rmax = numpy.nanpercentile(numpy.abs(numpy.concatenate([(data - m_).ravel() for _, m_ in rows])), 99)
    img_cmap = matplotlib.colormaps["inferno"].copy(); img_cmap.set_bad("k", 1)
    res_cmap = matplotlib.colormaps["bwr"].copy(); res_cmap.set_bad("w", 1)
    x0, y0 = CROP_CENTRE[0] - wid//2, CROP_CENTRE[1] - wid//2
    hx, hy = float(e_b.hdr["TARSIAFX"]) - x0, float(e_b.hdr["TARSIAFY"]) - y0
    east, north = numpy.asarray(results["binary"]["final"].get(e_b.fit.map_param(e_b, "companion_offset")))
    sky = e_b.wcs.pixel_to_world(float(e_b.hdr["TARSIAFX"]), float(e_b.hdr["TARSIAFY"]))
    bx, by = [float(v) for v in e_b.wcs.world_to_pixel(sky.spherical_offsets_by(east*u.arcsec, north*u.arcsec))]
    bx, by = bx - x0, by - y0
    fig, axs = plt.subplots(3, 3, figsize=(19, 17), layout="constrained")
    for i, (name, m_) in enumerate(rows):
        for j, (img, title, kw) in enumerate([(stretch(data), "data", dict(cmap=img_cmap, vmin=0, vmax=smax)),
                                              (stretch(m_), name, dict(cmap=img_cmap, vmin=0, vmax=smax)),
                                              (numpy.where(numpy.isfinite(data), data - m_, numpy.nan), f"data - {name}",
                                               dict(cmap=res_cmap, vmin=-rmax, vmax=rmax))]):
            a = axs[i, j]; h = a.imshow(img, **kw); a.set_title(title, fontsize=18); a.set_xticks([]); a.set_yticks([])
            a.add_patch(plt.Circle((hx, hy), 0.3/0.0756, fill=False, color="w" if j < 2 else "k", lw=1, ls="--"))
            a.plot(bx, by, "+", color="c" if j < 2 else "g", ms=14, mew=2)
            if i == 2:
                fig.colorbar(h, ax=axs[:, j], shrink=0.6, location="bottom", label="(DN/s)$^{1/4}$" if j < 2 else "DN/s")
    fig.suptitle(f"HD 155915 {e_b.key} (ORIENTAT {e_b.orient:.1f}): single star vs binary", fontsize=20)
    fig.savefig(f"{out}/model_comparison_{e_b.key}.png", dpi=72); plt.close(fig)


# %%
# Fisher information of the binary fit, marginalised over every fitted parameter.
#   F = J^T W J (Gauss-Newton, the expected information) with J the image Jacobian w.r.t. the flattened fitted params
#   and W = 1/err^2 on the likelihood's pixels (whitening by the per-pixel noise; masked pixels have weight 0). Columns
#   by forward mode, one per parameter, with the tangent built from a traced index (a compile-time-constant tangent can
#   miscompile on this JAX version). Parameters shared across exposures (target/visit keys) map to the same column,
#   so F sums correctly over exposures.
#   F is rescaled by sqrt(diag F) before inversion (units span ~1e-6 to ~1e5), inverted by Cholesky (no pseudo-
#   inverse: a degenerate direction is reported, not silently dropped), and the companion block of the covariance is
#   the marginal one. Errors are given raw and inflated by chi2_red (the residuals are not white: a fixed ~1.5% pixel
#   pattern and PSF error remain, so even the inflated errors are likely underestimates). Valid at a minimum only: see
#   the drift diagnostic. Calibrator optics are fixed, so the errors are conditional on them; no ORIENTAT, plate-scale
#   or distortion systematics are included.
rb = results["binary"]
fitted = {k: rb["final"].params[k] for k in rb["fitted"]}
x0_, unravel = ravel_pytree(jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), fitted))
index = unravel(np.arange(x0_.size, dtype=np.float64))     # flat index of every fitted entry


def weighted_image(x, e):
    m = ModelParams(unravel(x)).inject(rb["model"])
    sw = np.where(e.bad | ~np.isfinite(e.err), 0., 1/np.where(np.isfinite(e.err) & (e.err > 0), e.err, 1.))
    return (np.nan_to_num(e.fit(m, e)) * sw).ravel()


@eqx.filter_jit
def jac_column(x, i, e):
    return jax.jvp(lambda xx: weighted_image(xx, e), (x,), (jax.nn.one_hot(i, x.size, dtype=x.dtype),))[1]


F = numpy.zeros((x0_.size, x0_.size))
for e in exposures["binary"]:
    J = numpy.stack([numpy.asarray(jac_column(x0_, np.asarray(i), e)) for i in range(x0_.size)])
    F += J @ J.T
print(f"Fisher: {x0_.size} fitted parameters", flush=True)

s = numpy.sqrt(numpy.diag(F)); Fn = F/numpy.outer(s, s)
ev = numpy.linalg.eigvalsh(Fn)
fisher_out = dict(F=F, eig=ev, cond=ev.max()/max(ev.min(), 1e-300))
try:
    L = numpy.linalg.cholesky(Fn)
    Linv = numpy.linalg.inv(L); cov = (Linv.T @ Linv)/numpy.outer(s, s)
except numpy.linalg.LinAlgError:
    cov = None
    w_, v_ = numpy.linalg.eigh(Fn)
    names = [f"{k}{numpy.unravel_index(j, numpy.shape(leaf))}" for k, sub in sorted(index.items())
             for leaf in jax.tree.leaves(sub) for j in range(int(numpy.size(leaf)))]
    top = numpy.argsort(-numpy.abs(v_[:, 0]))[:8]
    print("Fisher matrix is not positive definite after rescaling; smallest eigenvalue %.3g; its eigenvector is "
          "dominated by: %s" % (w_[0], ", ".join(f"{names[t]} ({v_[t, 0]:+.2f})" for t in top)))

key = exposures["binary"][0].fit.get_key(exposures["binary"][0], "companion_offset")
ix = numpy.concatenate([numpy.asarray(index["companion_offset"][key], int).ravel(), numpy.asarray(index["companion_contrast"][key], int).ravel()])
east, north = numpy.asarray(fitted["companion_offset"][key], float); contrast = float(fitted["companion_contrast"][key])
sep_, pa_ = numpy.hypot(east, north), numpy.rad2deg(numpy.arctan2(east, north)) % 360
lines = [f"B: sep {sep_*1e3:.2f} mas, PA {pa_:.2f} deg, contrast {contrast:.5f}",
         f"literature start (NACO Ks 2007): sep {sep*1e3:.0f} +/- 14 mas, PA {pa:.1f} +/- 0.8 deg, contrast {float(companion_contrast):.4f} (dKs {dmag} +/- 0.1)",
         f"Fisher condition number (rescaled) {fisher_out['cond']:.3g}"]
if cov is not None:
    C = cov[numpy.ix_(ix, ix)]                                   # marginal covariance of (east, north, contrast)
    Ccond = numpy.linalg.inv(F[numpy.ix_(ix, ix)])               # conditional on every other parameter
    # delta method: (east, north, contrast) -> (sep, PA, contrast)
    D = numpy.array([[east/sep_, north/sep_, 0.],
                     [numpy.rad2deg(north/sep_**2), numpy.rad2deg(-east/sep_**2), 0.],
                     [0., 0., 1.]])
    for lab, Cx in (("marginal", C), ("conditional", Ccond)):
        S = D @ Cx @ D.T
        for infl, f_ in (("raw", 1.), (f"x chi2_red {scale:.2f}", scale)):
            e_ = numpy.sqrt(numpy.diag(S)*f_)
            dm = f", dF160W {-2.5*numpy.log10(contrast):.3f} +/- {2.5/numpy.log(10)*e_[2]/contrast:.3f}" if contrast > 3*e_[2] else \
                 ", contrast < 3 sigma: no dF160W quoted"
            lines.append(f"{lab:11s} {infl:16s}: sep +/- {e_[0]*1e3:.3f} mas, PA +/- {e_[1]:.3f} deg, contrast +/- {e_[2]:.5f}{dm}")
    corr = C/numpy.sqrt(numpy.outer(numpy.diag(C), numpy.diag(C)))
    lines.append("marginal correlation (east, north, contrast):\n" + numpy.array2string(corr, precision=3))
    fisher_out |= dict(cov=cov, cov_companion=C, cov_companion_conditional=Ccond)

    # Convergence: drift of each fitted group over the last 10% of stage 2, in units of its marginal sigma (raw)
    hist = rb["hist"]; n10 = max(len(hist)//10, 2)
    d_ = numpy.abs(ravel_pytree(jax.tree.map(lambda a: np.asarray(a, np.float64), hist[-1]))[0]
                   - ravel_pytree(jax.tree.map(lambda a: np.asarray(a, np.float64), hist[-n10]))[0])
    sig = numpy.sqrt(numpy.diag(cov))
    lines.append(f"loss change over the last 10% of stage 2: {(rb['losses'][-n10] - rb['losses'][-1])/abs(rb['losses'][-1]):.2e} (relative)")
    lines.append("drift over the last 10% of stage 2 / marginal sigma (max per group; > 1 = still moving, raise its learning rate):")
    for k in sorted(index):
        ii = numpy.concatenate([numpy.asarray(l_, int).ravel() for l_ in jax.tree.leaves(index[k])])
        lines.append(f"  {k:20s} {numpy.max(d_[ii]/sig[ii]):10.2f}")
print("\n".join(lines)); open(f"{out}/fisher.txt", "w").write("\n".join(lines) + "\n")
numpy.savez(f"{out}/fisher.npz", **{k: numpy.asarray(v) for k, v in fisher_out.items()})
