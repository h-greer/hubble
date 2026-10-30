"""DRAFT batch/imlup-gp-e2e.py: END-TO-END GP fit of IM Lup that does not depend on the regularised L1/TSV fit (imlup-adi/).
Same model as imlup-gp-radial.py (Matern 1/2 DCT disc prior + radial trend, support mask, loop_order = 1.5, stages C-G2, F) but
every input that imlup-gp-radial.py took from the L1/TSV fit is derived here from the data and the calibrators:
  A   (new) optics-only fit with the disc OFF (a star-only fit, models.SinglePointFit) from the CALIBRATOR optics
      (data/calibrator_optical_params.npy + per-exposure global entries, as gmaur-wpca.py), with the disc-dominated annulus
      masked FOR THIS STAGE ONLY. Initial tilts from the FITS header (TARSIAFX/Y) and hole_tilt, star flux by total-flux
      matching on the unmasked pixels. Schedule: imlup-adi's optics-first stage 1 (A1), then the hybrid refit of
      imlup-gp-radial.py (A2: short first-order stage + one L-BFGS call), both without any disc.
  L   level_mu from (data - stage-A star model) over the support in disc units, no reference disc (estimate_level).
  S   support mask from the stage-A fitted tilts (primary_tilt + cold_mask_tilt), or the header (support_star = "header").
  B   a NEW LinearResponse at the stage-A optics in this run's own directory (never imlup-gp's, which is built at the L1/TSV
      optics), verified against the full model with zero disc and random GP-draw discs (no reference disc).
  C-G2, F as imlup-gp-radial.py (noise scales, D grids, E outer loop with the hybrid refit, G/G2 comparisons, products), with
  one more outer iteration allowed (n_outer + 1) and a second pass forced (min_outer = 2: the first iteration only RE-LINEARISES,
  its grid is computed at optics that the refit has just moved away from the stage-A point), plus per-iteration optics deltas
  and re-linearisation diagnostics.
Flags (environment variables override the defaults; every result goes to a directory named by them):
  optics_source  "calibrator" (this script's stage A) | "l1tsv" (imlup-adi's fit as optics, no stage A: reproduces imlup-gp-radial.py,
                 the validation mode; its shared response is reused)
  level_source   "data" (estimate_level) | "l1tsv" (reference_level of imlup-adi's disc)
  support_star   "fitted" (stage-A / reference tilts) | "header" (TARSIAFX/Y minus the crop origin)
Needs: data/calibrator_optical_params.npy (calibrator mode); imlup-adi/ only for the l1tsv flags. Resumable and fingerprinted
(stage A, response, noise, grids, outer refits). The validation hook at the end compares with imlup-gp-radial-m12-support/ (or -m12/)
when present: comparison-l1tsv.json."""
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax.random as jr
import jax
import numpy
import os
import warnings
import json
import time
import hashlib
import glob
from contextlib import contextmanager
from astropy.nddata import Cutout2D
from astropy.io import fits
from scipy.ndimage import gaussian_filter

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache (directory from JAX_COMPILATION_CACHE_DIR, set in the .sh); cache every compile
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# Optimisation imports
import equinox as eqx
import zodiax as zdx
import optax
import optimistix as optx

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
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
from fitting import *
from fitting import _loss_grad  # underscore names are not covered by `import *`
from plotting import *
from spectra import *
from batching import make_batch
from gp import *
from gp import _save_npz, _axes, _with_disc   # the trend prior comes with `import *`; underscore names are not covered

# ---------------------------------------------------------------- flags (see the docstring)
optics_source = os.environ.get("E2E_OPTICS", "calibrator")      # "calibrator" | "l1tsv"
level_source = os.environ.get("E2E_LEVEL", "data")              # "data" | "l1tsv"
support_star = os.environ.get("E2E_SUPPORT_STAR", "fitted")     # "fitted" | "header"
tilt_init = os.environ.get("E2E_TILT_INIT", "header")           # calibrator mode: "header" (TARSIAFX/Y) | "adi" (imlup-adi's hand-set)
assert optics_source in ("calibrator", "l1tsv") and level_source in ("data", "l1tsv"), (optics_source, level_source)
assert support_star in ("fitted", "header") and tilt_init in ("header", "adi"), (support_star, tilt_init)
_suffix = "".join([f"-optics-{optics_source}" if optics_source != "calibrator" else "",
                   f"-level-{level_source}" if level_source != "data" else "",
                   "-hdrstar" if support_star == "header" else ""])
# every other non-default setting also names the directory (a second pass must not overwrite the first): short hash of the set
_nondefault = {k_: os.environ[k_] for k_ in ("E2E_TILT_INIT", "E2E_LEVEL_SD", "E2E_LEVEL_MU", "E2E_MASK_RIN", "E2E_MASK_ROUT")
               if k_ in os.environ and not (k_ == "E2E_TILT_INIT" and os.environ[k_] == "header")}
if _nondefault:
    _suffix += "-" + hashlib.sha256(repr(sorted(_nondefault.items())).encode()).hexdigest()[:6]
out = "imlup-gp-e2e" + _suffix        # the default run (calibrator optics, data level, fitted star) is plain imlup-gp-e2e/
ref_dir = "imlup-adi"                # read only with an l1tsv flag
shared_dir = "imlup-gp"              # its response.npz / verify-*.npz are reused ONLY with optics_source = "l1tsv" (the L1/TSV optics)
compare_dirs = ["imlup-gp-radial-m12-support", "imlup-gp-radial-m12"]   # validation hook: first one with a summary.json
calibrator_file = "../data/calibrator_optical_params.npy"
os.makedirs(out, exist_ok=True)
with open(f"{out}/flags.json", "w") as f_:
    json.dump(dict(optics_source=optics_source, level_source=level_source, support_star=support_star, tilt_init=tilt_init, nondefault_env=_nondefault), f_, indent=2)

@contextmanager
def timed(stage):
    """Wall time of a stage, printed and appended to {out}/timings.txt (results are pulled to the host inside)."""
    t = time.perf_counter()
    yield
    dt = time.perf_counter() - t
    print(f"[timing] {stage}: {dt:.1f} s", flush=True)
    with open(f"{out}/timings.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage}: {dt:.1f} s\n")


def to_host(tree):
    return jax.tree.map(numpy.asarray, tree)


def fingerprint(*objs):
    """Short sha256 of arrays / scalars / strings, for the names of resumable results."""
    h = hashlib.sha256()
    for o in objs:
        a = numpy.asarray(o)
        h.update(a.tobytes() if a.dtype != object else repr(o).encode())
        h.update(str(a.shape).encode())
    return h.hexdigest()[:12]


def save_npy(path, obj):
    """Atomic numpy.save (a job killed mid-write must not leave a file that a rerun would trust)."""
    tmp = f"{path}.tmp{os.getpid()}.npy"
    numpy.save(tmp, obj)
    os.replace(tmp, path)


def response_fingerprint(resp):
    return fingerprint(resp.G, resp.b, resp.yy, resp.n_good, resp.lognorm, repr(resp.keys))


code_fp = hashlib.sha256(open("../gp.py", "rb").read()).hexdigest()[:12]   # results depend on the maths of gp.py

# %%
# As imlup-adi
wf_wid = 512
wid = 80
oversample = 4
pad = 2
nwavels = 200
npoly = 8
n_modes = 50
n_zernikes = 50
resolved_wid = 60          # 5.18" box at 0.0864" / px; a larger box is a later evidence comparison (gp-plan.md)
roots = ["n8zu11epq", "n8zu11eqq", "n8zu12exq", "n8zu12eyq"]


# Hyperparameter grids (gp_evidence_grid_nd extends an axis by one step while the optimum is on an edge, within its bounds),
# the same for the plain Matern and the Matern + trend. sigma: rms of the DCT field in dex; lengths in 0.0864" px
# (lambda/D at F160W = 0.1375" = 1.6 px), log-spaced by 2 and anchored so that the top cell reaches the 200 px bound.
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [200/2**0.5/2**i for i in range(5, -1, -1)]          # 4.42 .. 141.4 (6 values; growth reaches 0.1 below)
hyper_bounds = dict(sigma=(0.03, 8.), length=(0.1, 200.))        # log-uniform hyperpriors, the SAME for both models
ABBREV = {"sigma": "s", "length": "l"}
orders = [1., 1.5, 2., 2.5, 3.5]   # 2-D Matern nu = order - 1: nu 0 (TSV-like), 1/2, 1, 3/2, 5/2; with and without the trend
loop_order = 1.5            # the outer loop's Matern order: 1.5 = nu 1/2 (exponential), the final-response winner of the
                            # order-1 loop's stage G; None: stage D's best trend order (which ranked order 1 first)
trend_kinds = ("log_r", "r2")   # radial basis (gp.py radial_trend_basis): log10 sqrt(r^2 + 1 px^2), then r^2, orthonormalised
trend_sigma = 1.            # dex, rms contribution of each unit-rms basis column: FIXED broad (the evidence is insensitive to
                            # it: NOTES.md), not a grid axis, so the trend adds no hyperprior volume
trend_sigma_check = (0.3, 1., 3., 10.)   # reported sensitivity of log Z at the best point
refine_below = 3.           # nats: |dlogE| below this is re-evaluated on finer local grids before deciding
recheck_top = 3             # cold restarts of the best points
templates = [False]   # flat prior mean only here
# The template is estimated from these same data, so its variants' evidence is optimistic: they are reported, but
# stages E and F use the best flat-mean variant unless this is set True
final_allows_template = False
template_floor = 1e-3
# level_sd: see the end-to-end configuration below
n_iter = 200                # at most this many Levenberg-Marquardt steps per MAP (warm-started; stops when converged)
response_batch = 16         # disc-pixel jvps per batch when building the response

# Outer iteration (stage E): full refit at the chosen hyperparameters, as calibrator-joint / betapic-sparco
# (mapped=True: lax.map over exposures; per_block=False: one preconditioner compile)
precond_kwargs = dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32)
# The refit is the hybrid of imlup-gp-polar-fast.py / calibrator-joint.py: (a) a short first-order stage (the old adam/sgd
# schedule but WITHOUT the delayed adam(2e-2, 50) kick on primary_opd, whose first Adam step threw the loss up by ~250-300
# at epoch 50; small Adam rates from step 0 instead), then (b) ONE optimise_optimistix call (L-BFGS in u = (x - x0) /
# sqrt(C), never accepts a worse point) with C = gn_probe for ALL parameters including the disc.
fo_epochs = 200         # first-order epochs (~1.15 s each at production size)
fo_lr_opd = 5e-3        # Adam rate on primary_opd from step 0 (the old one: 2e-2 from epoch 50)
fo_lr_disc = 3e-3       # Adam rate on the disc latents z, C = 1 (the old one: 1e-2, which raised the loss by ~470)
optx_max_steps = 800    # L-BFGS iterations, accepted AND rejected line-search trials (each costs one loss+grad)
# Cost of an outer iteration (timings.txt of the first run): refit 400-840 s (the first one pays the compiles) + response 250 s +
# grid 12-15 s ~ 11-17 min, so n_outer = 4 adds at most ~25 min to the 33 min run (12 h allocation); the loop exits early once
# the hyperparameters stop moving (exact test) or have moved by <= 1 grid step at < converge_dlogz nats (a flat length plateau
# makes an exact argmax test wobble: the optics refitted at the old point are then as good as at the new one).
# n_outer: see the end-to-end configuration below
outer_half_width = 2        # initial local grid: 2n + 1 = 5 values per axis (was 3: the first run's 3 x 3 box + 2 growths was too narrow)
outer_max_growth = 8        # growth steps (one cell per face near the maximum, within hyper_bounds); caps each axis at 5 + 2*8 = 21
                            # values, so at most 441 points (typically a few dozen at ~0.5-1 s each)
grid_edge_dlogz = 6.        # grow a face while it holds a cell within this many nats of the best (gp default 5)
edge_target = 0.05          # a final grid is "contained" if its edge weight is below this
converge_dlogz = 0.5        # None: only the exact "best cell did not move" test
# Final grids (trend and plain): the hyperparameter posterior is very peaked (sd(ln sigma) ~ 0.028, sd(log10 length) ~ 0.09, correlation
# ~0.7), so coarse cells put ~all weight in one cell (sigma interval overstated ~3x, evidence biased ~2+ nats). Per-axis fine grids
# about the best point: sigma at ratio 2^(1/16) x 9 values (+-4 steps = +-0.17 in log2 sigma), length at ratio
# 2^(1/4) x 7 values (+-3 steps = +-0.23 dex); 63 points, grown within hyper_bounds if the edge weight is not small (at most
# 6 growth steps: <= 21 x 19 = 399 points; ~1.3-1.5 s/point, so ~1.5 min typical, 9 min worst, per grid; 3 fine grids).
fine_steps = dict(sigma=2**(1/16), length=2**0.25)
fine_half_width = dict(sigma=4, length=3)
fine_max_growth = 6
def fine_axes(center):
    """Per-axis fine local axes {name: values} about center (local_axes takes one half-width for all axes)."""
    return {k_: [c_*fine_steps[k_]**i_ for i_ in range(-fine_half_width[k_], fine_half_width[k_] + 1)] for k_, c_ in center.items()}
backfill_grad_split = False # True: compute a |sqrt(C) g| split for resumed refits lacking one (a fresh gn_probe compile + run per resumed outer,
                            # fresh C so not comparable with grad_norm_u_end; a GPU OOM there is not caught): off by default

if os.path.isdir('../data/data/'):
    ddir = '../data/data/'                 # the cluster copy, as imlup-adi
else:
    ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'
    print(f"WARNING: ../data/data/ not found; using {ddir}. The reference (imlup-adi) must have been fitted to the "
          "same reduction, which the linearity check (stage B) cannot detect")
print(f"data directory: {ddir}")

# %%
# ---------------------------------------------------------------- end-to-end configuration (new)
# STAGE A mask. The annulus r_in <= r <= r_out (arcsec from the header star position) is masked FOR STAGE A ONLY (stages B onward use
# every good pixel, the disc is then fitted). Which pixels constrain the OPTICS best, disc-free?
#  keep r < r_in (the ring between the hole mask's edge, 0.3" + hole_margin 0.3" = 0.6", and r_in): the coronagraphic residual there
#     is the brightest and most informative light: it fixes the star's position (tilts, cold-mask shift), the occulter radius /
#     ellipticity / defocus1-2 and the low-order Zernikes, and the star flux. The disc is a minority there (the L1/TSV disc's
#     median is 15-30 % of the data within 1", ~4x below the star halo: batch/imlup-adi), but not zero: its inner part is the
#     biggest risk for the star-flux and tilt estimates (the E loop refits both with the disc on);
#  mask r_in <= r <= r_out: the disc footprint (IM Lup's scattered light reaches ~1.2" in SPHERE, ~2.5" in NICMOS at its 158 pc), where
#     a star-only model is asked to explain light that is largely smooth disc: it would bend the low-order wavefront and the
#     spectrum to absorb it;
#  keep r > r_out (the crop corners, ~55 % of the area beyond 2.4"): the disc is faint there (L1/TSV: ~13 % of the data at 2.5-3")
#     and the halo and diffraction spikes (spiders: primary_secondary, spider_width, secondary_radius, shift) and the high-order
#     speckles of primary_opd (Fourier mode k at k lambda / 3 m: modes up to k ~ 33 reach 3.2") live there.
# The defaults are an a-priori compromise, not tuned: stage A reports the star-only residual in the masked annulus (a smooth positive
# excess = the disc; a structured / negative one = a bad star model) and chi2 / N kept vs masked. Smaller r_out -> more optics
# information, more disc contamination; E2E_MASK_RIN / E2E_MASK_ROUT override.
stage_a_mask_radii = (float(os.environ.get("E2E_MASK_RIN", 0.9)), float(os.environ.get("E2E_MASK_ROUT", 2.4)))
a1_epochs = 300             # A1: imlup-adi's stage 1 schedule (500 epochs from hand-set optics; the calibrator state needs fewer)
level_sd = float(os.environ.get("E2E_LEVEL_SD", 2.0 if level_source == "data" else 1.0))   # dex; first pass 1.5-2 (see estimate_level),
                            # 1 when the level comes from the L1/TSV disc (imlup-gp-radial.py's)
level_mu_override = os.environ.get("E2E_LEVEL_MU")     # dex: set the prior centre by hand (e.g. a first pass's final level for a second pass)
level_smooth_px = 2.0       # estimate_level: Gaussian smoothing of the residual (data px, sigma) before the log
level_k_floor = 2.0         # estimate_level: pixels below k sigma of the smoothed noise are floored at k sigma (upper limits)
level_f_min = 0.5           # estimate_level: use pixels whose unit-disc response is >= this fraction of its median (drops the occulter / edge rims)
n_outer = 5                 # one more than imlup-gp-radial.py's 4 (stage A starts further from the optimum than the L1/TSV optics)
min_outer = 2               # at least this many outer iterations before "converged" (the first only re-linearises at the refit optics)
lin_dlogz_warn = 5.         # nats: warn if re-linearising changed log Z at fixed hyperparameters (and noise) by more than this
verify_draw_sigmas = (0.5, 1.0)   # dex rms of the random GP-draw test discs in the response verification (zero disc is always included)

# ---------------------------------------------------------------- pure helpers (tested in a tiny exec harness, no jax needed)
def stage_a_mask(shape, star_cr, scale, r_in, r_out):
    """bool (shape): True (masked in stage A) where r_in <= r <= r_out, r the distance in arcsec from the star at crop pixel
    (col, row) = star_cr (0-indexed pixel centres), scale arcsec / data pixel."""
    rows, cols = numpy.indices(shape)
    r = numpy.hypot(cols - star_cr[0], rows - star_cr[1])*scale
    return (r >= r_in) & (r <= r_out)


def estimate_level(data, err, good, star, f, smooth=2.0, k_floor=2.0, f_min_frac=0.5):
    """The prior centre of resolved_level (log10 of the geometric-mean disc pixel, in model disc units) from data - star, no reference
    disc. Lists of (ny, nx) arrays, one per exposure: data and err (any value where ~good), good (bool), star (the star-only model
    image, DN/s per data pixel) and f (the model image of a UNIT-valued disc over the support minus the star image: DN/s per data
    pixel per unit of disc value, from the full forward model).

    Units. The disc value d_j of a disc pixel is a point source of flux d_j at that sky position (the star has `point.flux`): its PSF
    deposits d_j x (that PSF's integral over the crop, i.e. the throughput of coronagraph + crop) DN/s on the detector. For a disc that
    is smooth on the PSF scale the surface brightness at a data pixel (area a = scale_x scale_y) is then
        data - star = f d_local,   f = (a / pitch^2) x T   (analytic: (0.0755/0.0864)^2 = 0.764 times the throughput T <= 1,
        with pitch = 0.0864" the disc pixel and T = 1 for no PSF loss and no occulter attenuation),
    because a lattice of point sources of pitch p and flux d has surface density d / p^2. Rather than trust T, f is MEASURED per pixel
    from the model itself (unit disc render minus the star render), which carries the PSF-convolution (halo flux from the whole
    disc, not just the local pixel), the occulter attenuation of sources within ~0.3" of the star, the roll and the support
    truncation at the crop edge. Caveats: (i) a PSF-smooth f assumes the disc is smooth on the PSF core scale; the bright inner disc leaks
    through the halo into the faint outer pixels, so d_local is OVER-estimated there (the level is a mean over the box, dominated by
    the outer pixels: biased high by at most a few tenths of a dex); (ii) pixels near the occulter (f < f_min_frac x median f) are
    dropped, so the brightest inner disc is under-represented (biased low): the two roughly cancel, hence a level_sd of 1.5-2 dex
    for a first pass; (iii) unsubtracted star residuals add to data - star (speckles; smoothing averages them down, the floor stops
    noise from entering the log); (iv) `level` is the mean of log10 d over the WHOLE support, which is what is estimated here, not
    the log of the mean.
    Smoothing: Gaussian, sigma `smooth` px, normalised over the good pixels. d_s = (smoothed data - star)/f; per-pixel noise of the
    smoothed residual from err; d = max(d_s, k_floor sigma_d) (negative or noise-level pixels become upper limits, not -inf).
    Returns dict: level (pooled mean of log10 d over the used pixels of all exposures), level_per_exposure, n_pixels,
    frac_floored (share of used pixels at the floor), d_maps (list; nan where unused), f_median."""
    sums, counts, floored, per_exp, maps, fmeds = 0., 0, 0, [], [], []
    for y, s_, g, st, ff in zip(data, err, good, star, f):
        y, s_, st, ff = (numpy.asarray(a, dtype=float) for a in (y, s_, st, ff))
        w = numpy.asarray(g, dtype=bool) & numpy.isfinite(y) & numpy.isfinite(s_) & numpy.isfinite(ff) & numpy.isfinite(st)
        f_ref = float(numpy.median(ff[w])) if w.any() else numpy.nan
        w = w & (ff > f_min_frac*f_ref)
        wf = w.astype(float)
        kw = dict(mode="constant", truncate=3.)
        den = gaussian_filter(wf, smooth, **kw)
        ok = w & (den > 0.3)
        safe = numpy.where(den > 0, den, 1.)
        res = numpy.where(w, y - st, 0.)
        r_s = gaussian_filter(wf*res, smooth, **kw)/safe
        f_s = gaussian_filter(wf*numpy.where(w, ff, 0.), smooth, **kw)/safe
        var = gaussian_filter(wf*numpy.where(w, s_, 0.)**2, smooth/numpy.sqrt(2), **kw)/(4*numpy.pi*smooth**2)/safe**2
        with numpy.errstate(divide="ignore", invalid="ignore"):
            d_s, sd_d = r_s/f_s, numpy.sqrt(var)/f_s
            d = numpy.maximum(d_s, k_floor*sd_d)
        ok = ok & numpy.isfinite(d) & (d > 0)
        lg = numpy.where(ok, numpy.log10(numpy.where(ok, d, 1.)), numpy.nan)
        n = int(ok.sum())
        sums += float(numpy.nansum(lg)); counts += n
        floored += int((ok & (d_s < k_floor*sd_d)).sum())
        per_exp.append(float(numpy.nansum(lg)/n) if n else float("nan"))
        maps.append(lg)
        fmeds.append(f_ref)
    if counts == 0:
        raise RuntimeError("estimate_level: no usable pixels")
    return dict(level=sums/counts, level_per_exposure=per_exp, n_pixels=counts, frac_floored=floored/counts, d_maps=maps,
                f_median=fmeds)


def tilt_of(p, e, name):
    """Exposure e's tilt (arcsec, (x, y)) from the params dict p: keyed by get_key, or per exposure (cold_mask_tilt is "global" in
    PARAM_KEYS but stored per exposure in the imlup-adi params)."""
    d = p[name]
    for k_ in (e.fit.get_key(e, name), e.key):
        if k_ in d:
            return numpy.asarray(d[k_], dtype=float)
    raise KeyError(f"params[{name!r}] has neither {e.fit.get_key(e, name)!r} nor {e.key!r}: {list(d)}")


def star_pixel(p, e, wid, scale):
    """The model's star position (col, row) in crop pixels: the optical axis + primary_tilt + cold_mask_tilt (they all shift the PSF
    identically), (wid - 1)/2 + tilts / scale (arcsec / data px)."""
    return (wid - 1)/2 + (tilt_of(p, e, "primary_tilt") + tilt_of(p, e, "cold_mask_tilt"))/scale


def optics_deltas(p_old, p_new, exps, wid, scale):
    """How far the optics moved between two parameter dicts (JSON-safe): primary_opd rms change (nm, piston removed) and the new
    rms, star position change per exposure (data px), cold_mask_shift / defocus1 / defocus2 changes, occulter_radius change and
    the ratio of the leading spectrum coefficient (star flux, approximately)."""
    e0 = exps[0]
    ko = e0.fit.get_key(e0, "primary_opd")
    a, b = numpy.asarray(p_old["primary_opd"][ko], dtype=float).copy(), numpy.asarray(p_new["primary_opd"][ko], dtype=float).copy()
    a[0, 0] = b[0, 0] = 0.
    res = dict(primary_opd_rms_change_nm=float(numpy.sqrt(numpy.mean((b - a)**2))), primary_opd_rms_new_nm=float(numpy.sqrt(numpy.mean(b**2))),
               star_shift_px=[float(numpy.hypot(*(star_pixel(p_new, e, wid, scale) - star_pixel(p_old, e, wid, scale)))) for e in exps])
    for name in ("cold_mask_shift", "defocus1", "defocus2"):
        res[f"{name}_max_abs_change"] = float(max(numpy.max(numpy.abs(numpy.asarray(p_new[name][e.fit.get_key(e, name)], dtype=float)
                                                                      - numpy.asarray(p_old[name][e.fit.get_key(e, name)], dtype=float)))
                                                  for e in exps))
    res["occulter_radius_change"] = float(numpy.asarray(p_new["occulter_radius"]) - numpy.asarray(p_old["occulter_radius"]))
    ks = e0.fit.get_key(e0, "spectrum")
    res["spectrum0_ratio"] = float(numpy.asarray(p_new["spectrum"][ks])[0]/numpy.asarray(p_old["spectrum"][ks])[0])
    return res


def compare_runs(new, ref, wid, scale):
    """Validation hook: this run vs a reference run (imlup-gp-radial-m12-support/). new, ref: dicts with params (optics), disc
    (log10 MAP map, nan outside its support), support (wid x wid), logZ, evidence, hyper, exps (the exposure objects, for keys).
    Returns JSON-safe dict: final logZ and loop-model evidence (not strictly like-for-like: different level prior, noise scales and
    support), optics deltas (primary_opd rms difference and the references's rms, primary_tilt / cold_mask_tilt per exposure, arcsec
    and star position in data px) and the MAP disc difference within the common support (rms, median, 5/95 %, max |.|, correlation of
    the log10 maps, ratio of the linear-flux sums)."""
    exps = new["exps"]
    d = dict(logZ_new=new["logZ"], logZ_ref=ref["logZ"], logZ_new_minus_ref=new["logZ"] - ref["logZ"],
             evidence_new=new.get("evidence"), evidence_ref=ref.get("evidence"),
             hyper_new=new.get("hyper"), hyper_ref=ref.get("hyper"))
    if new.get("evidence") is not None and ref.get("evidence") is not None:
        d["evidence_new_minus_ref"] = new["evidence"] - ref["evidence"]
    d["optics"] = optics_deltas(ref["params"], new["params"], exps, wid, scale)
    d["optics"]["per_exposure_tilt_arcsec"] = {
        e.key: dict(primary_tilt_new=tilt_of(new["params"], e, "primary_tilt").tolist(), primary_tilt_ref=tilt_of(ref["params"], e, "primary_tilt").tolist(),
                    cold_mask_tilt_new=tilt_of(new["params"], e, "cold_mask_tilt").tolist(), cold_mask_tilt_ref=tilt_of(ref["params"], e, "cold_mask_tilt").tolist())
        for e in exps}
    sup = (numpy.asarray(new["support"]) > 0) & (numpy.asarray(ref["support"]) > 0)
    a, b = numpy.asarray(new["disc"], dtype=float), numpy.asarray(ref["disc"], dtype=float)
    ok = sup & numpy.isfinite(a) & numpy.isfinite(b)
    dd = (a - b)[ok]
    d["disc"] = dict(n_common=int(ok.sum()), n_new=int((numpy.asarray(new["support"]) > 0).sum()), n_ref=int((numpy.asarray(ref["support"]) > 0).sum()),
                     rms_dex=float(numpy.sqrt(numpy.mean(dd**2))), median_dex=float(numpy.median(dd)), p05_dex=float(numpy.percentile(dd, 5)),
                     p95_dex=float(numpy.percentile(dd, 95)), max_abs_dex=float(numpy.max(numpy.abs(dd))),
                     corr_log10=float(numpy.corrcoef(a[ok], b[ok])[0, 1]), flux_ratio=float((10**a[ok]).sum()/(10**b[ok]).sum()))
    return d



# %%
# ---------------------------------------------------------------- hybrid refit machinery (imlup-gp-radial.py's), shared by stage A and E
g_sgd = 5e-2
things_gp = {
    "primary_opd": adam(2e-2, 50),
    "spectrum": sgd(g_sgd*1, 0), "primary_tilt": sgd(g_sgd*1., 0), "cold_mask_tilt": sgd(g_sgd*1, 0),
    "cold_mask_opd": sgd(g_sgd*1., 0), "defocus1": sgd(g_sgd*1, 0), "defocus2": sgd(g_sgd*1, 0), "bias": sgd(g_sgd*3, 0),
    "cold_mask_shift": sgd(g_sgd*1, 0), "cold_mask_rot": sgd(g_sgd*1., 0.), "primary_rot": sgd(g_sgd*3., 0),
    "primary_low": sgd(g_sgd*1, 0), "cold_mask_scale": sgd(g_sgd*1, 0), "cold_mask_distortion": sgd(g_sgd*3, 0),
    "occulter_radius": sgd(g_sgd*1., 0), "occulter_ellipticity": sgd(g_sgd*3, 300), "secondary_radius": sgd(g_sgd*1., 0),
    "spider_width": sgd(g_sgd*1., 0), "primary_spider": sgd(g_sgd*1., 0), "primary_secondary": sgd(g_sgd*1., 0),
    "secondary_shift": sgd(g_sgd*1, 0), "primary_distortion": sgd(g_sgd*3, 0), "fnumber": sgd(g_sgd*1., 0),
    "anisotropy": sgd(g_sgd*1., 0), "distortion": sgd(g_sgd*1, 0),
    # (primary_opd and resolved are replaced by small Adam rates from step 0 in the first-order stage of refit())
    "resolved": adam(1e-2, 0), "resolved_level": sgd(g_sgd*1, 0),
}


def _path_str(p):
    return str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p))))


def grad_split(gr):
    """Split of the whitened gradient gr = sqrt(C) grad (a tree shaped like the parameters) by parameter group: norm (the
    total |sqrt(C) g|), groups {name: norm} (sorted, descending; "resolved" is split into resolved_dct (the DCT latents z)
    and resolved_trend (the n_b trend latents b); "resolved_level" is the level), frac_sq (share of |.|^2), top_leaves
    (the 8 largest single leaves, e.g. one exposure's primary_opd), dct_max_mode ((row, col, value) of the largest DCT
    entry) and trend_components (the n_b whitened trend gradients)."""
    nd = resolved_wid**2
    sq, leaves, extra = {}, [], {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(gr)[0]:
        a = numpy.asarray(leaf, dtype=float).ravel()
        name = _path_str(path[0])
        label = "/".join(_path_str(p) for p in path)
        parts = {name: a}
        if name == "resolved":
            parts = {"resolved_dct": a[:nd], "resolved_trend": a[nd:]}
            j = int(numpy.argmax(numpy.abs(a[:nd])))
            extra = dict(dct_max_mode=[j//resolved_wid, j % resolved_wid, float(a[j])],
                         trend_components=[float(v_) for v_ in a[nd:]])
        for nm, part in parts.items():
            sq[nm] = sq.get(nm, 0.) + float(numpy.sum(part**2))
            leaves.append((f"{label}" if nm == name else f"{label}[{nm}]", float(numpy.sqrt(numpy.sum(part**2)))))
    tot = sum(sq.values())
    order_ = sorted(sq, key=sq.get, reverse=True)
    return dict(norm=float(numpy.sqrt(tot)), groups={k_: float(numpy.sqrt(sq[k_])) for k_ in order_},
                frac_sq={k_: float(sq[k_]/tot) if tot > 0 else 0. for k_ in order_},
                top_leaves=sorted(leaves, key=lambda t_: -t_[1])[:8], **extra)


def precond_grad_split(x, C, model_b, exps_b):
    """grad_split of |sqrt(C) grad loss| at x: the gradient in L-BFGS's own coordinates u = (x - x0) / sqrt(C). The same
    _loss_grad call (already compiled by the refit) as before; norm is nan (and the split empty) if it fails."""
    try:
        _, gr = _loss_grad(x, jax.tree.map(np.sqrt, C), make_batch(exps_b, model_b), model_b)
        return grad_split(gr)
    except Exception as e:   # never lose a 15-minute refit to a diagnostic
        print(f"[refit] gradient norm failed: {e!r}")
        return dict(norm=float("nan"), groups={}, frac_sq={}, top_leaves=[])


def precond_grad_norm(x, C, model_b, exps_b):
    """|sqrt(C) grad loss| at x (nan if it fails)."""
    return precond_grad_split(x, C, model_b, exps_b)["norm"]


def print_split(label, sp, top=6):
    """The largest contributors to |sqrt(C) g|^2 of a grad_split."""
    try:
        _print_split(label, sp, top)
    except Exception as e:   # a diagnostic print must never discard a refit
        print(f"[grad] print failed: {e!r}")


def _print_split(label, sp, top):
    if not sp.get("groups"):
        return
    print(f"[grad] {label}: |sqrt(C) g| {sp['norm']:.3g}; top groups " + ", ".join(
        f"{k_} {sp['groups'][k_]:.3g} ({100*sp['frac_sq'][k_]:.0f}%)" for k_ in list(sp["groups"])[:top]), flush=True)
    print(f"[grad]   largest leaves: " + ", ".join(f"{l_} {v_:.3g}" for l_, v_ in sp["top_leaves"][:4])
          + (f"; trend {sp['trend_components']}; DCT max mode {sp['dct_max_mode']}" if "dct_max_mode" in sp else ""), flush=True)


def get_C(opt_params, model_b, exps_b, things):
    """The gn_probe preconditioner C for all parameters (one discarded 1-epoch step), as refit() computes it."""
    return optimise_new(opt_params, model_b, exps_b, things, 1, precond_method="gn_probe",
                        precond_kwargs=precond_kwargs, mapped=True, return_c=True, progress=False)[2]


def refit(opt_params, model_b, exps_b, things, log=print):
    """Hybrid refit of opt_params (the things keys): fo_epochs of first-order steps (C = 1 on the disc), then one L-BFGS
    call from there (C = gn_probe on everything). Returns (params, losses, info): losses = the first-order losses (before
    each step) followed by the L-BFGS end loss; info = optimise_optimistix's info dict (result, num_steps, loss_start,
    loss_end) plus grad_norm_u_start / grad_norm_u_end (|sqrt(C) grad| before / after L-BFGS) and max_steps."""
    # the preconditioner for all parameters (disc included) at the start: one gn_probe call, via a discarded 1-epoch step
    C = get_C(opt_params, model_b, exps_b, things)
    has_disc = "resolved" in opt_params      # stage A (no disc) has none
    C_fo = dict(C) | ({"resolved": jax.tree.map(np.ones_like, C["resolved"])} if has_disc else {})      # first order: C = 1 for z, as before
    things_fo = dict(things) | {"primary_opd": adam(fo_lr_opd, 0)} | ({"resolved": adam(fo_lr_disc, 0)} if has_disc else {})
    t0 = time.time()
    losses, history = optimise_new_resolved(opt_params, model_b, exps_b, things_fo, fo_epochs, use_c=C_fo, mapped=True)
    if len(losses) < fo_epochs:
        raise RuntimeError(f"first-order stage stopped after {len(losses)}/{fo_epochs} steps (non-finite)")
    x1 = history[-1]
    log(f"[refit] first order: {fo_epochs} epochs, loss {losses[0]:.1f} -> {losses[-1]:.1f} ({time.time() - t0:.0f} s)")
    t0 = time.time()
    x2, info = optimise_optimistix(x1, model_b, exps_b, use_c=C, max_steps=optx_max_steps, return_info=True,
                                   mapped=True, verbose=True)
    try:   # diagnostics must never discard the refit
        gs_start, gs_end = precond_grad_split(x1, C, model_b, exps_b), precond_grad_split(x2, C, model_b, exps_b)
    except Exception as e:
        print(f"[refit] gradient diagnostics failed: {e!r}")
        gs_start = gs_end = dict(norm=float("nan"), groups={}, frac_sq={}, top_leaves=[])
    info = dict(info, max_steps=optx_max_steps, grad_norm_u_start=gs_start["norm"], grad_norm_u_end=gs_end["norm"],
                grad_split_start=gs_start, grad_split_end=gs_end)
    log(f"[refit] L-BFGS: {info['result']} after {info['num_steps']}/{optx_max_steps} iterations, loss "
        f"{info['loss_start']:.1f} -> {info['loss_end']:.1f}, |sqrt(C) grad| {info['grad_norm_u_start']:.3g} -> "
        f"{info['grad_norm_u_end']:.3g} ({time.time() - t0:.0f} s)")
    print_split("L-BFGS start", gs_start)
    print_split("L-BFGS end", gs_end)
    if not all(bool(numpy.all(numpy.isfinite(numpy.asarray(l)))) for l in jax.tree.leaves(x2)):
        raise RuntimeError("L-BFGS returned non-finite parameters")
    return to_host(x2), numpy.append(losses, info["loss_end"]), info


# Stage A schedules (no disc, so no "resolved" / "resolved_level"). A1: imlup-adi's stage 1 (optics-first: no wavefront yet, the geometry,
# focus, cold mask, low orders and spectrum, delayed starts staggered); A2: stage E's parameter set (refit() above).
things_A = {k: v for k, v in things_gp.items() if k not in ("resolved", "resolved_level")}
g_a = 5e-2
things_start_A = {
    "spectrum": sgd(g_a*3, 30.), "primary_tilt": sgd(g_a*5., 15.), "cold_mask_tilt": sgd(g_a*1., 0), "cold_mask_opd": sgd(g_a*0.1, 40),
    "defocus1": sgd(g_a*1, 60), "defocus2": sgd(g_a*10, 60), "bias": sgd(g_a*3, 50), "cold_mask_shift": sgd(g_a*0.3, 70),
    "cold_mask_rot": sgd(g_a*0.1, 85), "primary_rot": sgd(g_a*0.03, 85), "primary_low": sgd(g_a*2., 150),
    "cold_mask_shear": sgd(g_a*0.1, 100), "cold_mask_distortion": sgd(g_a*0.1, 150), "primary_shear": sgd(g_a*1, 100),
    "occulter_radius": sgd(g_a*1., 120), "outer_radius": sgd(g_a*0.1, 120), "secondary_radius": sgd(g_a*0.1, 120),
    "spider_width": sgd(g_a*0.1, 120), "primary_spider": sgd(g_a*0.1, 120), "primary_secondary": sgd(g_a*0.1, 120),
    "primary_pad": sgd(g_a*0.1, 120), "primary_outer": sgd(g_a*0.1, 120), "secondary_shift": sgd(g_a*0.3, 120),
    "primary_distortion": sgd(g_a*0.1, 150), "fnumber": sgd(g_a*1., 200), "anisotropy": sgd(g_a*2., 200),
}


# %%
# ---------------------------------------------------------------- A0: exposures, models, initial values
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_F160W_{nwavels}.npy")[:, :npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

trend_basis = radial_trend_basis(resolved_wid, trend_kinds)
n_b = trend_basis.shape[1]


class GPDiscFit(TrendGPResolvedFit):
    """TrendGPResolvedFit (Matern DCT + radial trend) with imlup-adi's ResolvedFit keys."""
    PARAM_KEYS = TrendGPResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                  "primary_opd": "global", "cold_mask_opd": "exposure",
                                                  "defocus3": "exposure"}


class StarFit(SinglePointFit):
    """Stage A's star-only fit (no disc at all: cheaper than a disc at 1e-30, no disc-light gradient): the same per-parameter keys as
    GPDiscFit, so its fitted params drop straight into the GP model (the spectrum is keyed by target in both)."""
    PARAM_KEYS = SinglePointFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                              "primary_opd": "global", "cold_mask_opd": "exposure",
                                              "defocus3": "exposure"}


for name_, how_ in StarFit.PARAM_KEYS.items():
    if name_ in GPDiscFit.PARAM_KEYS:
        assert GPDiscFit.PARAM_KEYS[name_] == how_, f"StarFit and GPDiscFit key {name_} differently: {how_} vs {GPDiscFit.PARAM_KEYS[name_]}"

files = [ddir + f'{root}_m_clc_calf.fits' for root in roots]
disc_pitch = 0.0432*2
crop_origin = Cutout2D(numpy.zeros((256, 256)), CROP_CENTRE, wid).origin_original      # (x0, y0) of the crop on the detector
# the FITS header's star position in crop pixels (col, row): the NIC2-CORON aperture reference point (TARSIAFX/Y), the nominal hole /
# target position, ~1-2 px from the fitted star (printed below)
stars_header = numpy.array([[float(fits.getheader(f, ext=0)["TARSIAFX"]) - crop_origin[0],
                             float(fits.getheader(f, ext=0)["TARSIAFY"]) - crop_origin[1]] for f in files])
pixel_scale = lambda fnumber: float(dlu.rad2arcsec(optics.layers["prop1"].spec.d/(float(fnumber)*2.4))*oversample)   # arcsec / data px (MFT pixel x oversample)

optical_params = None
if optics_source == "calibrator":
    if not os.path.exists(calibrator_file):
        raise FileNotFoundError(f"{calibrator_file} missing (the calibrator-joint medians, notebooks/calibrator-pca.ipynb)")
    optical_params = numpy.load(calibrator_file, allow_pickle=True)[()]
    scale0 = pixel_scale(optical_params["fnumber"])
else:
    scale0 = pixel_scale(45.7)
star_offsets = (stars_header - (wid - 1)/2)*scale0      # arcsec (x, y) of the header star from the crop centre
print(f"header stars (col, row) {numpy.round(stars_header, 2).tolist()}; offsets from the crop centre {numpy.round(star_offsets/scale0, 2).tolist()} px; "
      f"pixel scale {scale0:.5f}\"; optics_source {optics_source}, level_source {level_source}, support_star {support_star}", flush=True)

# n_share: all four exposures share the target-keyed disc, so each carries 1/4 of its prior
exposures = with_n_share([
    exposure_from_file(f, GPDiscFit(spectrum_basis, "F160W", wid=resolved_wid, trend_sigma=1., kinds=trend_kinds),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for f in files
])
check_n_share(exposures)
key = exposures[0].fit.get_key(exposures[0], "resolved")
exposures_fp = lambda exps: fingerprint(repr([e.key for e in exps]), 0.3, wid, *[numpy.nan_to_num(numpy.asarray(a_, dtype=float), nan=-1.) for e in exps
                                                                              for a_ in (e.data, e.err)], *[numpy.asarray(e.bad) for e in exps])
data_fp = exposures_fp(exposures)    # data, errors, bad / hole mask (hole_margin 0.3) and crop of the exposures: part of every cache key below


def initial_params(exps, source, optical_params=None, star_offsets=None, tilt_init="header"):
    """The starting optics + star values (no disc), keyed by exps[0].fit's PARAM_KEYS.
    source "calibrator" (gmaur-wpca.py): data/calibrator_optical_params.npy (calibrator-joint medians, F160W, global keys {"global": v}
    for the shared parameters) for everything it has; cold_mask_shift / cold_mask_opd / defocus1 / defocus2 per exposure start at its
    medians; defocus3, primary_opd (zero: no calibrator wavefront for this PSF state), primary_low, bias are zero; the spectrum starts as
    unit flux in its first basis function (rescaled by the caller); tilts: tilt_init "header": cold_mask_tilt = hole_tilt(exp) (the
    era's approximate hole offset) and primary_tilt = (header star offset from the crop centre) - hole_tilt, so that their SUM (the image
    position) is the header star position (the split is arbitrary at ~0.4 px: primary_tilt acts upstream of the occulter, cold_mask_tilt
    downstream, so the split sets the star's offset relative to the occulter, i.e. its attenuation; stage A fits it); "adi": imlup-adi's hand-set primary_tilt.
    source "l1tsv": imlup-adi's hand-set initial values (the template of imlup-gp-radial.py), to be overwritten by its fitted params."""
    names = ["spectrum", "primary_opd", "primary_low", "primary_tilt", "cold_mask_tilt", "bias", "defocus3"]
    params = {k: {} for k in names}
    if source == "calibrator":
        per_exposure_init = [k for k in ["cold_mask_shift", "cold_mask_opd", "defocus1", "defocus2", "defocus3"] if k in optical_params]
        params |= {k: {} for k in per_exposure_init}
    else:
        params |= {k: {} for k in ("cold_mask_opd", "cold_mask_shift", "cold_mask_rot", "cold_mask_shear", "cold_mask_scale",
                                   "cold_mask_distortion", "defocus1", "defocus2", "primary_rot", "primary_shear")}
        params |= {
            "occulter_radius": 0.8, "occulter_ellipticity": np.zeros(2), "fnumber": 45.7, "anisotropy": 1. + 1e-5,
            "outer_radius": 1.2*0.9768, "secondary_radius": 0.365*1.2, "spider_width": 0.072*1.2, "primary_spider": 0.0256,
            "primary_secondary": 0.330*1.2, "primary_outer": 1.2, "primary_pad": 0.065*1.2, "secondary_shift": np.zeros(2),
            "primary_distortion": np.zeros((2, 7)), "distortion": np.zeros((2, 7)),
        }
    for i, exp in enumerate(exps):
        k = lambda name: exp.fit.get_key(exp, name)
        params["spectrum"][k("spectrum")] = (np.zeros(npoly).at[0].set(1.) if source == "calibrator"
                                             else np.zeros(npoly).at[0].set((np.nansum(exp.data)/nwavels)*100))
        hole = numpy.asarray(hole_tilt(exp), dtype=float)
        if source == "calibrator" and tilt_init == "header":
            params["cold_mask_tilt"][k("cold_mask_tilt")] = np.asarray(hole)
            params["primary_tilt"][k("primary_tilt")] = np.asarray(star_offsets[i] - hole)
        else:
            params["cold_mask_tilt"][k("cold_mask_tilt")] = np.asarray(hole)
            params["primary_tilt"][k("primary_tilt")] = np.array([-0.75, -0.05])*0.075
        params["primary_opd"][k("primary_opd")] = np.zeros((n_modes, n_modes))
        params["primary_low"][k("primary_low")] = np.zeros(n_zernikes)
        params["bias"][k("bias")] = 0.
        params["defocus3"][k("defocus3")] = 0.
        if source == "calibrator":
            for name in per_exposure_init:
                if name != "defocus3":
                    params[name][k(name)] = np.asarray(optical_params[name]["global"])
        else:
            params["cold_mask_opd"][k("cold_mask_opd")] = np.zeros(16).at[0].set(120.)
            params["cold_mask_shift"][k("cold_mask_shift")] = np.array([13., 10.])
            params["cold_mask_rot"][k("cold_mask_rot")] = 3.2
            params["primary_rot"][k("primary_rot")] = -0.6
            params["cold_mask_scale"][k("cold_mask_scale")] = np.asarray([1., 1.])
            params["cold_mask_shear"][k("cold_mask_shear")] = np.asarray([0.06, -0.06])
            params["primary_shear"][k("primary_shear")] = np.asarray([0., 0.])
            for name in ("defocus1", "defocus2"):
                params[name][k(name)] = 0.
            params["cold_mask_distortion"][k("cold_mask_distortion")] = np.zeros((2, 7))
    if source == "calibrator":
        params |= {k_: v for k_, v in optical_params.items() if k_ not in per_exposure_init}
    return params


def check_param_keys(exps, params):
    """Every dict-valued param must hold the key the fit looks it up by (a KeyError deep inside a jit is much harder to read)."""
    for exp in exps:
        for name, v in params.items():
            if isinstance(v, dict) and name in exp.fit.PARAM_KEYS:
                assert exp.fit.get_key(exp, name) in v, f"params[{name!r}] has no key {exp.fit.get_key(exp, name)!r}: {list(v)}"


core_fp = hashlib.sha256(b"".join(open(f"../{f_}", "rb").read() for f_ in ("models.py", "fitting.py", "gp.py", "precond.py"))).hexdigest()[:12]
render = eqx.filter_jit(lambda m, e: e.fit(m, e))
render_d = eqx.filter_jit(lambda m, e, d: _with_disc(e.fit, d)(m, e))     # the image of a given disc d (wid x wid), as the LinearResponse sees it


def scale_star_flux(exps, params):
    """params with the spectrum rescaled so that the star model's total flux matches the data's over the unmasked pixels of all
    exposures (gmaur-wpca.py; the initial disc is zero, so there is nothing to subtract; one global spectrum, so one factor)."""
    model = set_array(NICMOSModel(exps, params, optics, detector))
    img = [render(model, e) for e in exps]
    c = (sum(np.nansum(np.where(e.bad, 0., e.data)) for e in exps)/sum(np.nansum(np.where(e.bad, 0., s)) for e, s in zip(exps, img)))
    return params | {"spectrum": jax.tree.map(lambda s: s*c, params["spectrum"])}, float(c)


# %%
# ---------------------------------------------------------------- A: optics-only fit, disc OFF, calibrator start (stage A)
stage_a_info = None
if optics_source == "calibrator":
    # the stage-A bad mask: the annulus around the header star on top of the usual bad / hole pixels
    exposures_A = [exposure_from_file(f, StarFit(spectrum_basis, "F160W"), crop=wid, hole_margin=0.3,
                                      extra_bad=stage_a_mask((wid, wid), sh, scale0, *stage_a_mask_radii))
                   for f, sh in zip(files, stars_header)]
    n_keep = [int((~numpy.asarray(e.bad)).sum()) for e in exposures_A]
    n_all = [int((~numpy.asarray(e.bad)).sum()) for e in exposures]
    print(f"stage A: annulus {stage_a_mask_radii[0]:.2f}-{stage_a_mask_radii[1]:.2f}\" masked; good pixels per exposure {n_keep} "
          f"of {n_all} (the disc-free fit uses {100*sum(n_keep)/sum(n_all):.0f} %)", flush=True)
    assert all(a_ < b_ for a_, b_ in zip(n_keep, n_all)) and min(n_keep) > 500, "stage-A mask empty or too aggressive"
    fpA = fingerprint(hashlib.sha256(open(calibrator_file, "rb").read()).hexdigest()[:12], repr(files), repr(stage_a_mask_radii),
                      tilt_init, a1_epochs, fo_epochs, fo_lr_opd, optx_max_steps, g_a, g_sgd, repr(sorted(things_start_A)), repr(sorted(things_A)),
                      repr(sorted(precond_kwargs.items())), core_fp, wid, n_modes, n_zernikes, nwavels, npoly, oversample, pad, exposures_fp(exposures_A))
    pA1, pA, iA = (f"{out}/stageA1-{fpA}-params.npy", f"{out}/stageA-{fpA}-params.npy", f"{out}/stageA-{fpA}-info.json")
    with timed("A: optics-only fit (disc off)"):
        if os.path.exists(pA) and os.path.exists(iA):
            params_opt = numpy.load(pA, allow_pickle=True).item()
            stage_a_info = json.load(open(iA))
            print(f"stage A resumed from {pA}")
        else:
            p0 = initial_params(exposures_A, "calibrator", optical_params, star_offsets, tilt_init)
            check_param_keys(exposures_A, p0)
            p0, c_star = scale_star_flux(exposures_A, p0)
            print(f"initial star flux scaling: x{c_star:.3g}", flush=True)
            if os.path.exists(pA1):
                p1 = numpy.load(pA1, allow_pickle=True).item()
            else:   # A1: optics-first stage (imlup-adi's stage 1, shorter)
                model0 = set_array(NICMOSModel(exposures_A, p0, optics, detector))
                opt0 = set_array({k_: p0[k_] for k_ in things_start_A})
                losses1, hist1 = optimise_new(opt0, model0, exposures_A, things_start_A, a1_epochs, precond_method="gn_probe",
                                              precond_kwargs=precond_kwargs, mapped=True)
                if not (numpy.all(numpy.isfinite(losses1)) and losses1[-1] <= losses1[0]):
                    raise RuntimeError(f"stage A1: loss {losses1[0]:.1f} -> {losses1[-1]:.1f} (non-finite or worse than the start)")
                p1 = to_host(dict(p0) | dict(hist1[-1]))
                save_npy(pA1, p1)
                save_npy(f"{out}/stageA1-{fpA}-losses.npy", losses1)
                fig = plt.figure(figsize=(10, 10)); plt.plot(losses1); fig.savefig(f"{out}/stageA1-losses.png"); plt.close(fig)
                print(f"A1: loss {losses1[0]:.1f} -> {losses1[-1]:.1f} ({a1_epochs} epochs)", flush=True)
            # A2: the hybrid refit with the wavefront (primary_opd), same parameter set as stage E, no disc
            model1 = set_array(NICMOSModel(exposures_A, p1, optics, detector))
            opt1 = set_array({k_: p1[k_] for k_ in things_A})
            x2, losses2, optx2 = refit(opt1, model1, exposures_A, things_A)
            if not (numpy.all(numpy.isfinite(losses2)) and losses2[-1] <= losses2[0]):
                raise RuntimeError(f"stage A2: loss {losses2[0]:.1f} -> {losses2[-1]:.1f} (non-finite or worse than the start)")
            params_opt = dict(p1) | x2
            fig = plt.figure(figsize=(10, 10)); plt.plot(losses2[:-1]); plt.plot([fo_epochs], losses2[-1:], "o"); fig.savefig(f"{out}/stageA2-losses.png"); plt.close(fig)
            ok_str = str(optx.RESULTS.successful)
            stage_a_info = dict(initial_flux_scale=c_star, loss_A2_start=float(losses2[0]), loss_A2_end=float(losses2[-1]),
                                lbfgs=dict(optx2), successful=bool(str(optx2["result"]) == ok_str and optx2["num_steps"] < optx_max_steps),
                                mask_radii_arcsec=list(stage_a_mask_radii), n_good_per_exposure=n_keep, n_good_all=n_all, tilt_init=tilt_init,
                                n_epochs_A1=a1_epochs, fingerprint=fpA)
            save_npy(pA, params_opt)
            with open(iA, "w") as f_:
                json.dump(stage_a_info, f_, indent=2, default=str)
    if not stage_a_info.get("successful", False):
        print(f"WARNING: stage A's L-BFGS did not report success ({stage_a_info['lbfgs'].get('result')}, {stage_a_info['lbfgs'].get('num_steps')} its): "
              "the starting optics of the GP stages are under-converged (stage E refits them)")
    # ---- stage-A diagnostics: where does the star-only model leave light? (smooth positive excess in the masked annulus = the disc)
    with timed("A: diagnostics"):
        scale_A = pixel_scale(params_opt["fnumber"])
        model_A = set_array(NICMOSModel(exposures_A, params_opt, optics, detector))
        img_A = [numpy.asarray(render(model_A, e)) for e in exposures_A]
        rr, cc = numpy.indices((wid, wid))
        diag = dict(exposures={}, bins_arcsec=[0.6, 0.9, 1.2, 1.5, 2.0, 2.4, 3.0, 4.5])
        rad = []
        for e_A, e_full, im_ in zip(exposures_A, exposures, img_A):
            sc_ = star_pixel(params_opt, e_full, wid, scale_A)
            r_ = numpy.hypot(cc - sc_[0], rr - sc_[1])*scale_A
            y_, s_ = numpy.asarray(e_full.data), numpy.asarray(e_full.err)
            g_full = ~numpy.asarray(e_full.bad) & numpy.isfinite(y_) & numpy.isfinite(s_)
            g_kept = ~numpy.asarray(e_A.bad) & numpy.isfinite(y_) & numpy.isfinite(s_)
            g_mask = g_full & ~g_kept
            chi = lambda m_: float(numpy.sum(((y_ - im_)[m_]/s_[m_])**2)/max(m_.sum(), 1))
            prof = []
            for lo_, hi_ in zip(diag["bins_arcsec"][:-1], diag["bins_arcsec"][1:]):
                m_ = g_full & (r_ >= lo_) & (r_ < hi_)
                if m_.any():
                    prof.append(dict(r=[lo_, hi_], n=int(m_.sum()), data_median=float(numpy.median(y_[m_])), residual_median=float(numpy.median((y_ - im_)[m_])),
                                     residual_over_data=float(numpy.median((y_ - im_)[m_])/numpy.median(y_[m_]))))
            diag["exposures"][e_A.key] = dict(star_px=sc_.tolist(), star_header_px=stars_header[len(rad)].tolist(), chi2_per_pixel_kept=chi(g_kept),
                                              chi2_per_pixel_masked=chi(g_mask), profile=prof)
            rad.append((y_, im_, g_kept, g_mask, r_))
        with open(f"{out}/stageA-diagnostics.json", "w") as f_:
            json.dump(diag, f_, indent=2)
        numpy.savez(f"{out}/stageA-residual.npz", data=numpy.stack([r_[0] for r_ in rad]), model=numpy.stack([r_[1] for r_ in rad]),
                    kept=numpy.stack([r_[2] for r_ in rad]), masked=numpy.stack([r_[3] for r_ in rad]), r=numpy.stack([r_[4] for r_ in rad]))
        for k_, d_ in diag["exposures"].items():
            print(f"[stage A] {k_}: star at {numpy.round(d_['star_px'], 2).tolist()} px (header {numpy.round(d_['star_header_px'], 2).tolist()}); chi2/pixel kept "
                  f"{d_['chi2_per_pixel_kept']:.2f}, masked annulus {d_['chi2_per_pixel_masked']:.2f}; (data - star)/data by radius: "
                  + ", ".join(f"{p_['r'][0]:.1f}-{p_['r'][1]:.1f}\" {p_['residual_over_data']:+.2f}" for p_ in d_["profile"]), flush=True)
        y0, m0, k0, q0, r0 = rad[0]
        fig, axs = plt.subplots(1, 3, figsize=(33, 10))
        axs[0].imshow(numpy.where(k0 | q0, y0, numpy.nan)**0.25); axs[0].set_title("data^0.25")
        axs[1].imshow(numpy.where(k0 | q0, y0 - m0, numpy.nan)); axs[1].contour(q0, [0.5], colors="c"); axs[1].set_title("data - star model (cyan: stage-A mask)")
        for k_, (y_, im_, kk, qq, r_) in enumerate(rad):
            m_ = kk | qq
            axs[2].plot(r_[m_], ((y_ - im_)/numpy.where(y_ == 0, numpy.nan, y_))[m_], ",", color=f"C{k_}", alpha=0.3)
        axs[2].axhline(0, color="k"); axs[2].axvspan(*stage_a_mask_radii, color="c", alpha=0.15); axs[2].set(xlabel="r (arcsec)", ylabel="(data - star)/data", ylim=(-1, 1))
        fig.savefig(f"{out}/stageA-residual.png"); plt.close(fig)
        try:
            plot_comparison_detailed(model_A, ModelParams(set_array(params_opt)), exposures_A, quadrature=False, wf_size=wf_wid, percentile=99, save=f"{out}/stageA-comparison")
        except Exception as e_:
            print(f"[stage A] comparison plot failed: {e_!r}")

# ---- the L1/TSV fit: only read with an l1tsv flag (optics and / or level)
d_ref = None
if optics_source == "l1tsv" or level_source == "l1tsv":
    l1 = {}
    for f_ in ("intermediate-params.npy", "params_amp.npy"):
        path_ = f"{ref_dir}/{f_}"
        if not os.path.exists(path_):
            raise FileNotFoundError(f"{path_} missing: an l1tsv flag needs imlup-adi.py's fit (run it first)")
        l1 |= numpy.load(path_, allow_pickle=True).item()
    d_ref = 10**np.asarray(l1.pop("resolved")[key])           # the L1/TSV disc
    if optics_source == "l1tsv":
        params_opt = initial_params(exposures, "l1tsv") | l1
check_param_keys(exposures, params_opt)


# %%
# ---------------------------------------------------------------- S: disc support mask (gp.py support_mask), built ONCE from the optics above
# Geometry as models.py: the disc pixel centre ((col - 29.5) pitch, (row - 29.5) pitch) is rolled by -ORIENTAT about the star, and a point
# (x', y') arcsec from the star lands at crop pixel (star_col + x'/scale, star_row + y'/scale). The star position is the MODEL's
# (star_pixel: optical axis + primary_tilt + cold_mask_tilt, from the stage-A params, or the L1/TSV params with optics_source "l1tsv"),
# or the header's with support_star = "header". Fixed for the run: the outer-loop refits do not update it (print below: how far the final
# star ends up from it).
support_shape = "coverage"   # "coverage": keep covered pixels; "circle": the largest star-centred circle of them
support_cover = "any"        # "any": keep a pixel if at least one exposure covers it; "all": only if every exposure does
support_footprint = True     # cover = the whole rotated pixel footprint (4 corners) inside the crop, not just its centre
support_margin_px = 0.       # data px a pixel must be inside the crop edge (evidence cost grows quickly with it)
support_radius = None        # arcsec: override the circle's radius (shape "circle" only)
support_taper = 0.           # disc px cos^2 edge roll-off (shape "circle" only); 0 = hard edge
support_scale = pixel_scale(params_opt["fnumber"])    # arcsec / data px
stars_fitted = numpy.array([star_pixel(params_opt, e, wid, support_scale) for e in exposures])
support_stars = stars_fitted if support_star == "fitted" else stars_header
support_orients = numpy.array([float(e.orient) for e in exposures])
sm_kw = dict(margin=support_margin_px, shape=support_shape, cover=support_cover, footprint=support_footprint,
             radius=support_radius, taper=support_taper)
support, support_info = support_mask(resolved_wid, disc_pitch, support_orients, support_stars, wid, support_scale, **sm_kw)
n_header = support_mask(resolved_wid, disc_pitch, support_orients, stars_header, wid, support_scale, **sm_kw)[1]["n_support"]
circle_txt = (f"circle radius {support_info['radius_arcsec']:.3f}\" = {support_info['radius_px']:.1f} px; "
              if support_shape == "circle" else "")
print(f"support mask ({support_shape}, cover {support_cover}, footprint {support_footprint}, margin {support_margin_px:g} px, star {support_star}): "
      f"{support_info['n_support']} / {resolved_wid**2} disc pixels kept ({n_header} with the header star); pixel scale {support_scale:.5f}\"; "
      f"per-exposure coverage {numpy.round(support_info['coverage_fraction'], 3)}; nearest uncovered pixel "
      f"{support_info['r_coverage_arcsec']:.3f}\"; {circle_txt}fitted stars {numpy.round(stars_fitted, 2).tolist()} vs header "
      f"{numpy.round(stars_header, 2).tolist()}; orients {support_orients.tolist()}", flush=True)
numpy.save(f"{out}/support.npy", support)
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
axs[0].imshow(support); axs[0].set_title(f"support M ({support_shape}, {support_cover}), n = {support_info['n_support']}")
for k_ in range(len(exposures)):
    axs[1].contour(support_info["coverage"][k_], [0.5], colors=f"C{k_}")
axs[1].set_title("coverage of each exposure (contours)")
axs[2].imshow(support_info["coverage"].sum(0)); axs[2].set_title("number of exposures covering each disc pixel")
fig.savefig(f"{out}/support.png")
plt.close(fig)
exposures = with_support(exposures, support)      # every prior the fit builds (surrogate_prior) and the full model carry M
check_n_share(exposures)
# the prior-independent results below depend on the mask through the priors, so every fingerprint (code_fp) includes it
code_fp = hashlib.sha256(open("../gp.py", "rb").read() + numpy.asarray(support, dtype=float).tobytes()).hexdigest()[:12]

# %%
# ---------------------------------------------------------------- L: the level prior (resolved_level ~ N(level_mu, level_sd^2)), no reference disc
# `level` is the mean of log10 d over the support in model disc units (module docstring of gp.py). level_source "data": estimate_level
# from (data - star model) over the good pixels of the full exposures, with the unit-disc response f measured from the full forward
# model at the stage-A optics; "l1tsv": reference_level of the L1/TSV disc over the support (imlup-gp-radial.py).
params0 = dict(params_opt) | {"resolved": {key: np.zeros(resolved_wid**2 + n_b)}, "resolved_level": {key: np.asarray(0.)}}
model_single = set_array(NICMOSModel(exposures, params0, optics, detector))
level_report = {}
with timed("L: level estimate"):
    img0 = [numpy.asarray(render_d(model_single, e, np.zeros((resolved_wid, resolved_wid)))) for e in exposures]
    img1 = [numpy.asarray(render_d(model_single, e, np.asarray(support, dtype=float))) for e in exposures]
    f_unit = [a_ - b_ for a_, b_ in zip(img1, img0)]
    good_full = [~numpy.asarray(e.bad) & numpy.isfinite(numpy.asarray(e.err)) & numpy.isfinite(numpy.asarray(e.data)) for e in exposures]
    est = estimate_level([numpy.asarray(e.data) for e in exposures], [numpy.asarray(e.err) for e in exposures], good_full, img0, f_unit,
                         level_smooth_px, level_k_floor, level_f_min)
    f_analytic = support_scale**2/disc_pitch**2
    level_report |= dict(data_estimate=est["level"], per_exposure=est["level_per_exposure"], n_pixels=est["n_pixels"], frac_floored=est["frac_floored"],
                         f_median_measured=est["f_median"], f_analytic_no_throughput=f_analytic, smooth_px=level_smooth_px, k_floor=level_k_floor,
                         f_min_frac=level_f_min)
    print(f"level estimate from the data: {est['level']:.3f} dex (per exposure {numpy.round(est['level_per_exposure'], 3).tolist()}; {est['n_pixels']} pixels, "
          f"{100*est['frac_floored']:.0f} % at the noise floor); unit-disc response f median {numpy.round(est['f_median'], 3).tolist()} DN/s per data pixel "
          f"per unit disc value vs analytic (scale/pitch)^2 = {f_analytic:.3f} x throughput", flush=True)
    if optics_source == "calibrator":
        # consistency of the two star models: stage A's star-only fit and the GP fit's disc = 0 image must agree (same optics, same source)
        dif = max(float(numpy.max(numpy.abs(a_ - b_))/numpy.max(numpy.abs(b_))) for a_, b_ in zip(img0, img_A))
        level_report["star_only_vs_full_model_max_rel_diff"] = dif
        print(f"star-only (stage A) vs GP-model (d = 0) image: max relative difference {dif:.2e}")
        if dif > 1e-6:
            print("WARNING: the star-only and the disc-model star images differ: stage A's flux / optics do not transfer one-to-one "
                  "(the Scene's point source differs from SinglePointFit's?); stage E refits them but check models.py")
    if d_ref is not None:
        level_report["l1tsv_reference_level"] = reference_level(d_ref, None, support)
        print(f"(for reference: the L1/TSV disc's support-mean level {level_report['l1tsv_reference_level']:.3f} dex)")
    if level_mu_override is not None:
        level_mu0, level_how = float(level_mu_override), "override"
    elif level_source == "data":
        level_mu0, level_how = est["level"], "data"
    else:
        level_mu0, level_how = level_report["l1tsv_reference_level"], "l1tsv"
    level_report |= dict(level_mu=level_mu0, level_sd=level_sd, how=level_how)
mean_t = None if d_ref is None else log10_template(d_ref, template_floor)
level_mu = {t: level_mu0 for t in templates}        # flat mean only (templates = [False])
print(f"level prior: mu = {level_mu[False]:.3f} dex ({level_how}), sd = {level_sd:g} dex", flush=True)

params = params0 | {"resolved_level": {key: np.asarray(level_mu[False])}}
ref_params = params          # the starting params of the GP stages: stage-A (or L1/TSV) optics, zero latents, the level prior's centre
check_gp_setup(ref_params)
check_param_keys(exposures, ref_params)
model_single = set_array(NICMOSModel(exposures, ref_params, optics, detector))
n = resolved_wid**2
mean_of = lambda template: mean_t if template else None
lp_of = lambda template: (level_mu[template], level_sd)
check_gp_setup(ref_params, things_gp)

# %%
# ---------------------------------------------------------------- B: linear response at the stage-A optics (a NEW one in this run's directory)
# The response depends only on the optics / star / bias, not on the prior or the support; here those are stage A's (or the L1/TSV's with
# optics_source "l1tsv", the one case where imlup-gp's shared response IS valid and is reused to reproduce imlup-gp-radial.py).
params_fp = fingerprint(data_fp, core_fp, *[numpy.asarray(l_) for l_ in jax.tree.leaves({k_: v_ for k_, v_ in ref_params.items() if k_ not in ("resolved", "resolved_level")})])
own_resp = f"{out}/response-B-{params_fp}.npz"
with timed("B: response"):
    reuse_shared = (optics_source == "l1tsv")
    rpath0 = next((p_ for p_ in ([f"{shared_dir}/response.npz"] if reuse_shared else []) + [own_resp] if os.path.exists(p_)), own_resp)
    assert optics_source == "l1tsv" or os.path.dirname(rpath0) != shared_dir, "the shared (L1/TSV-optics) response must not be used here"
    if os.path.exists(rpath0):
        resp = LinearResponse.load(rpath0)
    else:
        resp = build_linear_response(model_single, exposures, batch_size=response_batch)
        resp.save(rpath0)
    resp_fp = response_fingerprint(resp)
print(f"response ready ({rpath0}, fingerprint {resp_fp})")

with timed("B: verify"):
    # the fingerprint names the response's content, so a verification of the same response is reusable
    vpath = next((p_ for p_ in ([f"{shared_dir}/verify-{resp_fp}.npz"] if reuse_shared else []) + [f"{out}/verify-{resp_fp}.npz"] if os.path.exists(p_)),
                 f"{out}/verify-{resp_fp}.npz")
    if not os.path.exists(vpath):
        # no reference disc: zero, a uniform disc at the level prior's centre, and random draws from a GP prior (sigma 0.3 and 0.6 dex,
        # lengths 5 and 2.5 px, level at its centre) -- realistic amplitudes, so the NLL changes are neither tiny nor astronomical (+ the L1/TSV
        # disc when it was loaded). The full model's NLL must match the surrogate's. Compared as changes from the no-disc NLL.
        test_discs = [np.zeros(n), (np.asarray(support, dtype=float)*10**level_mu[False]).ravel()]
        for i_, (s_, l_) in enumerate(zip(verify_draw_sigmas, (5., 2.5))):
            pr_ = gp_prior(resolved_wid, s_, l_, 2., None, (level_mu[False], level_sd), support=support)
            th_ = jr.normal(jr.PRNGKey(i_), (pr_.n_latent + 1,)).at[-1].set(0.)
            test_discs.append(pr_.distribution(th_).ravel())
        if d_ref is not None:
            test_discs.append(np.asarray(d_ref).ravel())
        ds = np.stack(test_discs)
        full, sur = (numpy.asarray(x_) for x_ in verify_linear_response(resp, model_single, exposures, ds))
        dfull, dsur = full[:, 1:] - full[:, :1], sur[:, 1:] - sur[:, :1]
        # saved BEFORE the threshold test, so a (false) failure is diagnosable from the file
        _save_npz(vpath, full=full, surrogate=sur, abs_diff=numpy.abs(full - sur),
                  rel_change=numpy.abs(dfull - dsur)/numpy.maximum(numpy.abs(dfull), 1.))
    v = numpy.load(vpath)
    verify_max_abs = float(v["abs_diff"].max())      # captured now: `v` is a short name, do not rely on it later
    print(f"linear response vs full model: max |NLL difference| {v['abs_diff'].max():.2e}, max relative difference "
          f"of NLL changes {v['rel_change'].max():.2e}")
    # the absolute tolerance scales with the NLL magnitude (float64 roundoff of sums of ~1e6-1e9 terms), floor 1e-3
    abs_tol = max(1e-3, 1e-9*float(numpy.abs(v["full"]).max()))
    print(f"verify tolerances: abs {abs_tol:.2e} (NLL magnitude {float(numpy.abs(v['full']).max()):.2e}), rel 1e-6; per test disc max abs diff "
          f"{numpy.round(v['abs_diff'].max(axis=0), 12).tolist()}")
    if v["rel_change"].max() > 1e-6 or v["abs_diff"].max() > abs_tol:
        raise RuntimeError("the surrogate does not reproduce the full model: is the image linear in the disc at fixed optics? "
                           f"max abs {v['abs_diff'].max():.3e} (tol {abs_tol:.2e}), max rel {v['rel_change'].max():.3e}; per-disc results in {vpath} "
                           "(it is kept for diagnosis; delete it to rerun the check)")


# %%
# ---------------------------------------------------------------- C: first MAP, gradient check, noise scales
prior0 = gp_prior(resolved_wid, 1., 1.6, 2., None, lp_of(False), support=support)   # single kernel: the noise scales are the DCT run's

with timed("C: noise"):
    npath = f"{out}/noise-{fingerprint(resp_fp, level_mu[False], level_sd, n_iter, code_fp)}.npz"
    if not os.path.exists(npath):
        theta1, hist1 = gp_map(resp, prior0, n_iter=n_iter)
        ev1 = to_host(gp_log_evidence(resp, prior0, theta1))

        # The full model's gradient in z at the surrogate MAP (optics fixed) should vanish, relative to its value at
        # z = 0; and its loss should equal the surrogate's (including the prior)
        # trend_sigma = 0 (and b = 0) is exactly the Matern prior0: this also checks the trend fit's nesting
        exps0 = set_hyper(exposures, sigma=1., length=1.6, order=2., trend_sigma=0., level_prior=lp_of(False))
        batch0 = make_batch(exps0, model_single)

        def full_grad(z, level):
            sub = set_array({"resolved": {key: z}, "resolved_level": {key: level}})
            return _loss_grad(sub, jax.tree.map(np.ones_like, sub), batch0, model_single)

        loss_0, g_0 = full_grad(np.zeros(resolved_wid**2 + n_b), np.asarray(level_mu[False]))
        z1, lev1 = prior0.params(theta1)
        loss_1, g_1 = full_grad(np.concatenate([np.ravel(z1), np.zeros(n_b)]), lev1)
        g_ratio = float(np.linalg.norm(g_1["resolved"][key])/np.linalg.norm(g_0["resolved"][key]))
        loss_sur = float(ev1["nll"].sum() + ev1["prior"])
        print(f"full-model |grad z| at the surrogate MAP / at z = 0: {g_ratio:.2e} (Newton decrement "
              f"{float(ev1['newton_decrement']):.1e}); full loss {float(loss_1):.6f} vs surrogate {loss_sur:.6f}")

        s, theta_s, ev_s = gp_error_scale(resp, prior0, theta1, verbose=True)
        _save_npz(npath, s=s, theta1=theta1, theta_s=theta_s, g_ratio=g_ratio, loss_full=loss_1,
                  loss_surrogate=loss_sur, chi2_1=ev1["chi2"], n_good=resp.n_good, gamma_e=ev_s["gamma_e"])
    noise = numpy.load(npath)
    s_noise = noise["s"]
    print("noise scales per exposure:", dict(zip(resp.keys, numpy.round(s_noise, 4))))
    if noise["g_ratio"] > 1e-3:
        print(f"WARNING: the full-model gradient at the surrogate MAP is not small ({float(noise['g_ratio']):.1e} of "
              "its value at z = 0): the MAP is unconverged or the surrogate is not the full model")
    if numpy.any(s_noise > 3):
        print("WARNING: s >> 1: the model does not fit those exposures; rescaling the errors hides a model failure")
beta = 1/np.asarray(s_noise)**2


# %%
# ---------------------------------------------------------------- D: evidence grids (single Matern vs Matern + radial trend)
# Matched hyperpriors: the trend model has the SAME (sigma, length) axes, grids and bounds as the Matern, and its extra
# hyperparameter trend_sigma is fixed (not an axis): the trend adds no hyperprior volume, only its n_b latents' Occam cost
# (~ n_b ln trend_sigma + data-dependent terms), so the log_evidence_bounds difference is a like-for-like one. The
# sensitivity to the fixed trend_sigma is reported at the best point (trend_sigma_sensitivity).
base_fp = fingerprint(resp_fp, beta, level_mu[False], level_sd, n_iter, code_fp)
singles, singles_post, trends, trends_post = {}, {}, {}, {}
sg = lambda order: (lambda sigma, length: gp_prior(resolved_wid, sigma, length, order, None, lp_of(False), support=support))
tf = lambda order, s_h=trend_sigma: trend_prior_factory(resolved_wid, order, s_h, None, lp_of(False), basis=trend_basis, support=support)
theta_s_trend = embed_trend_theta(noise["theta_s"], n_b)
with timed("D: single-kernel grids"):
    for order in orders:
        g = gp_evidence_grid_nd(resp, {"sigma": sigmas0, "length": lengths0}, sg(order), beta=beta, outdir=f"{out}/grid",
                                tag=f"o{order:g}_t0_" + base_fp, abbrev=ABBREV, bounds=hyper_bounds, n_iter=n_iter,
                                theta0=noise["theta_s"], max_growth=6, recheck_top=recheck_top)
        singles[order], singles_post[order] = g, gp_hyper_posterior(g)    # on the grid's own bounds = hyper_bounds
        print(f"[single o{order:g}] best converged logZ {converged_max(g):.2f} at {gp_best(g)}; log_evidence_bounds "
              f"{singles_post[order]['log_evidence_bounds']:.2f}; edge {singles_post[order]['edge_weight']:.3f}")

with timed("D: trend grids"):
    for order in orders:
        g = gp_evidence_grid_nd(resp, {"sigma": sigmas0, "length": lengths0}, tf(order), beta=beta, outdir=f"{out}/grid",
                                tag=f"trend_o{order:g}_" + base_fp, abbrev=ABBREV, bounds=hyper_bounds, n_iter=n_iter,
                                theta0=theta_s_trend, max_growth=6, recheck_top=recheck_top)
        trends[order], trends_post[order] = g, gp_hyper_posterior(g)
        print(f"[trend  o{order:g}] best converged logZ {converged_max(g):.2f} at {gp_best(g)}; log_evidence_bounds "
              f"{trends_post[order]['log_evidence_bounds']:.2f}; edge {trends_post[order]['edge_weight']:.3f}")
    _save_npz(f"{out}/grid-trend.npz", **{f"o{o:g}_{k}": v for o, g in trends.items() for k, v in
                                          (("logZ", g["logZ"]), ("converged", g["converged"]), ("best", numpy.array(g["best"])),
                                           ("sigma", g["axes"]["sigma"]), ("length", g["axes"]["length"]))})

best_single = max(singles_post, key=lambda o: singles_post[o]["log_evidence_bounds"])
best_trend = max(trends_post, key=lambda o: trends_post[o]["log_evidence_bounds"])
g1, gt = singles[best_single], trends[best_trend]
b1, bt = gp_best(g1), gp_best(gt)
th_1, th_t = np.asarray(g1["thetas"][g1["best"]]), np.asarray(gt["thetas"][gt["best"]])
prior_1 = sg(best_single)(**b1)
prior_t = tf(best_trend)(**bt)

# nesting check on the data's prior: trend_sigma = 0 IS the Matern prior (amplitudes and the surrogate's evidence)
p_nest = gp_trend_prior(resolved_wid, b1["sigma"], b1["length"], best_single, 0., None, lp_of(False), support=support)
nest = float(np.abs(p_nest.amp - prior_1.amp).max())
ev_n1 = gp_log_evidence(resp, prior_1, th_1, beta)
ev_n0 = gp_log_evidence(resp, p_nest, embed_trend_theta(th_1, n_b), beta)
nest_logZ = float(np.abs(ev_n1["logZ"] - ev_n0["logZ"]))
print(f"nesting: max |amp difference| {nest:.1e}; |logZ(trend_sigma = 0) - logZ(Matern)| at the Matern best {nest_logZ:.1e}")
assert nest < 1e-12
if nest_logZ > 1e-3:   # eigh roundoff differs between the 3601- and 3603-dim problems (GPU paths); warn, don't abort
    warnings.warn(f"nesting logZ difference {nest_logZ:.2e} > 1e-3 nats: check the trend prior at trend_sigma = 0")

dlogE_coarse = float(trends_post[best_trend]["log_evidence_bounds"] - singles_post[best_single]["log_evidence_bounds"])
dlogE_same_order = {f"o{o:g}": float(trends_post[o]["log_evidence_bounds"] - singles_post[o]["log_evidence_bounds"])
                    for o in orders}
dlogZ_max = float(converged_max(gt) - max(converged_max(g) for g in singles.values()))
unconv = lambda g: float(1 - numpy.asarray(g["converged"]).astype(bool).mean())
diag = {**{f"single_o{o:g}": dict(unconverged_fraction=unconv(g), edge_weight=singles_post[o]["edge_weight"],
                                  n_points=int(g["logZ"].size)) for o, g in singles.items()},
        **{f"trend_o{o:g}": dict(unconverged_fraction=unconv(g), edge_weight=trends_post[o]["edge_weight"],
                                 n_points=int(g["logZ"].size)) for o, g in trends.items()}}
print("grid quality (unconverged fraction, edge weight):", {k: (round(d_["unconverged_fraction"], 3),
                                                              round(d_["edge_weight"], 3)) for k, d_ in diag.items()})

# Refinement: a small difference is re-evaluated on finer LOCAL grids (5 x 5 values, ratio sqrt 2, about each model's best
# cell: 25 + 25 points, ~1 min; the BPL's 625-point 4-D refinement took 800 s and had 0.39 edge weight) under the same
# bounds. Valid only if both posteriors are contained (edge weight < 0.05); else the coarse values stand.
dlogE, refined = dlogE_coarse, None
if abs(dlogE_coarse) < refine_below:
    with timed("D: refinement"):
        steps2 = dict(sigma=2**0.5, length=2**0.5)
        g1r = gp_evidence_grid_nd(resp, local_axes(b1, steps2, n=2), sg(best_single), beta=beta, outdir=f"{out}/grid",
                                  tag=f"o{best_single:g}fine_" + base_fp, abbrev=ABBREV, bounds=hyper_bounds, n_iter=n_iter,
                                  theta0=th_1, max_growth=1, recheck_top=recheck_top)
        gtr = gp_evidence_grid_nd(resp, local_axes(bt, steps2, n=2), tf(best_trend), beta=beta, outdir=f"{out}/grid",
                                  tag=f"trend_o{best_trend:g}fine_" + base_fp, abbrev=ABBREV, bounds=hyper_bounds,
                                  n_iter=n_iter, theta0=th_t, max_growth=1, recheck_top=recheck_top)
        p1r, ptr = gp_hyper_posterior(g1r), gp_hyper_posterior(gtr)
        valid = p1r["edge_weight"] < 0.05 and ptr["edge_weight"] < 0.05
        refined = dict(trend_log_evidence=ptr["log_evidence_bounds"], single_log_evidence=p1r["log_evidence_bounds"],
                       dlogE=float(ptr["log_evidence_bounds"] - p1r["log_evidence_bounds"]),
                       trend_edge_weight=ptr["edge_weight"], single_edge_weight=p1r["edge_weight"], valid=bool(valid),
                       trend_unconverged=unconv(gtr), single_unconverged=unconv(g1r))
        if valid:
            dlogE = refined["dlogE"]
        print(f"refinement: coarse dlogE {dlogE_coarse:+.2f}; refined {refined['dlogE']:+.2f} (edge weights trend "
              f"{ptr['edge_weight']:.3f}, single {p1r['edge_weight']:.3f}; used: {'refined' if valid else 'coarse'})")
use_trend = dlogE > 0
print(f"Matern + trend (o{best_trend:g}) vs single Matern (o{best_single:g}): difference of log evidence under the matched "
      f"hyperpriors {dlogE:+.2f} nats (coarse {dlogE_coarse:+.2f}; same order {dlogE_same_order}; max converged logZ "
      f"difference {dlogZ_max:+.2f}; the Laplace log Z is biased up with sigma, gp.py: validate by injection). "
      f"Use the trend for E/F: {use_trend}")

# trend amplitudes and the lowest DCT modes before / after; the trend_sigma sensitivity at the trend's best point
lap_1, lap_t = gp_laplace(resp, prior_1, th_1, beta), gp_laplace(resp, prior_t, th_t, beta)
tc = trend_components(prior_t, th_t, lap_t)


def low_modes(theta, laplace):
    """(z, <z^2> = z^2 + posterior variance) of the DCT modes (0, 2) and (2, 0)."""
    mu_, V_ = laplace
    z = numpy.asarray(theta)[:resolved_wid**2].reshape(resolved_wid, resolved_wid)
    var = numpy.asarray(np.sum(V_[:resolved_wid**2]**2/(1 + mu_), axis=1)).reshape(resolved_wid, resolved_wid)
    return {f"({j},{k})": dict(z=float(z[j, k]), z2=float(z[j, k]**2), posterior_power=float(z[j, k]**2 + var[j, k]))
            for j, k in ((0, 2), (2, 0))}


modes_before, modes_after = low_modes(th_1, lap_1), low_modes(th_t, lap_t)
print("DCT modes (0,2), (2,0) at the Matern's best:", modes_before, "\n  with the trend:", modes_after)
print(f"trend coefficients (dex, per unit-rms basis column {trend_kinds}): {numpy.asarray(tc['coef'])} +- "
      f"{numpy.asarray(tc['coef_sd'])}; trend map range {float(tc['trend'].min()):.2f} .. {float(tc['trend'].max()):.2f} dex")
sens = {}
for s_h in trend_sigma_check:
    p_s = tf(best_trend, s_h)(**bt)
    start_s = np.concatenate([p_s.fit_field(np.asarray(prior_t.field(th_t[:-1]))), th_t[-1:]])
    th_s, _ = gp_map(resp, p_s, start_s, beta, n_iter)
    sens[f"{s_h:g}"] = float(gp_log_evidence(resp, p_s, th_s, beta)["logZ"])
print("logZ at the trend's best (sigma, length) vs trend_sigma:", sens)

spec_1 = gp_spectral_diagnostic(prior_1, th_1, lap_1)
spec_t = gp_spectral_diagnostic(prior_t, th_t, lap_t)     # the trend latents are excluded
numpy.savez(f"{out}/spectral-diagnostic-D.npz", **{f"single_{k}": v for k, v in spec_1.items()},
            **{f"trend_{k}": v for k, v in spec_t.items()})
for name, sd_ in (("single", spec_1), ("trend", spec_t)):
    print(f"spectral diagnostic ({name}): <z^2> per bin {numpy.round(sd_['power'][:12], 2)} ... max |log10| deviation "
          f"{numpy.max(numpy.abs(numpy.log10(sd_['power']))):.2f}")
fig, ax = plt.subplots(figsize=(11, 9))
for name, sd_ in (("single Matern", spec_1), ("Matern + radial trend", spec_t)):
    ax.errorbar(sd_["freq"], sd_["power"], sd_["power_err"], fmt="o-", label=name)
ax.axhline(1, color="k", ls="--")
ax.set(xscale="log", yscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="<z^2> (prior: 1)")
ax.legend()
fig.savefig(f"{out}/spectral-diagnostic-D.png")
plt.close(fig)

comparison = dict(
    best_single_order=best_single, best_trend_order=best_trend, use_trend=use_trend,
    max_converged_logZ_difference_trend_minus_single=dlogZ_max, log_evidence_difference_matched_hyperpriors=dlogE,
    log_evidence_difference_coarse=dlogE_coarse, log_evidence_difference_same_order=dlogE_same_order, refinement=refined,
    grid_quality=diag, refine_below_nats=refine_below,
    trend_sigma=trend_sigma, trend_kinds=list(trend_kinds), trend_sigma_sensitivity_logZ=sens,
    trend_best=bt, single_best={f"o{o:g}": gp_best(g) for o, g in singles.items()},
    trend_best_per_order={f"o{o:g}": gp_best(g) for o, g in trends.items()},
    trend_amplitudes_dex=dict(coef=numpy.asarray(tc["coef"]).tolist(), coef_sd=numpy.asarray(tc["coef_sd"]).tolist(),
                              range_dex=[float(tc["trend"].min()), float(tc["trend"].max())]),
    dct_modes_single_best=modes_before, dct_modes_with_trend=modes_after,
    trend_log_evidence={f"o{o:g}": q["log_evidence_bounds"] for o, q in trends_post.items()},
    single_log_evidence={f"o{o:g}": q["log_evidence_bounds"] for o, q in singles_post.items()},
    nesting_amp_max_abs_diff=nest, nesting_logZ_abs_diff=nest_logZ,
    hyperpriors=dict(single=hyper_bounds, trend=hyper_bounds, trend_sigma="fixed"),
    spectral_max_abs_log10_single=float(numpy.max(numpy.abs(numpy.log10(spec_1["power"])))),
    spectral_max_abs_log10_trend=float(numpy.max(numpy.abs(numpy.log10(spec_t["power"])))))
with open(f"{out}/comparison-D.json", "w") as f:
    json.dump(comparison, f, indent=2, default=float)
if not use_trend:
    print("the radial trend does not beat the single Matern under the matched hyperpriors: stages E/F are not run; "
          "use imlup-gp.py's single-kernel pipeline. See comparison-D.json")
    sys.exit(0)

# the outer loop's order: stage D's best trend order, or loop_order (stage D's ranking flipped once the optics were refitted)
best_trend_D = best_trend
if loop_order is not None:
    assert loop_order in trends, f"loop_order {loop_order} is not in orders {orders}"
    best_trend = float(loop_order)
    gt = trends[best_trend]
    bt = gp_best(gt)
    th_t = np.asarray(gt["thetas"][gt["best"]])
    prior_t = tf(best_trend)(**bt)
    print(f"outer loop at order {best_trend:g} (loop_order; stage D's best trend order was {best_trend_D:g}), starting at {bt}")
# %%
# ---------------------------------------------------------------- E: outer iteration (Matern + radial trend)
checks = {}      # the end-to-end checks, written to e2e-checks.json (and summary.json) in stage F
mean_b, lp_b = None, lp_of(False)
make_b = lambda: trend_prior_factory(resolved_wid, best_trend, trend_sigma, mean_b, lp_b, basis=trend_basis, support=support)
hyp_b = dict(bt)                                    # {sigma, length}; the order and trend_sigma are fixed
outer_steps = dict(sigma=2**0.5, length=2**0.5)     # local grid: 3 values per axis (n=1)
hypers_of = lambda h: dict(sigma=h["sigma"], length=h["length"], order=best_trend, trend_sigma=trend_sigma)
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=gt,
               theta=th_t, idx=gt["best"], stage="D")

# the trend surrogate against the full model at the stage-D MAP (scaled errors): the loss must agree (cf. stage C)
try:
    exps_chk = scale_errors(set_hyper(exposures, **hypers_of(hyp_b), mean=numpy.zeros((resolved_wid, resolved_wid)),
                                      level_prior=lp_of(False)), numpy.asarray(s_noise))
    z_c, l_c = prior_t.params(th_t)
    sub = set_array({"resolved": {key: z_c}, "resolved_level": {key: l_c}})
    loss_c, g_c = _loss_grad(sub, jax.tree.map(np.ones_like, sub), make_batch(exps_chk, model_single), model_single)
    ev_c = gp_log_evidence(resp, prior_t, th_t, beta)
    sur_loss = float(ev_c['nll'].sum() + ev_c['prior'])
    checks["stage_D_surrogate"] = dict(full_loss=float(loss_c), surrogate_loss=sur_loss, abs_diff=abs(float(loss_c) - sur_loss),
                                       grad_norm_z=float(np.linalg.norm(g_c['resolved'][key])))
    print(f"trend surrogate vs full model at the stage-D MAP: full loss {float(loss_c):.4f}, surrogate "
          f"{sur_loss:.4f}; |grad| {float(np.linalg.norm(g_c['resolved'][key])):.2e}")
    if checks["stage_D_surrogate"]["abs_diff"] > 1e-3:
        print("WARNING: the trend surrogate does not reproduce the full model at the stage-D MAP (|dloss| > 1e-3): the response, prior or support "
              "disagree with the model: do not trust the grids")
except Exception as e:   # a diagnostic: never lose the stage to it
    checks["stage_D_surrogate"] = dict(error=repr(e))
    print(f"[check] trend surrogate vs full model failed: {e!r}")
outer_converged, outer_reason = False, "not converged"
outer_history = []
lbfgs_info = {}
for it in range(1, n_outer + 1):
    prior_b = make_b()(**hyp_b)
    h = fingerprint(it, list(hyp_b.values()), best_trend, trend_sigma, current["s"], current["resp_fp"], current["theta"], fo_epochs,
                    fo_lr_opd, fo_lr_disc, optx_max_steps, lp_b, repr(sorted(precond_kwargs.items())),
                    repr(sorted(things_gp)), code_fp)
    with timed(f"E{it}: refit"):
        exps_b = scale_errors(set_hyper(exposures, **hypers_of(hyp_b), mean=numpy.zeros((resolved_wid, resolved_wid)),
                                        level_prior=lp_b), current["s"])
        path = f"{out}/outer{it}-{h}-params.npy"
        ipath = f"{out}/outer{it}-{h}-lbfgs.json"
        if os.path.exists(path):
            p_new = numpy.load(path, allow_pickle=True).item()
            if os.path.exists(ipath):
                lbfgs_info[it] = json.load(open(ipath))
                if backfill_grad_split and "grad_split_end" not in lbfgs_info[it]:
                    # an older refit without the split: |sqrt(C) g| at its saved result with a FRESH gn_probe C (random probes,
                    # computed at the end point, not the start's) -- indicative, stored as grad_split_end_backfill
                    try:
                        model_r = set_array(ModelParams(p_new).inject(model_single))
                        opt_r = set_array({k_: p_new[k_] for k_ in things_gp})
                        gs_r = precond_grad_split(opt_r, get_C(opt_r, model_r, exps_b, things_gp), model_r, exps_b)
                        lbfgs_info[it]["grad_split_end_backfill"] = gs_r
                        print_split(f"outer {it} (resumed; fresh C)", gs_r)
                        with open(ipath, "w") as f:
                            json.dump(lbfgs_info[it], f, indent=2, default=str)
                    except Exception as e:
                        print(f"[grad] backfill for outer {it} failed: {e!r}")
        else:
            z_b, l_b = prior_b.params(current["theta"])
            start = dict(current["params"]) | {"resolved": {key: z_b}, "resolved_level": {key: l_b}}
            model_b = set_array(ModelParams(start).inject(model_single))
            opt_params = set_array({k: start[k] for k in things_gp})
            x_new, losses, optx_info = refit(opt_params, model_b, exps_b, things_gp)
            ok_str = str(optx.RESULTS.successful)     # core returns str(sol.result); this version prints 'optimistix._solution.RESULTS<>'
            lbfgs_info[it] = dict(optx_info, fo_loss_start=float(losses[0]), fo_loss_end=float(losses[-2]),
                                  reached_max_steps=bool(optx_info["num_steps"] >= optx_max_steps),
                                  successful=bool(str(optx_info["result"]) == ok_str
                                                  and optx_info["num_steps"] < optx_max_steps))
            fig = plt.figure(figsize=(10, 10))
            plt.plot(losses[:-1])
            plt.plot([fo_epochs], losses[-1:], "o", label=f"after L-BFGS ({optx_info['num_steps']} its)")
            plt.legend()
            fig.savefig(f"{out}/outer{it}-losses.png")
            plt.close(fig)
            if not (numpy.all(numpy.isfinite(losses)) and losses[-1] <= losses[0]):   # never worse than the start
                raise RuntimeError(f"outer refit {it}: loss {losses[0]:.1f} -> {losses[-1]:.1f} (non-finite or worse "
                                   "than the start); not saving it")
            p_new = start | x_new
            save_npy(f"{out}/outer{it}-{h}-losses.npy", losses)
            with open(ipath, "w") as f:
                json.dump(lbfgs_info[it], f, indent=2, default=str)
            save_npy(path, p_new)

    with timed(f"E{it}: response"):
        model_new = set_array(ModelParams(p_new).inject(model_single))
        rpath = f"{out}/response-outer{it}-{h}.npz"
        if os.path.exists(rpath):
            resp_new = LinearResponse.load(rpath)
        else:
            resp_new = build_linear_response(model_new, exposures, batch_size=response_batch)
            resp_new.save(rpath)
        resp_new_fp = response_fingerprint(resp_new)

    with timed(f"E{it}: grid"):
        npath = f"{out}/noise-outer{it}-{fingerprint(resp_new_fp, h)}.npz"
        if os.path.exists(npath):
            f = numpy.load(npath)
            s_new, theta_new = f["s"], np.asarray(f["theta"])
        else:
            theta_new = prior_b.theta(np.asarray(p_new["resolved"][key]), np.asarray(p_new["resolved_level"][key]))
            s_new, theta_new, _ = gp_error_scale(resp_new, prior_b, theta_new)
            _save_npz(npath, s=s_new, theta=theta_new)
        beta_new = 1/np.asarray(s_new)**2
        tag = f"outer{it}_trend_" + fingerprint(resp_new_fp, beta_new, lp_b, n_iter, code_fp)
        # local (2n + 1)^2 grid about the refit hyperparameters, grown (within hyper_bounds, at most outer_max_growth steps) while
        # a face holds posterior mass; values outside the bounds are dropped. Existing points are reused (same tag).
        grid_new = gp_evidence_grid_nd(resp_new, local_axes(hyp_b, outer_steps, n=outer_half_width), make_b(),
                                       beta=beta_new, outdir=f"{out}/grid", tag=tag, abbrev=ABBREV,
                                       bounds=hyper_bounds, n_iter=n_iter, theta0=theta_new,
                                       max_growth=outer_max_growth, edge_dlogz=grid_edge_dlogz, recheck_top=3)
        idx_new = grid_new["best"]
        new = {k: float(grid_new["axes"][k][i]) for k, i in zip(grid_new["names"], idx_new)}
        post_new = gp_hyper_posterior(grid_new)
        print(f"outer {it}: s {numpy.round(numpy.asarray(s_new), 4)}; best {new} (was {hyp_b}); grid "
              f"{ {k: len(grid_new['axes'][k]) for k in grid_new['names']} } edge weight {post_new['edge_weight']:.3f}; "
              f"16/50/84 % { {k: [round(float(x_), 3) for x_ in v_] for k, v_ in post_new['quantiles'].items()} }")
    # ---- re-linearisation diagnostics: how much did moving the optics (and re-building the response) change things at FIXED hyperparameters
    # and noise (beta_new for both responses)? Large values: the previous linearisation point was poor, another iteration is needed.
    try:
        th_old, _ = gp_map(current["resp"], prior_b, current["theta"], beta_new, n_iter)
        th_new = np.asarray(theta_new)
        lz_old = float(gp_log_evidence(current["resp"], prior_b, th_old, beta_new)["logZ"])
        lz_new = float(gp_log_evidence(resp_new, prior_b, th_new, beta_new)["logZ"])
        sup_b = numpy.asarray(support) > 0
        dmap = numpy.asarray(prior_b.log10_image(th_new)) - numpy.asarray(prior_b.log10_image(th_old))
        relin = dict(logZ_old_response=lz_old, logZ_new_response=lz_new, dlogZ=lz_new - lz_old,
                     map_disc_rms_change_dex=float(numpy.sqrt(numpy.mean(dmap[sup_b]**2))),
                     level_old=float(prior_b.params(th_old)[1]), level_new=float(prior_b.params(th_new)[1]))
    except Exception as e_r:     # a diagnostic: never lose the iteration to it
        relin = dict(error=repr(e_r))
    od = optics_deltas(current["params"], p_new, exposures, wid, support_scale)
    refit_drop = float(lbfgs_info[it]["fo_loss_start"] - lbfgs_info[it]["loss_end"]) if "fo_loss_start" in lbfgs_info.get(it, {}) else float("nan")
    print(f"outer {it} re-linearisation: refit loss drop {refit_drop:.1f}; at fixed (hyper, beta) log Z {relin.get('logZ_old_response', float('nan')):.1f} -> "
          f"{relin.get('logZ_new_response', float('nan')):.1f} (d {relin.get('dlogZ', float('nan')):+.1f}); MAP disc moved "
          f"{relin.get('map_disc_rms_change_dex', float('nan')):.3f} dex rms; optics: primary_opd rms change {od['primary_opd_rms_change_nm']:.2f} nm "
          f"(new rms {od['primary_opd_rms_new_nm']:.1f}), star moved {numpy.round(od['star_shift_px'], 3).tolist()} px, spectrum x{od['spectrum0_ratio']:.3f}", flush=True)
    moved = not all(numpy.isclose(new[k], hyp_b[k], rtol=1e-6, atol=1e-9) for k in hyp_b)
    # how far (grid steps, per axis) and how much log Z (nats) the best cell moved relative to the refit hyperparameters
    steps_moved = {k: float(abs(numpy.log(new[k]/hyp_b[k]))/numpy.log(outer_steps[k])) for k in hyp_b}
    old_cell = tuple(int(numpy.argmin(numpy.abs(numpy.log(grid_new["axes"][k]/hyp_b[k])))) for k in grid_new["names"])
    old_in_grid = all(numpy.isclose(grid_new["axes"][k][i], hyp_b[k], rtol=1e-6) for k, i in zip(grid_new["names"], old_cell))
    dlogZ_old = float(grid_new["logZ"][old_cell] - grid_new["logZ"][idx_new]) if old_in_grid else float("nan")
    loose = (moved and converge_dlogz is not None and old_in_grid and all(v_ <= 1 + 1e-6 for v_ in steps_moved.values())
             and dlogZ_old > -converge_dlogz and bool(grid_new["converged"][old_cell]))
    outer_history.append(dict(it=it, refit_at=dict(hyp_b), best=new, moved=bool(moved), steps_moved=steps_moved,
                              dlogZ_refit_cell_minus_best=dlogZ_old, loose_converged=bool(loose),
                              edge_weight=post_new["edge_weight"], contained=bool(post_new["edge_weight"] < edge_target),
                              quantiles={k: list(map(float, v_)) for k, v_ in post_new["quantiles"].items()},
                              axes_size={k: len(grid_new["axes"][k]) for k in grid_new["names"]},
                              n_points=int(grid_new["logZ"].size), unconverged=int((~grid_new["converged"].astype(bool)).sum()),
                              refit_loss_drop=refit_drop, relinearisation=relin, optics_change=od))
    hyp_b = new
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=np.asarray(grid_new["thetas"][idx_new]), idx=idx_new, stage=f"E{it}")
    if (not moved or loose) and it < min_outer:
        print(f"outer {it}: hyperparameters stationary, but at least {min_outer} iterations are required (the first only re-linearises at the "
              "refit optics): continuing")
    elif not moved or loose:
        outer_converged, outer_reason = True, ("best cell did not move" if not moved else
                                               f"moved <= 1 step and {-dlogZ_old:.2f} < {converge_dlogz} nats (refit cell nearly as good)")
        print(f"outer {it}: converged ({outer_reason})")
        break
if not outer_converged:
    print(f"WARNING: the hyperparameters still moved at outer iteration {n_outer}: the optics were refitted at the "
          "previous hyperparameters; raise n_outer")
if outer_history and abs(outer_history[-1]["relinearisation"].get("dlogZ", 0.)) > lin_dlogz_warn:
    print(f"WARNING: the last outer iteration's re-linearisation still changed log Z by {outer_history[-1]['relinearisation']['dlogZ']:+.1f} nats "
          f"(> {lin_dlogz_warn:g}) at fixed hyperparameters: the optics are not at the joint optimum; raise n_outer")
checks["outer_history_relinearisation"] = [dict(it=h_["it"], refit_loss_drop=h_["refit_loss_drop"], **{k_: h_["relinearisation"].get(k_) for k_ in ("dlogZ", "map_disc_rms_change_dex")},
                                               primary_opd_rms_change_nm=h_["optics_change"]["primary_opd_rms_change_nm"], star_shift_px=h_["optics_change"]["star_shift_px"],
                                               spectrum0_ratio=h_["optics_change"]["spectrum0_ratio"]) for h_ in outer_history]
if outer_history and outer_history[-1] and abs(outer_history[-1]["optics_change"]["spectrum0_ratio"] - 1) > 0.1:
    print("WARNING: the star flux (leading spectrum coefficient) still moved by > 10 % in the last outer iteration")
if outer_history and outer_history[-1]["edge_weight"] >= edge_target:
    print(f"WARNING: the final local grid still has edge weight {outer_history[-1]['edge_weight']:.3f} >= {edge_target} "
          f"(axes {outer_history[-1]['axes_size']}): its quantiles are truncated (growth capped at {outer_max_growth})")
for it, info in lbfgs_info.items():
    if not info.get("successful", False) or info.get("reached_max_steps", False):
        print(f"WARNING: outer {it} L-BFGS did not report success ({info.get('result')}, {info.get('num_steps')} its, "
              f"|sqrt(C) grad| end {info.get('grad_norm_u_end')}): the optics may be under-converged")


# %%
# ---------------------------------------------------------------- G: like-for-like re-evaluation, every order, plain and trend
# The first run established the +29 nats only at the stage-D optics, and a later run showed the order preference depends on the
# optics (stage D favoured trend o1, the final response puts plain o1.5 within 6 nats of it, on a grid that was not contained). So
# on the FINAL response evaluate the plain Matern AND Matern + trend at EVERY order in `orders` (same beta, level prior lp_b,
# hyper_bounds; coarse pass from that model's stage-D best, then fine grids grown until contained where the budgets allow), rank
# all of them, and (G2) add a continuous-order evidence grid for each of the two families.
# Each fine grid is grown until its edge weight is < edge_target: gp_evidence_grid_nd extends a face while it holds a cell within
# g_edge_dlogz nats of the best (looser than edge_target alone guarantees: 8 nats ~ 3e-4 per edge cell), one growth step at a
# time (re-calling it with max_growth + 1 replays the saved points, so only the new cells are computed). Before each step the
# size of the next grid is predicted (the faces that will grow) and the step is refused if it would pass g_max_points, if the
# predicted time (new points x the measured s/point of this grid, 1.4 s before any is measured) would pass g_max_seconds (this
# grid) or the stage budget g_stage_budget_s (all of G and G2: later growth is skipped; the first, unavoidable evaluation of each
# grid is not), or when the length axis is flat (a growth step on a length face that holds the edge mass changed the profile
# logZ(length) by < 1 nat: "length_unidentified", more points would not contain it). Whatever stops a grid, its record has
# "contained" (edge weight < edge_target) and "stop_reason".
# The edge weight used here treats a face that can neither grow (the next cell lies past the hyperprior's bound) nor already
# "reaches" the bound (its cell stops within half a step of it: gp_hyper_posterior counts that face as edge) as at-bound, so
# a flat length posterior that runs to the 200 px bound is not falsely uncontained. gp's own number is kept as edge_weight_gp.
g_edge_dlogz = 8.
g_max_growth = 14
g_max_points = 250
g_max_seconds = 240.        # wall clock per grid for growth steps (checked against the predicted cost of the next step)
g_stage_budget_s = 25*60.   # all of G and G2: after it the remaining growth is skipped, and G2 is skipped if it has already passed
g_flat_dlogz = 1.           # a length profile changing by less than this over the last growth step: length unidentified
g_decisive_dlogE = 5.       # nats: |dlogE| of the best model over the runner-up that counts as decisive (with both grids contained)
# G2: order as a third (linear) axis. nu = order - 1 >= 0, so order < 1 is not a valid 2-D Matern: lower bound 1. Upper bound 3.5 =
# the largest order in `orders` (the discrete evidence is already ~160 nats below the best there, and beyond it the Matern is
# ever closer to the squared exponential). The hyperprior is uniform on [1, 3.5] (sigma, length: log-uniform on hyper_bounds, as
# before), so the order-marginal evidence (log_evidence_bounds) is comparable between plain and trend. Initial order axis 1 .. 2 in
# steps of 0.25 (grown up to the bound if posterior mass reaches its top face; widened to best discrete order + 0.5 when that is
# higher); sigma / length about each family's best discrete (order, sigma, length): sigma 5 values at ratio 2^(1/12), length 5 at
# 2^(1/2) (125 base points per family), grown at most g_cont_max_growth steps and g_cont_max_points points.
g_continuous_order = True
g_order_bounds = (1., 3.5)
g_order_step, g_order_hi0 = 0.25, 2.
g_cont_steps = dict(sigma=2**(1/12), length=2**0.5)
g_cont_half_width = dict(sigma=2, length=2)
g_cont_max_growth = 2
g_cont_max_points = 350
ORDER_ABBREV = ABBREV | {"order": "o"}    # the point file name carries the order (the tag is unique too)


def effective_edge_weight(grid, post, only=None):
    """Posterior weight on faces that are neither at the hyperprior's bound (gp's at_bound) nor unable to grow (the next cell
    would lie past the bound, so the sliver to the bound is unreachable): gp_hyper_posterior's edge_weight without those
    slivers. only: restrict to these axis names."""
    w, bnd = post["weights"], grid.get("bounds", {})
    edge = numpy.zeros(w.shape, dtype=bool)
    for i, a in enumerate(_axes(grid)):
        if len(a.vals) < 2 or a.scale == "periodic" or (only is not None and a.name not in only):
            continue
        for lower, face in ((True, 0), (False, len(a.vals) - 1)):
            if a.at_bound(lower, bnd.get(a.name)) or not a.can_grow(lower, bnd.get(a.name)):
                continue
            sl = [slice(None)]*w.ndim
            sl[i] = face
            edge[tuple(sl)] = True
    return float(w[edge].sum())


def grid_record(grid, post, info=None, **extra):
    """Per-model evidence record of a grid (JSON-safe): log_evidence = log_evidence_bounds, best, quantiles, edge weight
    (effective, see effective_edge_weight; gp's in edge_weight_gp), contained (edge weight < edge_target), unconverged cells,
    size, and info from contained_grid (stop_reason, growth_steps, seconds)."""
    ew = effective_edge_weight(grid, post)
    return dict(**extra, **(info or {}), log_evidence=float(post["log_evidence_bounds"]), best=gp_best(grid), edge_weight=ew,
                edge_weight_gp=float(post["edge_weight"]), contained=bool(ew < edge_target), n_points=int(grid["logZ"].size),
                axes_size={k: len(grid["axes"][k]) for k in grid["names"]},
                unconverged=int((~grid["converged"].astype(bool)).sum()), max_converged_logZ=float(converged_max(grid)),
                quantiles={k: list(map(float, v_)) for k, v_ in post["quantiles"].items()})


def contained_grid(resp, beta, axes, make_prior, tag, theta0, scales=None, bounds=None, abbrev=ABBREV, growth_cap=g_max_growth,
                   points_cap=g_max_points, seconds_cap=g_max_seconds):
    """gp_evidence_grid_nd grown one step at a time (from 0, replaying the saved points) until the effective edge weight is
    < edge_target or a limit stops it (stop_reason): "contained", "no_growth_possible" (a step added nothing), "length_unidentified",
    "max_growth", "max_points" (the next grid would pass points_cap), "time" (predicted cost past seconds_cap),
    "stage_budget" (past g_deadline). Returns (grid, posterior, info)."""
    bnd = hyper_bounds if bounds is None else bounds
    t_start = time.perf_counter()
    n_files = lambda: len(glob.glob(f"{out}/grid/{tag}_*.npz"))

    def run(mg, rt):
        return gp_evidence_grid_nd(resp, axes, make_prior, beta=beta, outdir=f"{out}/grid", tag=tag, abbrev=abbrev, bounds=bnd,
                                   scales=scales, n_iter=n_iter, theta0=theta0, max_growth=mg, edge_dlogz=g_edge_dlogz,
                                   recheck_top=rt, verbose=False)
    mg, prev_size, prev_len, new_total, new_secs, reason = 0, None, None, 0, 0., None
    while True:
        n0, t0 = n_files(), time.perf_counter()
        grid = run(mg, 0)
        post = gp_hyper_posterior(grid)
        n_new = n_files() - n0
        if n_new > 0:
            new_total, new_secs = new_total + n_new, new_secs + time.perf_counter() - t0
        names, logZ = grid["names"], numpy.asarray(grid["logZ"], dtype=float)
        size = tuple(len(grid["axes"][k]) for k in names)
        ew = effective_edge_weight(grid, post)
        if ew < edge_target:
            reason = "contained"
        elif prev_size is not None and size == prev_size:
            reason = "no_growth_possible"
        elif mg >= growth_cap:
            reason = "max_growth"
        if reason is None and "length" in names and prev_len is not None:   # flat length profile over the last growth step?
            il = names.index("length")
            conv = numpy.asarray(grid["converged"]).astype(bool)
            prof = numpy.where(conv, logZ, -numpy.inf).max(axis=tuple(j for j in range(logZ.ndim) if j != il))
            lv = numpy.asarray(grid["axes"]["length"])
            lo_add, hi_add = int(numpy.sum(lv < prev_len[0]*(1 - 1e-9))), int(numpy.sum(lv > prev_len[-1]*(1 + 1e-9)))
            deltas = ([abs(prof[0] - prof[lo_add])] if lo_add else []) + ([abs(prof[-1] - prof[-1 - hi_add])] if hi_add else [])
            if deltas and max(deltas) < g_flat_dlogz and effective_edge_weight(grid, post, only=("length",)) >= 0.5*ew:
                reason = "length_unidentified"
        if reason is None:     # predict the next step: the faces within g_edge_dlogz of the best (and able to grow) get a cell
            near = logZ > numpy.max(logZ) - g_edge_dlogz
            nxt = list(size)
            for i, a in enumerate(_axes(grid)):
                for lower, face in ((True, 0), (False, -1)):
                    if numpy.take(near, face, axis=i).any() and a.can_grow(lower, bnd.get(a.name)):
                        nxt[i] += 1
            n_pts, n_next = int(logZ.size), int(numpy.prod(nxt))
            spp = new_secs/new_total if new_total else 1.4
            pred = (n_next - n_pts)*spp
            if n_next > points_cap:
                reason = "max_points"
            elif time.perf_counter() - t_start + pred > seconds_cap:
                reason = "time"
            elif time.perf_counter() + pred > g_deadline:
                reason = "stage_budget"
        if reason is not None:
            break
        prev_size, prev_len, mg = size, numpy.asarray(grid["axes"]["length"]) if "length" in names else None, mg + 1
    grid = run(mg, recheck_top)
    post = gp_hyper_posterior(grid)
    return grid, post, dict(stop_reason=reason, growth_steps=mg, seconds=float(time.perf_counter() - t_start), new_points=new_total)


def rank_models(models):
    """models {name: record}: ranking by log_evidence, the best over the runner-up and whether that is decisive."""
    names = sorted(models, key=lambda k_: -models[k_]["log_evidence"])
    top = models[names[0]]["log_evidence"]
    d_ru = float(top - models[names[1]]["log_evidence"]) if len(names) > 1 else float("inf")
    both = bool(models[names[0]]["contained"] and (len(names) < 2 or models[names[1]]["contained"]))
    return dict(ranking=[dict(model=k_, log_evidence=models[k_]["log_evidence"], dlogE_to_best=float(models[k_]["log_evidence"] - top),
                              contained=models[k_]["contained"], stop_reason=models[k_].get("stop_reason")) for k_ in names],
                best=names[0], runner_up=names[1] if len(names) > 1 else None, dlogE_best_minus_runner_up=d_ru,
                best_contained=bool(models[names[0]]["contained"]),
                runner_up_contained=bool(models[names[1]]["contained"]) if len(names) > 1 else None,
                decisive=bool(d_ru > g_decisive_dlogE and both), decisive_threshold_nats=g_decisive_dlogE)


def print_models(models, rank):
    print(f"{'model':<12}{'logE':>10}{'dlogE':>8}{'sigma':>8}{'length':>8}{'edge':>7}{'cont':>6}{'unconv':>7}{'npts':>6}  stop")
    for r_ in rank["ranking"]:
        m_ = models[r_["model"]]
        print(f"{r_['model']:<12}{m_['log_evidence']:>10.2f}{r_['dlogE_to_best']:>8.2f}{m_['best']['sigma']:>8.3f}"
              f"{m_['best']['length']:>8.1f}{m_['edge_weight']:>7.3f}{str(m_['contained']):>6}{m_['unconverged']:>7}{m_['n_points']:>6}"
              f"  {m_.get('stop_reason')}")


def log_mean_exp(v):
    v = numpy.asarray(v, dtype=float)
    return float(numpy.logaddexp.reduce(v) - numpy.log(v.size))


g_deadline = time.perf_counter() + g_stage_budget_s    # reset at the start of the stage below
with timed("G: plain Matern + trend at every order on the final response"):
    g_deadline = time.perf_counter() + g_stage_budget_s
    fin_resp, fin_beta = current["resp"], current["beta"]
    fin_fp = fingerprint(current["resp_fp"], fin_beta, lp_b, n_iter, code_fp)
    # the loop's trend (order best_trend): the same tag and axes as before (saved points reused), now grown until contained
    fin_grid, fin_post, fin_info = contained_grid(fin_resp, fin_beta, fine_axes(hyp_b), make_b(), "final_fine_trend_" + fin_fp,
                                                  current["theta"])
    hyp_b = {k: float(fin_grid["axes"][k][i]) for k, i in zip(fin_grid["names"], fin_grid["best"])}   # the fine best
    current = dict(current, grid=fin_grid, theta=np.asarray(fin_grid["thetas"][fin_grid["best"]]), idx=fin_grid["best"],
                   stage=current["stage"] + "+fine")          # stage F's products / mixture use the loop model's fine grid
    fin_trends = {f"o{best_trend:g}": grid_record(fin_grid, fin_post, fin_info, order=best_trend, coarse_log_evidence=None)}
    fin_trend_grids = {best_trend: fin_grid}
    print(f"[final trend o{best_trend:g}] log_evidence_bounds {fin_post['log_evidence_bounds']:.2f} at {hyp_b}; edge "
          f"{fin_trends[f'o{best_trend:g}']['edge_weight']:.3f}; {fin_trends[f'o{best_trend:g}']['n_points']} points; "
          f"{fin_info['stop_reason']}")
    # the other orders' trends: coarse pass from the stage-D trend best of that order (its theta warm-starts it), then fine
    for o_t in orders:
        if o_t == best_trend:
            continue
        gt_o = trends[o_t]
        gc_t = gp_evidence_grid_nd(fin_resp, local_axes(gp_best(gt_o), outer_steps, n=outer_half_width), tf(o_t), beta=fin_beta,
                                   outdir=f"{out}/grid", tag=f"final_trend_o{o_t:g}_" + fin_fp, abbrev=ABBREV,
                                   bounds=hyper_bounds, n_iter=n_iter, theta0=np.asarray(gt_o["thetas"][gt_o["best"]]),
                                   max_growth=outer_max_growth, edge_dlogz=grid_edge_dlogz, recheck_top=recheck_top)
        gf_t, pf_t, if_t = contained_grid(fin_resp, fin_beta, fine_axes(gp_best(gc_t)), tf(o_t), f"final_fine_trend_o{o_t:g}_" + fin_fp,
                                          np.asarray(gc_t["thetas"][gc_t["best"]]))
        fin_trends[f"o{o_t:g}"] = grid_record(gf_t, pf_t, if_t, order=o_t,
                                              coarse_log_evidence=float(gp_hyper_posterior(gc_t)["log_evidence_bounds"]))
        fin_trend_grids[o_t] = gf_t
        print(f"[final trend o{o_t:g}] log_evidence_bounds {pf_t['log_evidence_bounds']:.2f} at {gp_best(gf_t)}; edge "
              f"{fin_trends[f'o{o_t:g}']['edge_weight']:.3f}; {fin_trends[f'o{o_t:g}']['n_points']} points; {if_t['stop_reason']}")
    fin_trend = fin_trends[f"o{best_trend:g}"]
    fin_plain, fin_plain_grids = {}, {}
    for o_p in orders:
        gs_o = singles[o_p]
        # coarse pass from the stage-D best (the plain optimum may have moved), then the same fine grid as the trend's
        gc_p = gp_evidence_grid_nd(fin_resp, local_axes(gp_best(gs_o), outer_steps, n=outer_half_width), sg(o_p), beta=fin_beta,
                                   outdir=f"{out}/grid", tag=f"final_plain_o{o_p:g}_" + fin_fp, abbrev=ABBREV,
                                   bounds=hyper_bounds, n_iter=n_iter, theta0=np.asarray(gs_o["thetas"][gs_o["best"]]),
                                   max_growth=outer_max_growth, edge_dlogz=grid_edge_dlogz, recheck_top=recheck_top)
        gp_p, pp, if_p = contained_grid(fin_resp, fin_beta, fine_axes(gp_best(gc_p)), sg(o_p), f"final_fine_plain_o{o_p:g}_" + fin_fp,
                                        np.asarray(gc_p["thetas"][gc_p["best"]]))
        best_p = gp_best(gp_p)
        th_p = np.asarray(gp_p["thetas"][gp_p["best"]])
        # nesting at this point: the trend prior with trend_sigma = 0 (b = 0) must reproduce the plain Matern's log Z
        ev_p0 = gp_log_evidence(fin_resp, sg(o_p)(**best_p), th_p, fin_beta)
        ev_p1 = gp_log_evidence(fin_resp, gp_trend_prior(resolved_wid, best_p["sigma"], best_p["length"], o_p, 0., None,
                                                         lp_b, support=support), embed_trend_theta(th_p, n_b), fin_beta)
        fin_plain[f"o{o_p:g}"] = grid_record(gp_p, pp, if_p, order=o_p, coarse_log_evidence=float(gp_hyper_posterior(gc_p)["log_evidence_bounds"]),
                                             nesting_logZ_abs_diff=float(np.abs(ev_p0["logZ"] - ev_p1["logZ"])))
        fin_plain_grids[o_p] = gp_p
        print(f"[final plain o{o_p:g}] log_evidence_bounds {pp['log_evidence_bounds']:.2f} (coarse "
              f"{fin_plain[f'o{o_p:g}']['coarse_log_evidence']:.2f}) at {best_p}; edge {fin_plain[f'o{o_p:g}']['edge_weight']:.3f}; "
              f"{fin_plain[f'o{o_p:g}']['n_points']} points; {if_p['stop_reason']}; nesting |dlogZ| "
              f"{fin_plain[f'o{o_p:g}']['nesting_logZ_abs_diff']:.1e}")
    best_plain_key = max(fin_plain, key=lambda k_: fin_plain[k_]["log_evidence"])
    best_trend_key = max(fin_trends, key=lambda k_: fin_trends[k_]["log_evidence"])
    dlogE_final = float(fin_trend["log_evidence"] - fin_plain[best_plain_key]["log_evidence"])
    dlogE_final_same_order = float(fin_trend["log_evidence"] - fin_plain[f"o{best_trend:g}"]["log_evidence"])
    dlogE_best_families = float(fin_trends[best_trend_key]["log_evidence"] - fin_plain[best_plain_key]["log_evidence"])
    fin_valid = bool(fin_trend["contained"] and all(d_["contained"] for d_ in fin_plain.values()))
    fin_valid_all = bool(fin_valid and all(d_["contained"] for d_ in fin_trends.values()))
    all_models = {f"plain_o{k_[1:]}": v_ for k_, v_ in fin_plain.items()} | {f"trend_o{k_[1:]}": v_ for k_, v_ in fin_trends.items()}
    rank_final = rank_models(all_models)
    # discrete order marginal (uniform over `orders`) of each family: log mean exp of the per-order evidence
    order_marginal_discrete = dict(plain=log_mean_exp([v_["log_evidence"] for v_ in fin_plain.values()]),
                                   trend=log_mean_exp([v_["log_evidence"] for v_ in fin_trends.values()]))
    order_marginal_discrete["trend_minus_plain"] = order_marginal_discrete["trend"] - order_marginal_discrete["plain"]
    print("final response, every model (log_evidence_bounds, conditional on the order):")
    print_models(all_models, rank_final)
    print(f"best {rank_final['best']}; over the runner-up {rank_final['runner_up']}: {rank_final['dlogE_best_minus_runner_up']:+.2f} nats "
          f"(runner-up contained: {rank_final['runner_up_contained']}); decisive (> {g_decisive_dlogE:g} nats, both contained): "
          f"{rank_final['decisive']}; discrete order marginal trend - plain {order_marginal_discrete['trend_minus_plain']:+.2f}; "
          f"{time.perf_counter() - (g_deadline - g_stage_budget_s):.0f} s of the {g_stage_budget_s:.0f} s stage budget used")

    # recommendation (Stage F's products stay those of the loop's model: its optics were refitted with trend o{best_trend})
    loop_model = f"trend_o{best_trend:g}"
    recommendation = dict(loop_model=loop_model, best_model=rank_final["best"], switch_recommended=False)
    if rank_final["best"] != loop_model:
        d_loop = float(all_models[rank_final["best"]]["log_evidence"] - all_models[loop_model]["log_evidence"])
        recommendation |= dict(dlogE_best_minus_loop_model=d_loop, decisive=rank_final["decisive"])
        if d_loop > g_decisive_dlogE and all_models[rank_final["best"]]["contained"] and all_models[loop_model]["contained"]:
            recommendation["switch_recommended"] = True
            print(f"RECOMMENDATION: {rank_final['best']} beats the outer loop's {loop_model} by {d_loop:+.2f} nats on the final response "
                  f"(both grids contained). Stage F below still uses {loop_model}. The optics were refitted WITH {loop_model}, so a "
                  "like-for-like comparison needs an outer-loop rerun at the winning model (order / trend or plain): rerun it before "
                  "trusting this margin.")
        else:
            print(f"NOTE: {rank_final['best']} has the highest evidence on the final response ({d_loop:+.2f} nats over the loop's "
                  f"{loop_model}) but it is not decisive (threshold {g_decisive_dlogE:g} nats, both grids contained: "
                  f"{all_models[rank_final['best']]['contained'] and all_models[loop_model]['contained']}). Stage F uses {loop_model}; "
                  "an outer-loop rerun at the preferred order would be needed to settle it.")
    recommendation["note"] = ("Stage F products come from the outer loop's model; a switch needs an outer-loop rerun (new optics) at the "
                              "winning model for a like-for-like comparison")
    cont = {}
    comparison_final = dict(
        dlogE_trend_minus_best_plain=dlogE_final, best_plain=best_plain_key, dlogE_same_order=dlogE_final_same_order,
        grid_quality_valid=fin_valid, edge_target=edge_target, trend=fin_trend, plain=fin_plain,
        dlogE_stage_D=dlogE, note="fine grids (sigma 9 values at 2^(1/16), length 7 at 2^(1/4), grown until contained where the budgets "
        "allow) for both; same final response, same beta, same hyper_bounds and level prior; `trend` is the outer loop's order only "
        "(see trend_all for every order); the optics were refitted WITH the trend, so a plain Matern refit would shift too",
        trend_all=fin_trends, best_trend_order_key=best_trend_key, dlogE_best_trend_minus_best_plain=dlogE_best_families,
        grid_quality_valid_all=fin_valid_all, ranking=rank_final, order_marginal_discrete=order_marginal_discrete,
        continuous_order=cont, recommendation=recommendation,
        growth_settings=dict(edge_dlogz=g_edge_dlogz, max_growth=g_max_growth, max_points=g_max_points, max_seconds=g_max_seconds,
                             stage_budget_s=g_stage_budget_s, flat_dlogz=g_flat_dlogz, decisive_dlogE=g_decisive_dlogE))
    with open(f"{out}/comparison-final.json", "w") as f:     # written BEFORE G2, rewritten with its results below
        json.dump(comparison_final, f, indent=2, default=float)
    print(f"final-stage dlogE (loop trend - plain {best_plain_key}) {dlogE_final:+.2f} nats (same order {dlogE_final_same_order:+.2f}; "
          f"stage D {dlogE:+.2f}); grid quality valid (loop trend + plain contained): {fin_valid}; all models: {fin_valid_all}")

# %%
# G2: continuous order (a third, linear axis), plain and trend, local in sigma and length about each family's best discrete point.
# A failure here is recorded (cont["error"]) and must not lose the G ranking (already written) or stage F.
if g_continuous_order:
    with timed("G2: continuous-order evidence grids"):
        try:
            if time.perf_counter() > g_deadline:
                cont["skipped"] = "stage budget already used by G"
                print(f"G2 skipped: the {g_stage_budget_s:.0f} s stage budget is used up")
            else:
                cont_make = dict(
                    plain=lambda sigma, length, order: gp_prior(resolved_wid, sigma, length, order, None, lp_b, support=support),
                    trend=lambda sigma, length, order: trend_prior_factory(resolved_wid, order, trend_sigma, None, lp_b,
                                                                           basis=trend_basis, support=support)(sigma, length))
                for kind, recs, grids_k in (("plain", fin_plain, fin_plain_grids), ("trend", fin_trends, fin_trend_grids)):
                    key_o = max(recs, key=lambda k_: recs[k_]["log_evidence"])
                    o_best = recs[key_o]["order"]
                    g_best = grids_k[o_best]
                    c_ = recs[key_o]["best"]
                    hi_ = min(g_order_bounds[1], max(g_order_hi0, g_order_step*float(numpy.ceil((o_best + 0.5)/g_order_step))))
                    n_o = int(round((hi_ - g_order_bounds[0])/g_order_step))
                    axes_c = {k_: [c_[k_]*g_cont_steps[k_]**i_ for i_ in range(-g_cont_half_width[k_], g_cont_half_width[k_] + 1)]
                              for k_ in ("sigma", "length")}
                    axes_c["order"] = [g_order_bounds[0] + g_order_step*i_ for i_ in range(n_o + 1)]
                    gc_o, pc_o, ic_o = contained_grid(fin_resp, fin_beta, axes_c, cont_make[kind], f"final_cont_{kind}_" + fin_fp,
                                                      np.asarray(g_best["thetas"][g_best["best"]]), scales={"order": "linear"},
                                                      bounds=hyper_bounds | {"order": g_order_bounds}, abbrev=ORDER_ABBREV,
                                                      growth_cap=g_cont_max_growth, points_cap=g_cont_max_points)
                    w_o = pc_o["weights"].sum(axis=tuple(range(pc_o["weights"].ndim - 1)))    # order is the last axis
                    cont[kind] = grid_record(gc_o, pc_o, ic_o, centre_order=o_best,
                                             order_axis=list(map(float, gc_o["axes"]["order"])),
                                             order_marginal_weights=list(map(float, w_o)),
                                             order_prior_bounds=list(g_order_bounds), order_scale="linear")
                    print(f"[continuous order {kind}] order-marginal log_evidence_bounds {cont[kind]['log_evidence']:.2f}; order "
                          f"16/50/84 % {numpy.round(cont[kind]['quantiles']['order'], 3)}; best {cont[kind]['best']}; edge "
                          f"{cont[kind]['edge_weight']:.3f} (contained: {cont[kind]['contained']}, {cont[kind]['stop_reason']}); "
                          f"{cont[kind]['n_points']} points {cont[kind]['axes_size']}")
                cont["trend_minus_plain"] = float(cont["trend"]["log_evidence"] - cont["plain"]["log_evidence"])
                cont["both_contained"] = bool(cont["trend"]["contained"] and cont["plain"]["contained"])
                cont["note"] = ("order-marginal evidence = log_evidence_bounds on sigma, length (log-uniform on hyper_bounds) and order "
                                "(uniform on order_prior_bounds); sigma / length axes are local windows, so the plain and the trend "
                                "grids are comparable with each other, not with the per-order evidence of the discrete table")
                print(f"continuous-order marginal evidence: trend - plain {cont['trend_minus_plain']:+.2f} nats "
                      f"(both contained: {cont['both_contained']}; discrete-order marginal "
                      f"{order_marginal_discrete['trend_minus_plain']:+.2f})")
                q_ = cont["trend"]["quantiles"]["order"]
                if not (q_[0] <= best_trend <= q_[2]):
                    print(f"NOTE: the loop's order {best_trend:g} lies outside the continuous-order posterior 16-84 % interval "
                          f"{numpy.round(q_, 3)} of the trend model: the outer loop's order may be off; rerun it near the posterior median.")
        except Exception as e_g2:    # keep the G results and stage F
            cont["error"] = repr(e_g2)
            print(f"WARNING: G2 (continuous order) failed and was skipped: {e_g2!r}")
        with open(f"{out}/comparison-final.json", "w") as f:
            json.dump(comparison_final, f, indent=2, default=float)

# %%
# ---------------------------------------------------------------- F: products
with timed("F: products"):
    resp_f, beta_f, grid_f = current["resp"], current["beta"], current["grid"]
    prior_f = make_b()(**hyp_b)
    theta_f = current["theta"]
    ev_f = to_host(gp_log_evidence(resp_f, prior_f, theta_f, beta_f))
    lap = gp_laplace(resp_f, prior_f, theta_f, beta_f)
    nan_off = lambda a_: numpy.where(support > 0, numpy.asarray(a_), numpy.nan)    # the data say nothing outside the support
    log_map = nan_off(prior_f.log10_image(theta_f))
    log_std = numpy.asarray(gp_log10_std(prior_f, lap))
    z_f, l_f = prior_f.params(theta_f)
    tc_f = to_host(trend_components(prior_f, theta_f, lap))     # coef (dex), coef_sd, trend / trend_std maps, DCT field
    for k_ in ("trend", "trend_std", "field_dct"):
        tc_f[k_] = nan_off(tc_f[k_])

    # posterior mixture over the final (sigma, length) grid (every bound is an axis bound: gp_hyper_posterior's weights ARE the posterior)
    post_f = gp_hyper_posterior(grid_f)
    flat_w = post_f["weights"].ravel()
    top = numpy.argsort(flat_w)[::-1][:20]               # at most 20 cells (each costs an eigh): renormalised
    w_top = numpy.zeros_like(flat_w)
    w_top[top] = flat_w[top]/flat_w[top].sum()
    print(f"mixture: top 20 cells carry {flat_w[top].sum():.3f} of the posterior weight")
    draws, n_mix = [], 400
    for idx, w in numpy.ndenumerate(w_top.reshape(post_f["weights"].shape)):
        m = int(round(n_mix*w))
        if m == 0:
            continue
        pt = {k: float(grid_f["axes"][k][i]) for k, i in zip(grid_f["names"], idx)}
        prior_c = make_b()(**pt)
        th = np.asarray(grid_f["thetas"][idx])
        draws.append(jax.vmap(prior_c.distribution)(
            gp_laplace_samples(th, gp_laplace(resp_f, prior_c, th, beta_f), jr.PRNGKey(10 + hash(idx) % 1000), n=m)))
    mix = np.concatenate(draws)
    mix_mean, mix_std = nan_off(mix.mean(0).reshape(support.shape)), nan_off(mix.std(0).reshape(support.shape))
    spec = gp_spectral_diagnostic(prior_f, theta_f, lap)
    q_f = {k: list(map(float, v)) for k, v in post_f["quantiles"].items()}
    outside = {k: v for k, v in hyp_b.items() if not (q_f[k][0] <= v <= q_f[k][2])}
    if outside:
        print(f"NOTE: the selected (best-cell) values {outside} lie outside their marginal 16-84 % intervals "
              f"{ {k: q_f[k] for k in outside} } (joint mode vs marginal; the intervals respect every bound)")

    numpy.save(f"{out}/gp-disc-log10.npy", log_map)
    numpy.save(f"{out}/gp-disc-log10-std.npy", log_std)
    numpy.save(f"{out}/gp-mixture-mean.npy", mix_mean)
    numpy.save(f"{out}/gp-mixture-std.npy", mix_std)
    numpy.save(f"{out}/gp-trend-log10.npy", tc_f["trend"])          # the radial trend as its own image (log10, dex)
    numpy.save(f"{out}/gp-trend-log10-std.npy", tc_f["trend_std"])
    numpy.save(f"{out}/gp-field-dct-log10.npy", tc_f["field_dct"])  # the stationary Matern field (log10, dex)
    numpy.savez(f"{out}/spectral-diagnostic.npz", **spec)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    lo, hi = numpy.nanpercentile(log_map, 1), numpy.nanmax(log_map)
    panels = [(support, "support M", (None, None)),
              (log_map, "Matern + trend GP MAP (log10)", (lo, hi)),
              (numpy.log10(mix_mean), "posterior-mixture mean (log10)", (lo, hi)),
              (log_std, "Laplace std of log10 I (dex)", (None, None)),
              (tc_f["trend"], "radial trend (log10 dex)", (None, None)),
              (tc_f["field_dct"], "Matern field (log10 dex)", (None, None))]
    fig, axs = plt.subplots(1, len(panels), figsize=(12*len(panels), 11))
    for ax, (im, title, (vmin, vmax)) in zip(axs, panels):
        c = ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out}/discs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.errorbar(spec["freq"], spec["power"], spec["power_err"], fmt="o")
    ax.axhline(1, color="k", ls="--")
    ax.set(xscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="<z^2> (prior: 1)",
           title="Posterior power / prior (Matern + radial trend; trend latents excluded)")
    fig.savefig(f"{out}/spectral-diagnostic.png")
    plt.close(fig)

    exps_f = scale_errors(set_hyper(exposures, **hypers_of(hyp_b), mean=numpy.zeros((resolved_wid, resolved_wid)),
                                    level_prior=lp_b), current["s"])
    model_f = set_array(ModelParams(p_final).inject(model_single))
    plot_comparison_detailed(model_f, ModelParams(set_array(p_final)), exps_f, quadrature=False, wf_size=wf_wid,
                             percentile=99, save=f"{out}/comparison")

    n_good = numpy.asarray(resp_f.n_good)
    od_final = optics_deltas(params_opt, p_final, exposures, wid, support_scale)     # stage-A (or L1/TSV) optics -> the final ones
    star_final = numpy.array([star_pixel(p_final, e, wid, support_scale) for e in exposures])
    checks |= dict(gradient_ratio_stage_C=float(noise["g_ratio"]), verify_max_abs_B=verify_max_abs, optics_stageA_to_final=od_final,
                   star_px_final_minus_support_star=(star_final - support_stars).tolist(),
                   star_only_vs_full_model_max_rel_diff=level_report.get("star_only_vs_full_model_max_rel_diff"),
                   final_level=float(l_f), level_prior_mu=float(level_mu[False]), level_prior_sd=float(level_sd),
                   final_level_minus_prior_mu_in_sd=float((float(l_f) - level_mu[False])/level_sd))
    print(f"[e2e] final level {float(l_f):.3f} vs prior mu {level_mu[False]:.3f} sd {level_sd:g} ({checks['final_level_minus_prior_mu_in_sd']:+.2f} sd); "
          f"optics stage A -> final: primary_opd rms change {od_final['primary_opd_rms_change_nm']:.1f} nm, star moved {numpy.round(od_final['star_shift_px'], 2).tolist()} px, "
          f"spectrum x{od_final['spectrum0_ratio']:.3f}; support star vs final star (px) {numpy.round(star_final - support_stars, 2).tolist()}")
    if numpy.max(numpy.abs(star_final - support_stars)) > 1.:
        print("WARNING: the final star is > 1 px from the support mask's star: the fixed support (built once) may be off at the crop edge; rebuild with "
              "the final tilts (rerun with E2E_SUPPORT_STAR=fitted after copying them) if this matters")
    if abs(checks["final_level_minus_prior_mu_in_sd"]) > 1.5:
        print("WARNING: the final level is > 1.5 prior sd from the data-estimated centre: the level prior was mis-centred (see estimate_level's caveats); "
              "a second pass with E2E_LEVEL_MU = the final level and a smaller E2E_LEVEL_SD is advised")
    e2e_report = dict(flags=dict(optics_source=optics_source, level_source=level_source, support_star=support_star, tilt_init=tilt_init),
                      stage_a=stage_a_info, level=level_report, checks=checks, out=out, n_outer=n_outer, min_outer=min_outer)
    with open(f"{out}/e2e-checks.json", "w") as f:
        json.dump(e2e_report, f, indent=2, default=str)
    summary = dict(
        support={k_: v_ for k_, v_ in support_info.items() if k_ != "coverage"} | dict(
            stars=support_stars.tolist(), orients=support_orients.tolist(), scale=support_scale),
        hyper=hyp_b, order=best_trend, trend_sigma=trend_sigma, trend_kinds=list(trend_kinds),
        trend_coef_dex=tc_f["coef"].tolist(), trend_coef_sd_dex=tc_f["coef_sd"].tolist(),
        dct_modes_02_20_z=[float(tc_f["z_dct"][0, 2]), float(tc_f["z_dct"][2, 0])],
        final_stage=current["stage"], outer_converged=outer_converged, outer_converged_reason=outer_reason,
        outer_history=outer_history, comparison_final_stage=comparison_final, data_dir=ddir,
        lbfgs={str(it): info for it, info in lbfgs_info.items()},   # result status, iterations, |sqrt(C) grad| start / end
        noise_scale=dict(zip(resp_f.keys, map(float, current["s"]))),
        chi2_per_pixel_raw=dict(zip(resp_f.keys, map(float, ev_f["chi2"]/n_good))),
        logZ=float(ev_f["logZ"]), gamma=float(ev_f["gamma"]), newton_decrement=float(ev_f["newton_decrement"]),
        final_grid=dict(quantiles=q_f, selected_outside_16_84=outside, edge_weight=post_f["edge_weight"],
                        contained=bool(post_f["edge_weight"] < edge_target), n_points=int(grid_f["logZ"].size),
                        axes_size={k: len(grid_f["axes"][k]) for k in grid_f["names"]},
                        unconverged=int((~grid_f["converged"].astype(bool)).sum())),
        comparison_at_stage_D=comparison, mixture_top20_weight=float(flat_w[top].sum()),
        gradient_ratio=float(noise["g_ratio"]), verify_max_abs=verify_max_abs, e2e=e2e_report,
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


# %%
# ---------------------------------------------------------------- F2: validation hook: compare with the L1/TSV-based run (comparison-l1tsv.json)
# The reference is the first of compare_dirs holding summary.json + gp-params.npy. Its optics / disc come from the L1/TSV fit, so the
# differences are the end-to-end fit's departures from it. Caveats printed in the file: the reference's level prior (L1/TSV, sd 1), noise scales
# and, for imlup-gp-radial-m12/, its (absent) support differ from this run's, so the evidence difference is indicative, not like-for-like.
with timed("F2: comparison with the L1/TSV-based run"):
    ref_dir_c = next((d_ for d_ in compare_dirs if os.path.exists(f"{d_}/summary.json") and os.path.exists(f"{d_}/gp-params.npy")), None)
    if ref_dir_c is None:
        print(f"validation hook: none of {compare_dirs} has a summary.json + gp-params.npy; skipped")
    else:
        try:
            ref_sum = json.load(open(f"{ref_dir_c}/summary.json"))
            ref_disc = numpy.load(f"{ref_dir_c}/gp-disc-log10.npy")
            ref_has_support = os.path.exists(f"{ref_dir_c}/support.npy")
            ref_sup = numpy.load(f"{ref_dir_c}/support.npy") if ref_has_support else numpy.ones_like(ref_disc)
            ref_run = dict(params=numpy.load(f"{ref_dir_c}/gp-params.npy", allow_pickle=True).item(), disc=ref_disc, support=ref_sup, logZ=float(ref_sum["logZ"]),
                           evidence=ref_sum["comparison_final_stage"]["trend"]["log_evidence"], hyper=ref_sum["hyper"])
            new_run = dict(params=p_final, disc=log_map, support=support, logZ=float(ev_f["logZ"]), evidence=comparison_final["trend"]["log_evidence"],
                           hyper=hyp_b, exps=exposures)
            cmp_ = compare_runs(new_run, ref_run, wid, support_scale)
            cmp_ |= dict(reference_dir=ref_dir_c, reference_has_support=ref_has_support, this_run=out,
                         note="logZ / evidence are Laplace values under each run's own level prior (this run mu, sd above; the reference's L1/TSV level, sd 1), "
                              "noise scales and support: indicative only. Optics: this run vs the reference (rms of the difference, per-exposure tilts); disc stats on the "
                              "common support of the log10 MAP maps, the sign is new - ref.")
            with open(f"{out}/comparison-l1tsv.json", "w") as f:
                json.dump(cmp_, f, indent=2, default=str)
            o_, d_ = cmp_["optics"], cmp_["disc"]
            print(f"[vs {ref_dir_c}] logZ {cmp_['logZ_new']:.1f} vs {cmp_['logZ_ref']:.1f} ({cmp_['logZ_new_minus_ref']:+.1f}); loop-model evidence difference "
                  f"{cmp_.get('evidence_new_minus_ref', float('nan')):+.1f}; hyper {hyp_b} vs {ref_sum['hyper']}; primary_opd rms difference "
                  f"{o_['primary_opd_rms_change_nm']:.1f} nm (this run's primary_opd rms {o_['primary_opd_rms_new_nm']:.1f} nm; star position differs by "
                  f"{numpy.round(o_['star_shift_px'], 2).tolist()} px; MAP disc on the common support ({d_['n_common']} px): rms {d_['rms_dex']:.3f} dex, median "
                  f"{d_['median_dex']:+.3f}, max {d_['max_abs_dex']:.2f}, corr {d_['corr_log10']:.3f}, flux ratio {d_['flux_ratio']:.3f}", flush=True)
        except Exception as e_c:     # never lose the run to the hook
            print(f"WARNING: the comparison with {ref_dir_c} failed: {e_c!r}")

