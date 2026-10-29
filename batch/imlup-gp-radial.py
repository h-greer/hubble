"""DRAFT batch/imlup-gp-radial.py: IM Lup with a RADIAL-TREND term added to the Matern DCT disc prior,
log10 I = level + H s_h b + idctn(sqrt(S) z), b ~ N(0, I_nb) (gp.py TrendGPPrior / TrendGPResolvedFit; a degenerate
basis-function kernel, Rasmussen & Williams sec. 2.7), against the plain single-Matern prior of imlup-gp.py. Motivation: the
best single-Matern MAP puts z ~ -5.4, -5.0 in the DCT modes (0, 2), (2, 0) (the disc's ~3 dex radial fall-off from the star at the
box centre, made from the lowest cosine modes at ~27 nats of prior cost). trend_sigma = 0 is exactly the Matern prior, so the
comparison is nested; trend_sigma is a fixed broad hyperparameter (1 dex), not a grid axis, so the trend adds no hyperprior
volume. Stages A-C are those of imlup-gp-bpl.py (same optics, response, noise scales); then
  D  single-Matern grids (orders 1, 2) vs Matern + trend grids (same (sigma, length) axes and bounds), the nesting check,
     the trend amplitudes and the (0, 2), (2, 0) latents before / after, the trend_sigma sensitivity, the spectral
     diagnostic (trend latents excluded), a small local 5 x 5 refinement of a close call; comparison-D.json
  E  outer loop (hybrid refit of optics + disc latents (z, b) at the chosen hyperparameters, response rebuilt, local grid)
  F  products: MAP disc, Laplace std, posterior mixture, the trend as its own log10 image (gp-trend-log10.npy and its std),
     the DCT field, spectral diagnostic, summary.json (incl. L-BFGS status)
"resolved" holds the flat latents (z_dct, b), length resolved_wid^2 + n_b (as PolarDCTResolvedFit's). Needs the radial-trend
patch of gp.py (gp.patch). The refit is imlup-gp-bpl.py's hybrid (short first-order stage, then one L-BFGS call); the
prior-independent response.npz / verify-*.npz are read from imlup-gp/ when present. Resumable and fingerprinted as
imlup-gp-bpl.py."""
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
from contextlib import contextmanager

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache (directory from JAX_COMPILATION_CACHE_DIR, set in the .sh); cache every compile
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# Optimisation imports
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
from gp import _save_npz   # the trend prior comes with `import *`

out = "imlup-gp-radial"
ref_dir = "imlup-adi"
shared_dir = "imlup-gp"     # the prior-independent response.npz / verify-*.npz are read from here (no recompute)
os.makedirs(out, exist_ok=True)


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
orders = [1., 2.]           # imlup-gp.py's best was order 1; both with and without the trend
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
level_sd = 1.               # dex, Normal prior on resolved_level about the reference disc's level
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
n_outer = 4
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
# ---------------------------------------------------------------- A: exposures, model, reference
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


# n_share: all four exposures share the target-keyed disc, so each carries 1/4 of its prior
exposures = with_n_share([
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits', GPDiscFit(spectrum_basis, "F160W", wid=resolved_wid, trend_sigma=1., kinds=trend_kinds),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for root in roots
])
check_n_share(exposures)
key = exposures[0].fit.get_key(exposures[0], "resolved")

# imlup-adi's initial values, then its stage-1 (intermediate) and stage-2 (params_amp) results
params = {k: {} for k in ("spectrum", "primary_opd", "primary_low", "primary_tilt", "cold_mask_opd", "cold_mask_tilt",
                          "cold_mask_shift", "cold_mask_rot", "cold_mask_shear", "cold_mask_scale",
                          "cold_mask_distortion", "defocus1", "defocus2", "defocus3", "primary_rot", "primary_shear",
                          "bias")}
params |= {
    "occulter_radius": 0.8, "occulter_ellipticity": np.zeros(2), "fnumber": 45.7, "anisotropy": 1. + 1e-5,
    "outer_radius": 1.2*0.9768, "secondary_radius": 0.365*1.2, "spider_width": 0.072*1.2, "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2, "primary_outer": 1.2, "primary_pad": 0.065*1.2, "secondary_shift": np.zeros(2),
    "primary_distortion": np.zeros((2, 7)), "distortion": np.zeros((2, 7)),
}
for exp in exposures:
    k = lambda name: exp.fit.get_key(exp, name)
    params["spectrum"][k("spectrum")] = np.zeros(npoly).at[0].set((np.nansum(exp.data)/nwavels)*100)
    params["primary_tilt"][k("primary_tilt")] = np.array([-0.75, -0.05])*0.075
    params["cold_mask_tilt"][k("cold_mask_tilt")] = hole_tilt(exp)
    params["primary_opd"][k("primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_low"][k("primary_low")] = np.zeros(n_zernikes)
    params["cold_mask_opd"][k("cold_mask_opd")] = np.zeros(16).at[0].set(120.)
    params["cold_mask_shift"][k("cold_mask_shift")] = np.array([13., 10.])
    params["cold_mask_rot"][k("cold_mask_rot")] = 3.2
    params["primary_rot"][k("primary_rot")] = -0.6
    params["cold_mask_scale"][k("cold_mask_scale")] = np.asarray([1., 1.])
    params["cold_mask_shear"][k("cold_mask_shear")] = np.asarray([0.06, -0.06])
    params["primary_shear"][k("primary_shear")] = np.asarray([0., 0.])
    params["bias"][k("bias")] = 0.
    for name in ("defocus1", "defocus2", "defocus3"):
        params[name][k(name)] = 0.
    params["cold_mask_distortion"][k("cold_mask_distortion")] = np.zeros((2, 7))

for f in ("intermediate-params.npy", "params_amp.npy"):
    path = f"{ref_dir}/{f}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run imlup-adi.py first (its fit is the reference optics and disc)")
    params |= numpy.load(path, allow_pickle=True).item()

d_ref = 10**np.asarray(params.pop("resolved")[key])           # the L1/TSV disc
mean_t = log10_template(d_ref, template_floor)
level_mu = {t: reference_level(d_ref, mean_t if t else None) for t in templates}
print(f"reference level (dex): flat mean {level_mu[False]:.3f}")

params["resolved"] = {key: np.zeros(resolved_wid**2 + n_b)}      # the flat latents (z_dct, b)
params["resolved_level"] = {key: np.asarray(level_mu[False])}
ref_params = params
check_gp_setup(ref_params)
# the reference must be keyed as GPDiscFit keys its params (a KeyError deep inside a jit is much harder to read)
for exp in exposures:
    for name, v in ref_params.items():
        if isinstance(v, dict):
            assert exp.fit.get_key(exp, name) in v, f"params[{name!r}] has no key {exp.fit.get_key(exp, name)!r}"
model_single = set_array(NICMOSModel(exposures, ref_params, optics, detector))
n = resolved_wid**2
mean_of = lambda template: mean_t if template else None
lp_of = lambda template: (level_mu[template], level_sd)

# %%
# ---------------------------------------------------------------- B: linear response at the reference optics
with timed("B: response"):
    # the linear response at the reference optics does not depend on the prior: reuse imlup-gp's, else build it here
    rpath0 = next((p for p in (f"{shared_dir}/response.npz", f"{out}/response.npz") if os.path.exists(p)),
                  f"{out}/response.npz")
    if os.path.exists(rpath0):
        resp = LinearResponse.load(rpath0)
    else:
        resp = build_linear_response(model_single, exposures, batch_size=response_batch)
        resp.save(rpath0)
    resp_fp = response_fingerprint(resp)
print(f"response ready ({rpath0}, fingerprint {resp_fp})")

with timed("B: verify"):
    # the fingerprint names the response's content, so imlup-gp's verification of the same response is reusable
    vpath = next((p for p in (f"{shared_dir}/verify-{resp_fp}.npz", f"{out}/verify-{resp_fp}.npz") if os.path.exists(p)),
                 f"{out}/verify-{resp_fp}.npz")
    if not os.path.exists(vpath):
        # the reference disc, no disc, and the reference disc perturbed by 0.3 dex rms: the full model's NLL must match.
        # Compared as changes from the no-disc NLL (lognorm, ~-2e4 per exposure, would dilute a relative test)
        ds = np.stack([np.zeros(n), d_ref.ravel(), (d_ref*10**(0.3*jr.normal(jr.PRNGKey(0), d_ref.shape))).ravel()])
        full, sur = (numpy.asarray(x) for x in verify_linear_response(resp, model_single, exposures, ds))
        dfull, dsur = full[:, 1:] - full[:, :1], sur[:, 1:] - sur[:, :1]
        _save_npz(vpath, full=full, surrogate=sur, abs_diff=numpy.abs(full - sur),
                  rel_change=numpy.abs(dfull - dsur)/numpy.maximum(numpy.abs(dfull), 1.))
    v = numpy.load(vpath)
    verify_max_abs = float(v["abs_diff"].max())      # captured now: `v` is a short name, do not rely on it later
    print(f"linear response vs full model: max |NLL difference| {v['abs_diff'].max():.2e}, max relative difference "
          f"of NLL changes {v['rel_change'].max():.2e}")
    if v["rel_change"].max() > 1e-6 or v["abs_diff"].max() > 1e-3:
        raise RuntimeError("the surrogate does not reproduce the full model: is the image linear in the disc at fixed "
                           f"optics? (delete {vpath} after fixing)")

# %%
# ---------------------------------------------------------------- C: first MAP, gradient check, noise scales
prior0 = gp_prior(resolved_wid, 1., 1.6, 2., None, lp_of(False))   # single kernel: the noise scales are the DCT run's

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
        z1, l1 = prior0.params(theta1)
        loss_1, g_1 = full_grad(np.concatenate([np.ravel(z1), np.zeros(n_b)]), l1)
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
sg = lambda order: (lambda sigma, length: gp_prior(resolved_wid, sigma, length, order, None, lp_of(False)))
tf = lambda order, s_h=trend_sigma: trend_prior_factory(resolved_wid, order, s_h, None, lp_of(False), basis=trend_basis)
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
p_nest = gp_trend_prior(resolved_wid, b1["sigma"], b1["length"], best_single, 0., None, lp_of(False))
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

# %%
# ---------------------------------------------------------------- E: outer iteration (Matern + radial trend)
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
check_gp_setup(ref_params, things_gp)


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
    C_fo = dict(C) | {"resolved": jax.tree.map(np.ones_like, C["resolved"])}      # first order: C = 1 for z, as before
    things_fo = dict(things) | {"primary_opd": adam(fo_lr_opd, 0), "resolved": adam(fo_lr_disc, 0)}
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


mean_b, lp_b = None, lp_of(False)
make_b = lambda: trend_prior_factory(resolved_wid, best_trend, trend_sigma, mean_b, lp_b, basis=trend_basis)
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
    print(f"trend surrogate vs full model at the stage-D MAP: full loss {float(loss_c):.4f}, surrogate "
          f"{float(ev_c['nll'].sum() + ev_c['prior']):.4f}; |grad| {float(np.linalg.norm(g_c['resolved'][key])):.2e}")
except Exception as e:   # a diagnostic: never lose the stage to it
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
                              n_points=int(grid_new["logZ"].size), unconverged=int((~grid_new["converged"].astype(bool)).sum())))
    hyp_b = new
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=np.asarray(grid_new["thetas"][idx_new]), idx=idx_new, stage=f"E{it}")
    if not moved or loose:
        outer_converged, outer_reason = True, ("best cell did not move" if not moved else
                                               f"moved <= 1 step and {-dlogZ_old:.2f} < {converge_dlogz} nats (refit cell nearly as good)")
        print(f"outer {it}: converged ({outer_reason})")
        break
if not outer_converged:
    print(f"WARNING: the hyperparameters still moved at outer iteration {n_outer}: the optics were refitted at the "
          "previous hyperparameters; raise n_outer")
if outer_history and outer_history[-1]["edge_weight"] >= edge_target:
    print(f"WARNING: the final local grid still has edge weight {outer_history[-1]['edge_weight']:.3f} >= {edge_target} "
          f"(axes {outer_history[-1]['axes_size']}): its quantiles are truncated (growth capped at {outer_max_growth})")
for it, info in lbfgs_info.items():
    if not info.get("successful", False) or info.get("reached_max_steps", False):
        print(f"WARNING: outer {it} L-BFGS did not report success ({info.get('result')}, {info.get('num_steps')} its, "
              f"|sqrt(C) grad| end {info.get('grad_norm_u_end')}): the optics may be under-converged")

# %%
# ---------------------------------------------------------------- G: like-for-like re-evaluation of the plain Matern
# The first run established the +29 nats only at the stage-D optics. Re-evaluate the plain Matern (orders 1 and 2, the same
# (sigma, length) bounds, level prior and beta, trend_sigma = 0 nesting) on the FINAL response, against the trend's final grid.
with timed("G: plain Matern on the final response"):
    fin_resp, fin_beta = current["resp"], current["beta"]
    fin_fp = fingerprint(current["resp_fp"], fin_beta, lp_b, n_iter, code_fp)
    # trend: fine 7 x 7 grid at ratio 2^(1/4) about the loop's best (the coarse loop grid is its warm start and fallback)
    fin_grid = gp_evidence_grid_nd(fin_resp, fine_axes(hyp_b), make_b(), beta=fin_beta,
                                   outdir=f"{out}/grid", tag="final_fine_trend_" + fin_fp, abbrev=ABBREV, bounds=hyper_bounds,
                                   n_iter=n_iter, theta0=current["theta"], max_growth=fine_max_growth,
                                   edge_dlogz=grid_edge_dlogz, recheck_top=recheck_top)
    fin_post = gp_hyper_posterior(fin_grid)
    hyp_b = {k: float(fin_grid["axes"][k][i]) for k, i in zip(fin_grid["names"], fin_grid["best"])}   # the fine best
    current = dict(current, grid=fin_grid, theta=np.asarray(fin_grid["thetas"][fin_grid["best"]]), idx=fin_grid["best"],
                   stage=current["stage"] + "+fine")          # stage F's products / mixture use the fine grid
    fin_trend = dict(order=best_trend, log_evidence=fin_post["log_evidence_bounds"], best=gp_best(fin_grid),
                     edge_weight=fin_post["edge_weight"], n_points=int(fin_grid["logZ"].size),
                     axes_size={k: len(fin_grid["axes"][k]) for k in fin_grid["names"]},
                     unconverged=int((~fin_grid["converged"].astype(bool)).sum()),
                     quantiles={k: list(map(float, v_)) for k, v_ in fin_post["quantiles"].items()})
    print(f"[final trend o{best_trend:g}] log_evidence_bounds {fin_post['log_evidence_bounds']:.2f} at {hyp_b}; edge "
          f"{fin_post['edge_weight']:.3f}; {fin_trend['n_points']} points")
    fin_plain = {}
    for o_p in orders:
        gs_o = singles[o_p]
        # coarse pass from the stage-D best (the plain optimum may have moved), then the same fine grid as the trend's
        gc_p = gp_evidence_grid_nd(fin_resp, local_axes(gp_best(gs_o), outer_steps, n=outer_half_width), sg(o_p), beta=fin_beta,
                                   outdir=f"{out}/grid", tag=f"final_plain_o{o_p:g}_" + fin_fp, abbrev=ABBREV,
                                   bounds=hyper_bounds, n_iter=n_iter, theta0=np.asarray(gs_o["thetas"][gs_o["best"]]),
                                   max_growth=outer_max_growth, edge_dlogz=grid_edge_dlogz, recheck_top=recheck_top)
        gp_p = gp_evidence_grid_nd(fin_resp, fine_axes(gp_best(gc_p)), sg(o_p), beta=fin_beta,
                                   outdir=f"{out}/grid", tag=f"final_fine_plain_o{o_p:g}_" + fin_fp, abbrev=ABBREV,
                                   bounds=hyper_bounds, n_iter=n_iter, theta0=np.asarray(gc_p["thetas"][gc_p["best"]]),
                                   max_growth=fine_max_growth, edge_dlogz=grid_edge_dlogz, recheck_top=recheck_top)
        pp = gp_hyper_posterior(gp_p)
        best_p = gp_best(gp_p)
        th_p = np.asarray(gp_p["thetas"][gp_p["best"]])
        # nesting at this point: the trend prior with trend_sigma = 0 (b = 0) must reproduce the plain Matern's log Z
        ev_p0 = gp_log_evidence(fin_resp, sg(o_p)(**best_p), th_p, fin_beta)
        ev_p1 = gp_log_evidence(fin_resp, gp_trend_prior(resolved_wid, best_p["sigma"], best_p["length"], o_p, 0., None,
                                                         lp_b), embed_trend_theta(th_p, n_b), fin_beta)
        fin_plain[f"o{o_p:g}"] = dict(log_evidence=pp["log_evidence_bounds"], best=best_p, edge_weight=pp["edge_weight"],
                                      n_points=int(gp_p["logZ"].size), unconverged=int((~gp_p["converged"].astype(bool)).sum()),
                                      axes_size={k: len(gp_p["axes"][k]) for k in gp_p["names"]},
                                      max_converged_logZ=float(converged_max(gp_p)),
                                      coarse_log_evidence=gp_hyper_posterior(gc_p)["log_evidence_bounds"],
                                      quantiles={k: list(map(float, v_)) for k, v_ in pp["quantiles"].items()},
                                      nesting_logZ_abs_diff=float(np.abs(ev_p0["logZ"] - ev_p1["logZ"])))
        print(f"[final plain o{o_p:g}] log_evidence_bounds {pp['log_evidence_bounds']:.2f} (coarse "
              f"{fin_plain[f'o{o_p:g}']['coarse_log_evidence']:.2f}) at {best_p}; edge {pp['edge_weight']:.3f}; "
              f"{fin_plain[f'o{o_p:g}']['n_points']} points; nesting |dlogZ| {fin_plain[f'o{o_p:g}']['nesting_logZ_abs_diff']:.1e}")
    best_plain_key = max(fin_plain, key=lambda k_: fin_plain[k_]["log_evidence"])
    dlogE_final = float(fin_trend["log_evidence"] - fin_plain[best_plain_key]["log_evidence"])
    dlogE_final_same_order = float(fin_trend["log_evidence"] - fin_plain[f"o{best_trend:g}"]["log_evidence"])
    fin_valid = bool(fin_trend["edge_weight"] < edge_target and all(d_["edge_weight"] < edge_target for d_ in fin_plain.values()))
    comparison_final = dict(
        dlogE_trend_minus_best_plain=dlogE_final, best_plain=best_plain_key, dlogE_same_order=dlogE_final_same_order,
        grid_quality_valid=fin_valid, edge_target=edge_target, trend=fin_trend, plain=fin_plain,
        dlogE_stage_D=dlogE, note="fine grids (sigma 9 values at 2^(1/16), length 7 at 2^(1/4)) for both; same final response, same beta, same hyper_bounds and level prior; the trend is the "
        "outer loop's order only; the optics were refitted WITH the trend, so a plain Matern refit would shift too")
    with open(f"{out}/comparison-final.json", "w") as f:
        json.dump(comparison_final, f, indent=2, default=float)
    print(f"final-stage dlogE (trend - plain {best_plain_key}) {dlogE_final:+.2f} nats (same order {dlogE_final_same_order:+.2f}; "
          f"stage D {dlogE:+.2f}); grid quality valid (edge < {edge_target}): {fin_valid}")

# %%
# ---------------------------------------------------------------- F: products
with timed("F: products"):
    resp_f, beta_f, grid_f = current["resp"], current["beta"], current["grid"]
    prior_f = make_b()(**hyp_b)
    theta_f = current["theta"]
    ev_f = to_host(gp_log_evidence(resp_f, prior_f, theta_f, beta_f))
    lap = gp_laplace(resp_f, prior_f, theta_f, beta_f)
    log_map = numpy.asarray(prior_f.log10_image(theta_f))
    log_std = numpy.asarray(gp_log10_std(prior_f, lap))
    z_f, l_f = prior_f.params(theta_f)
    tc_f = to_host(trend_components(prior_f, theta_f, lap))     # coef (dex), coef_sd, trend / trend_std maps, DCT field

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
    mix_mean, mix_std = numpy.asarray(mix.mean(0)), numpy.asarray(mix.std(0))
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

    lo, hi = numpy.percentile(log_map, 1), log_map.max()
    panels = [(numpy.log10(numpy.asarray(d_ref)), "L1/TSV reference (log10)", (lo, hi)),
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
    summary = dict(
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
        gradient_ratio=float(noise["g_ratio"]), verify_max_abs=verify_max_abs,
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

