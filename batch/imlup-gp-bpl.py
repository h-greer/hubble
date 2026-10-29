"""DRAFT batch/imlup-gp-bpl.py: IM Lup with a BROKEN POWER-LAW spectrum disc prior (one spectrum, hyperparameters sigma,
length_b = 1 / k_b, gamma1, gamma2 = gamma1 + dgamma; gp.py bpl_*) against the single-kernel Matern DCT prior of imlup-gp.py.
gamma1 = 0 is exactly the Matern prior of order gamma2 / 2, so the comparison is a nested, like-for-like one on the same
hyperpriors. Stages A-C are those of imlup-gp-dct2.py (same optics, response, noise scales); then
  D  single-kernel Matern grids (order 1, 2; length <= 200 px, the BPL's break-scale bound) vs a 4-D BPL grid
     (sigma, length_b, gamma1, dgamma); the spectral diagnostic of both at the reference response
  E  outer loop (hybrid refit of optics + disc at the chosen hyperparameters, response rebuilt, local 4-D grid)
  F  products: MAP disc, Laplace std, posterior mixture, spectral diagnostic, summary.json (incl. L-BFGS status)
Every constraint of the BPL hyperprior is an axis bound (no mask), so gp_hyper_posterior's quantiles are the hyperprior's.
The refit is dct2's hybrid (short first-order stage, then one L-BFGS call with the disc whitened); the L-BFGS result status,
iteration count and final preconditioned gradient norm are saved (outer*-lbfgs.json, summary.json). The prior-independent
response.npz / verify-*.npz are read from imlup-gp/ when present. Resumable and fingerprinted as imlup-gp-dct2.py."""
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax.random as jr
import jax
import numpy
import os
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
from gp import _save_npz   # the BPL prior comes with `import *`

out = "imlup-gp-bpl"
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

# Hyperparameter grids (gp_evidence_grid_nd extends an axis by one step while the optimum is on an edge, within its bounds).
# sigma: rms of log10 I in dex; lengths in 0.0864" px (lambda/D at F160W = 0.1375" = 1.6 px).
# The break scale / Matern length axis is log-spaced by 2 and ANCHORED so that its top cell reaches the 200 px bound:
# values 200 / sqrt(2) / 2^i, cells half a step either side (the box is 60 px, its lowest DCT mode has 1 / k_min ~ 19 px:
# beyond ~100 px the spectrum is the pure power law k^-gamma2 and length_b, gamma1 are not separately identified).
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [200/2**0.5/2**i for i in range(5, -1, -1)]          # 4.42 .. 141.4 (6 values; growth reaches 0.1 below)
bpl0 = dict(sigma=sigmas0, length_b=lengths0, gamma1=[0., 1., 2., 3.], dgamma=[1., 2., 3., 4., 5., 6.])   # 4 x 6 x 4 x 6 = 576, then growth
# Hyperpriors (log-uniform sigma, length_b; uniform gamma1, dgamma): gamma2 = gamma1 + dgamma > gamma1 is a steepening
# break (the box in (gamma1, dgamma) has unit Jacobian from (gamma1, gamma2)); gamma1 = 0 face, dgamma = 2 order is the Matern
# prior (the gamma1 = 0 cell is half-width, clipped at the bound 0: its mass is half a step's). gamma1 <= 3.5:
# for gamma1 > 2 the normalisation puts > 85 % of sigma^2 in the 3 lowest modes (test_bpl.py), so higher values are an almost
# pure-large-scale field; raise it (and extend bpl0) deliberately. dgamma lower bound 0.5 keeps the break from vanishing;
# dgamma starts at 1..6 so that max_growth = 3 can reach 8 (top cell edge 8.5 = the bound; from 4 it would stop at 7).
bpl_bounds = dict(sigma=(0.03, 8.), length_b=(0.1, 200.), gamma1=(0., 3.5), dgamma=(0.5, 8.5))
single_bounds = dict(sigma=(0.03, 8.), length=(0.1, 200.))   # the Matern comparator: the SAME sigma and break-scale bounds
orders = [1., 2.]           # the single-kernel DCT comparison (order 1 was imlup-gp.py's best); BPL contains both
refine_below = 3.           # nats: |dlogE| below this is re-evaluated on finer local grids before deciding
recheck_top = 3             # cold restarts of the best points, the same for the single-kernel and BPL grids
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
n_outer = 2

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


class GPDiscFit(BPLGPResolvedFit):
    """BPLGPResolvedFit (broken power-law prior on log10 of the disc) with imlup-adi's ResolvedFit keys."""
    PARAM_KEYS = BPLGPResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                             "primary_opd": "global", "cold_mask_opd": "exposure",
                                             "defocus3": "exposure"}


# n_share: all four exposures share the target-keyed disc, so each carries 1/4 of its prior
exposures = with_n_share([
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits', GPDiscFit(spectrum_basis, "F160W", wid=resolved_wid),
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

params["resolved"] = {key: np.zeros((resolved_wid, resolved_wid))}
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
        # gamma1 = 0, gamma2 = 4 is exactly the Matern order 2 of prior0 (gp.py bpl_from_matern)
        exps0 = set_hyper(exposures, sigma=1., length_b=1.6, gamma1=0., gamma2=4., level_prior=lp_of(False))
        batch0 = make_batch(exps0, model_single)

        def full_grad(z, level):
            sub = set_array({"resolved": {key: z}, "resolved_level": {key: level}})
            return _loss_grad(sub, jax.tree.map(np.ones_like, sub), batch0, model_single)

        loss_0, g_0 = full_grad(np.zeros((resolved_wid, resolved_wid)), np.asarray(level_mu[False]))
        z1, l1 = prior0.params(theta1)
        loss_1, g_1 = full_grad(z1, l1)
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
# ---------------------------------------------------------------- D: evidence grids (single Matern vs broken power law)
# Matched hyperpriors: sigma log-uniform (0.03, 8) for both; Matern length / BPL length_b log-uniform (0.1, 200); the BPL's
# gamma1 (0, 3.5) and dgamma (0.5, 8.5) are its extra, uniform hyperparameters, charged by their prior volume in the log
# evidence (gp_hyper_posterior's log_evidence_bounds). The Matern is the gamma1 = 0 face: gamma2 = 2 order.
base_fp = fingerprint(resp_fp, beta, level_mu[False], level_sd, n_iter, code_fp)
singles, singles_post = {}, {}
with timed("D: single-kernel grids"):
    for order in orders:
        tag = f"o{order:g}_t0_" + base_fp
        g1 = gp_evidence_grid_nd(resp, {"sigma": sigmas0, "length": lengths0},
                                 lambda sigma, length, order=order: gp_prior(resolved_wid, sigma, length, order, None,
                                                                             lp_of(False)),
                                 beta=beta, outdir=f"{out}/grid", tag=tag, abbrev={"sigma": "s", "length": "l"},
                                 bounds=single_bounds, n_iter=n_iter, theta0=noise["theta_s"], max_growth=6,
                                 recheck_top=recheck_top)
        p1 = gp_hyper_posterior(g1)           # on the grid's own bounds = single_bounds
        singles[order], singles_post[order] = g1, p1
        bs = gp_best(g1)
        print(f"[single o{order:g}] best converged logZ {converged_max(g1):.2f} at {bs}; log_evidence_bounds "
              f"{p1['log_evidence_bounds']:.2f}; edge {p1['edge_weight']:.3f}")

with timed("D: BPL grid"):
    tag = "bpl_" + base_fp
    gb = gp_evidence_grid_nd(resp, bpl0, bpl_prior_factory(resolved_wid, None, lp_of(False)), beta=beta,
                             outdir=f"{out}/grid", tag=tag, abbrev=BPL_ABBREV, bounds=bpl_bounds, scales=BPL_SCALES,
                             n_iter=n_iter, theta0=noise["theta_s"], max_growth=3, recheck_top=recheck_top)
    # every constraint is an axis bound, so these quantiles / weights / evidence are all on the stated hyperprior
    pb = gp_hyper_posterior(gb)
    idx_b = gb["best"]
    best_b = {k: float(gb["axes"][k][i]) for k, i in zip(gb["names"], idx_b)}
    _save_npz(f"{out}/grid-bpl.npz", logZ=gb["logZ"], converged=gb["converged"], gamma=gb["gamma"], best=numpy.array(idx_b),
              weights=pb["weights"], log_evidence_bounds=pb["log_evidence_bounds"], edge_weight=pb["edge_weight"],
              **{f"axis_{k}": v for k, v in gb["axes"].items()})
    print(f"[bpl] best converged logZ {float(gb['logZ'][idx_b]):.2f} at {best_b} (gamma2 "
          f"{best_b['gamma1'] + best_b['dgamma']:.3g}); log_evidence_bounds {pb['log_evidence_bounds']:.2f}; edge weight "
          f"{pb['edge_weight']:.3f}; unconverged {int((~gb['converged'].astype(bool)).sum())}")
    print("[bpl] posterior 16/50/84 %:", {k: numpy.round(v, 3).tolist() for k, v in pb["quantiles"].items()})
    for ax_name, ax_val in best_b.items():   # the selected (joint-mode) cell vs the marginal quantiles: they can differ
        q = pb["quantiles"][ax_name]
        if not (q[0] <= ax_val <= q[2]):
            print(f"NOTE: selected {ax_name} = {ax_val:.4g} is outside its marginal 16-84 % interval {numpy.round(q, 4)}")
    # flags (information; the decision is the evidence): gamma1 on the Matern face, break unidentified
    flags = dict(gamma1_zero=bool(best_b["gamma1"] == 0.), length_b_ge_100px=bool(best_b["length_b"] >= 100.),
                 sigma_at_floor=bool(idx_b[0] == 0 and gb["axes"]["sigma"][0] <= 2*bpl_bounds["sigma"][0]))

# nesting check on the data's prior: the BPL spectrum with the Matern's mapped hyperparameters IS the Matern prior
best_single = max(singles_post, key=lambda o: singles_post[o]["log_evidence_bounds"])
g1 = singles[best_single]
b1 = gp_best(g1)
prior_1 = gp_prior(resolved_wid, b1["sigma"], b1["length"], best_single, None, lp_of(False))
lb_m, g1_m, g2_m = bpl_from_matern(b1["length"], best_single)
nest = float(np.abs(prior_1.amp - bpl_gp_prior(resolved_wid, b1["sigma"], lb_m, g1_m, g2_m, None, lp_of(False)).amp).max())
print(f"nesting: max |amp_matern - amp_bpl(gamma1 = 0, gamma2 = 2 order)| = {nest:.2e}")
assert nest < 1e-10

dlogZ_max = float(gb["logZ"][idx_b] - max(converged_max(g) for g in singles.values()))
dlogE_coarse = float(pb["log_evidence_bounds"] - singles_post[best_single]["log_evidence_bounds"])
unconv = lambda g: float(1 - numpy.asarray(g["converged"]).astype(bool).mean())
diag = dict(bpl=dict(unconverged_fraction=unconv(gb), edge_weight=pb["edge_weight"], n_points=int(gb["logZ"].size)),
            **{f"single_o{o:g}": dict(unconverged_fraction=unconv(g), edge_weight=singles_post[o]["edge_weight"],
                                      n_points=int(g["logZ"].size)) for o, g in singles.items()})
print("grid quality (unconverged fraction, edge weight):", {k: (round(d_["unconverged_fraction"], 3),
                                                              round(d_["edge_weight"], 3)) for k, d_ in diag.items()})

# Refinement: the coarse grids (ratio 2 / step 1) are crude quadrature, so a small evidence difference is re-evaluated on
# finer LOCAL grids about each model's best cell (5 values per axis: sigma, length ratio sqrt 2; gamma1, dgamma step 0.5;
# 625 BPL + 25 Matern points), under the same bounds. Valid only if the posterior is contained (edge_weight small), else the
# coarse values are kept. Both values are recorded; the refined one decides when valid.
dlogE, refined = dlogE_coarse, None
if abs(dlogE_coarse) < refine_below:
    with timed("D: refinement"):
        # n=2 BPL grid: 5^4; the Matern about its best cell: sqrt2 steps
        gbr = gp_evidence_grid_nd(resp, local_axes(best_b, dict(sigma=2**0.5, length_b=2**0.5, gamma1=0.5, dgamma=0.5),
                                                   scales=BPL_SCALES, n=2),
                                  bpl_prior_factory(resolved_wid, None, lp_of(False)), beta=beta, outdir=f"{out}/grid",
                                  tag="bplfine_" + base_fp, abbrev=BPL_ABBREV, bounds=bpl_bounds, scales=BPL_SCALES,
                                  n_iter=n_iter, theta0=np.asarray(gb["thetas"][idx_b]), max_growth=0, recheck_top=recheck_top)
        pbr = gp_hyper_posterior(gbr)
        g1r = gp_evidence_grid_nd(resp, local_axes(b1, dict(sigma=2**0.5, length=2**0.5), n=2), 
                                  lambda sigma, length, order=best_single: gp_prior(resolved_wid, sigma, length, order, None,
                                                                                    lp_of(False)),
                                  beta=beta, outdir=f"{out}/grid", tag=f"o{best_single:g}fine_" + base_fp,
                                  abbrev={"sigma": "s", "length": "l"}, bounds=single_bounds, n_iter=n_iter,
                                  theta0=np.asarray(g1["thetas"][g1["best"]]), max_growth=0, recheck_top=recheck_top)
        p1r = gp_hyper_posterior(g1r)
        valid = pbr["edge_weight"] < 0.05 and p1r["edge_weight"] < 0.05
        refined = dict(bpl_log_evidence=pbr["log_evidence_bounds"], single_log_evidence=p1r["log_evidence_bounds"],
                       dlogE=float(pbr["log_evidence_bounds"] - p1r["log_evidence_bounds"]),
                       bpl_edge_weight=pbr["edge_weight"], single_edge_weight=p1r["edge_weight"], valid=bool(valid),
                       bpl_unconverged=unconv(gbr), single_unconverged=unconv(g1r))
        if valid:
            dlogE = refined["dlogE"]
        print(f"refinement: coarse dlogE {dlogE_coarse:+.2f}; refined {refined['dlogE']:+.2f} (edge weights BPL "
              f"{pbr['edge_weight']:.3f}, single {p1r['edge_weight']:.3f}; used: {'refined' if valid else 'coarse'})")
use_bpl = dlogE > 0
print(f"BPL vs single-kernel o{best_single:g}: max converged logZ difference {dlogZ_max:+.2f} nats (>= 0 by nesting up to "
      f"the grid), difference of log evidence under the matched hyperpriors {dlogE:+.2f} nats (coarse {dlogE_coarse:+.2f}; "
      "gamma1 and dgamma are charged by their hyperprior volume; the Laplace log Z is biased up by an amount growing "
      f"with sigma, gp.py: validate by injection). Use the BPL for E/F: {use_bpl}")

# spectral diagnostic of both at the reference response: does the BPL spectrum fix the mismatch?
prior_b = bpl_prior_factory(resolved_wid, None, lp_of(False))(**best_b)
theta_b = np.asarray(gb["thetas"][idx_b])
theta_1 = np.asarray(g1["thetas"][g1["best"]])
spec_1 = gp_spectral_diagnostic(prior_1, theta_1, gp_laplace(resp, prior_1, theta_1, beta))
spec_b = gp_spectral_diagnostic(prior_b, theta_b, gp_laplace(resp, prior_b, theta_b, beta))
numpy.savez(f"{out}/spectral-diagnostic-D.npz", **{f"single_{k}": v for k, v in spec_1.items()},
            **{f"bpl_{k}": v for k, v in spec_b.items()})
for name, sd_ in (("single", spec_1), ("bpl", spec_b)):
    print(f"spectral diagnostic ({name}): <z^2> per bin {numpy.round(sd_['power'][:12], 2)} ... max |log10| deviation "
          f"{numpy.max(numpy.abs(numpy.log10(sd_['power']))):.2f}")
fig, ax = plt.subplots(figsize=(11, 9))
for name, sd_ in (("single Matern", spec_1), ("broken power law", spec_b)):
    ax.errorbar(sd_["freq"], sd_["power"], sd_["power_err"], fmt="o-", label=name)
ax.axhline(1, color="k", ls="--")
ax.set(xscale="log", yscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="<z^2> (prior: 1)")
ax.legend()
fig.savefig(f"{out}/spectral-diagnostic-D.png")
plt.close(fig)

comparison = dict(
    best_single_order=best_single, use_bpl=use_bpl,
    max_converged_logZ_difference_bpl_minus_single=dlogZ_max, log_evidence_difference_matched_hyperpriors=dlogE,
    log_evidence_difference_coarse=dlogE_coarse, refinement=refined, grid_quality=diag, refine_below_nats=refine_below,
    bpl_best=best_b, bpl_best_gamma2=best_b["gamma1"] + best_b["dgamma"], bpl_flags=flags,
    bpl_log_evidence=pb["log_evidence_bounds"],
    bpl_quantiles={k: list(map(float, v)) for k, v in pb["quantiles"].items()}, bpl_edge_weight=pb["edge_weight"],
    single_log_evidence={f"o{o:g}": q["log_evidence_bounds"] for o, q in singles_post.items()},
    single_best={f"o{o:g}": gp_best(g) for o, g in singles.items()}, nesting_amp_max_abs_diff=nest,
    hyperpriors=dict(single=single_bounds, bpl=bpl_bounds),   # log-uniform in sigma, length(_b); uniform in gamma1, dgamma
    spectral_max_abs_log10_single=float(numpy.max(numpy.abs(numpy.log10(spec_1["power"])))),
    spectral_max_abs_log10_bpl=float(numpy.max(numpy.abs(numpy.log10(spec_b["power"])))))
with open(f"{out}/comparison-D.json", "w") as f:
    json.dump(comparison, f, indent=2, default=float)
if not use_bpl:
    print("the broken power law does not beat the single Matern under the matched hyperpriors: stages E/F are not run; "
          "use imlup-gp.py's single-kernel pipeline. See comparison-D.json")
    sys.exit(0)

# %%
# ---------------------------------------------------------------- E: outer iteration (BPL)
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


def precond_grad_norm(x, C, model_b, exps_b):
    """|sqrt(C) grad loss| at x: the gradient in L-BFGS's own coordinates u = (x - x0) / sqrt(C) (nan if it fails)."""
    try:
        _, gr = _loss_grad(x, jax.tree.map(np.sqrt, C), make_batch(exps_b, model_b), model_b)
        return float(np.sqrt(sum(np.sum(l**2) for l in jax.tree.leaves(gr))))
    except Exception as e:   # never lose a 15-minute refit to a diagnostic
        print(f"[refit] gradient norm failed: {e!r}")
        return float("nan")


def refit(opt_params, model_b, exps_b, things, log=print):
    """Hybrid refit of opt_params (the things keys): fo_epochs of first-order steps (C = 1 on the disc), then one L-BFGS
    call from there (C = gn_probe on everything). Returns (params, losses, info): losses = the first-order losses (before
    each step) followed by the L-BFGS end loss; info = optimise_optimistix's info dict (result, num_steps, loss_start,
    loss_end) plus grad_norm_u_start / grad_norm_u_end (|sqrt(C) grad| before / after L-BFGS) and max_steps."""
    # the preconditioner for all parameters (disc included) at the start: one gn_probe call, via a discarded 1-epoch step
    _, _, C = optimise_new(opt_params, model_b, exps_b, things, 1, precond_method="gn_probe",
                           precond_kwargs=precond_kwargs, mapped=True, return_c=True, progress=False)
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
    info = dict(info, max_steps=optx_max_steps, grad_norm_u_start=precond_grad_norm(x1, C, model_b, exps_b),
                grad_norm_u_end=precond_grad_norm(x2, C, model_b, exps_b))
    log(f"[refit] L-BFGS: {info['result']} after {info['num_steps']}/{optx_max_steps} iterations, loss "
        f"{info['loss_start']:.1f} -> {info['loss_end']:.1f}, |sqrt(C) grad| {info['grad_norm_u_start']:.3g} -> "
        f"{info['grad_norm_u_end']:.3g} ({time.time() - t0:.0f} s)")
    if not all(bool(numpy.all(numpy.isfinite(numpy.asarray(l)))) for l in jax.tree.leaves(x2)):
        raise RuntimeError("L-BFGS returned non-finite parameters")
    return to_host(x2), numpy.append(losses, info["loss_end"]), info


mean_b, lp_b = None, lp_of(False)
make_b = lambda: bpl_prior_factory(resolved_wid, mean_b, lp_b)
hyp_b = dict(best_b)                                # {sigma, length_b, gamma1, dgamma}
outer_steps = dict(sigma=2**0.5, length_b=2**0.5, gamma1=0.5, dgamma=0.5)   # local grid: 3 values per axis (n=1)
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=gb,
               theta=theta_b, idx=idx_b, stage="D")
outer_converged = False
lbfgs_info = {}
for it in range(1, n_outer + 1):
    prior_b = make_b()(**hyp_b)
    h = fingerprint(it, list(hyp_b.values()), current["s"], current["resp_fp"], current["theta"], fo_epochs,
                    fo_lr_opd, fo_lr_disc, optx_max_steps, lp_b, repr(sorted(precond_kwargs.items())),
                    repr(sorted(things_gp)), code_fp)
    with timed(f"E{it}: refit"):
        exps_b = scale_errors(set_hyper(exposures, **bpl_hypers(hyp_b), mean=numpy.zeros((resolved_wid, resolved_wid)),
                                        level_prior=lp_b), current["s"])
        path = f"{out}/outer{it}-{h}-params.npy"
        ipath = f"{out}/outer{it}-{h}-lbfgs.json"
        if os.path.exists(path):
            p_new = numpy.load(path, allow_pickle=True).item()
            if os.path.exists(ipath):
                lbfgs_info[it] = json.load(open(ipath))
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
        tag = f"outer{it}_bpl_" + fingerprint(resp_new_fp, beta_new, lp_b, n_iter, code_fp)
        # local 3^4 grid; values outside the bounds are dropped (e.g. gamma1 = -0.5 about the Matern face)
        grid_new = gp_evidence_grid_nd(resp_new, local_axes(hyp_b, outer_steps, scales=BPL_SCALES, n=1), make_b(),
                                       beta=beta_new, outdir=f"{out}/grid", tag=tag, abbrev=BPL_ABBREV,
                                       bounds=bpl_bounds, scales=BPL_SCALES, n_iter=n_iter, theta0=theta_new,
                                       max_growth=2, recheck_top=3)
        idx_new = grid_new["best"]
        new = {k: float(grid_new["axes"][k][i]) for k, i in zip(grid_new["names"], idx_new)}
        print(f"outer {it}: s {numpy.round(numpy.asarray(s_new), 4)}; best {new} (was {hyp_b})")
    moved = not all(numpy.isclose(new[k], hyp_b[k], rtol=1e-6, atol=1e-9) for k in hyp_b)
    hyp_b = new
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=np.asarray(grid_new["thetas"][idx_new]), idx=idx_new, stage=f"E{it}")
    if not moved:
        outer_converged = True
        break
if not outer_converged:
    print(f"WARNING: the hyperparameters still moved at outer iteration {n_outer}: the optics were refitted at the "
          "previous hyperparameters; raise n_outer")
for it, info in lbfgs_info.items():
    if not info.get("successful", False) or info.get("reached_max_steps", False):
        print(f"WARNING: outer {it} L-BFGS did not report success ({info.get('result')}, {info.get('num_steps')} its, "
              f"|sqrt(C) grad| end {info.get('grad_norm_u_end')}): the optics may be under-converged")

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

    # posterior mixture over the final 4-D grid (every bound is an axis bound: gp_hyper_posterior's weights ARE the posterior)
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
    numpy.savez(f"{out}/spectral-diagnostic.npz", **spec)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    lo, hi = numpy.percentile(log_map, 1), log_map.max()
    panels = [(numpy.log10(numpy.asarray(d_ref)), "L1/TSV reference (log10)", (lo, hi)),
              (log_map, "BPL GP MAP (log10)", (lo, hi)),
              (numpy.log10(mix_mean), "posterior-mixture mean (log10)", (lo, hi)),
              (log_std, "Laplace std of log10 I (dex)", (None, None))]
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
           title="Posterior power / prior (broken power law)")
    fig.savefig(f"{out}/spectral-diagnostic.png")
    plt.close(fig)

    exps_f = scale_errors(set_hyper(exposures, **bpl_hypers(hyp_b), mean=numpy.zeros((resolved_wid, resolved_wid)),
                                    level_prior=lp_b), current["s"])
    model_f = set_array(ModelParams(p_final).inject(model_single))
    plot_comparison_detailed(model_f, ModelParams(set_array(p_final)), exps_f, quadrature=False, wf_size=wf_wid,
                             percentile=99, save=f"{out}/comparison")

    n_good = numpy.asarray(resp_f.n_good)
    summary = dict(
        hyper=hyp_b, gamma2=hyp_b["gamma1"] + hyp_b["dgamma"],
        break_wavenumber_cycles_per_px=float(1/(2*numpy.pi*hyp_b["length_b"])),
        final_stage=current["stage"], outer_converged=outer_converged, data_dir=ddir,
        lbfgs={str(it): info for it, info in lbfgs_info.items()},   # result status, iterations, |sqrt(C) grad| start / end
        noise_scale=dict(zip(resp_f.keys, map(float, current["s"]))),
        chi2_per_pixel_raw=dict(zip(resp_f.keys, map(float, ev_f["chi2"]/n_good))),
        logZ=float(ev_f["logZ"]), gamma=float(ev_f["gamma"]), newton_decrement=float(ev_f["newton_decrement"]),
        final_grid=dict(quantiles=q_f, selected_outside_16_84=outside, edge_weight=post_f["edge_weight"],
                        unconverged=int((~grid_f["converged"].astype(bool)).sum())),
        comparison_at_stage_D=comparison, mixture_top20_weight=float(flat_w[top].sum()),
        gradient_ratio=float(noise["g_ratio"]), verify_max_abs=verify_max_abs,
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
