"""DRAFT batch/imlup-gp-dct2.py: IM Lup with a TWO-KERNEL stationary DCT disc prior (sum of two Matern spectra on the same
latents, gp.py) against the single-kernel DCT prior of imlup-gp.py. Stages A-C are imlup-gp.py's (same optics,
response, noise scales, so the evidence of both models is on the same data and beta); then
  D  4-D evidence grid (sigma1, length1, sigma2, ratio = length2 / length1 > 1) for each order pair, plus the
     single-kernel grids as the comparison; the spectral diagnostic of both at the reference response
  E  outer loop (refit of optics + disc at the chosen hyperparameters, response rebuilt, local 4-D grid)
  F  products: MAP disc and its two scale components, Laplace std, posterior mixture, spectral diagnostic, summary.json
The multi-kernel prior (MultiGPResolvedFit, two_kernel_prior_factory, capped_posterior, ...) lives in gp.py. Stage E refits
with the hybrid schedule of imlup-gp-polar-fast.py (a short first-order stage, then one L-BFGS call). The prior-independent
response.npz / verify-*.npz are read from imlup-gp/ when present (built here otherwise). Resumable and fingerprinted
exactly as imlup-gp.py."""
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
from gp import _save_npz   # the multi-kernel prior comes with `import *`

out = "imlup-gp-dct2"
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

# Hyperparameter grid, each axis spaced by 2 (gp_evidence_grid extends it by 2 while the optimum is on an edge).
# sigma: rms of log10 I in dex; length: Matern scale in 0.0864" px (lambda/D at F160W = 0.1375" = 1.6 px)
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [0.4, 0.8, 1.6, 3.2, 6.4]
# Two-kernel grid: kernel 1 is the SHORT one (length1 < length2 = length1 * ratio breaks the label swap)
two0 = dict(sigma1=[0.125, 0.25, 0.5, 1., 2.], length1=[0.6, 1.2, 2.4, 4.8], sigma2=[0.125, 0.25, 0.5, 1., 2.],
            ratio=[2., 4., 8., 16.])                           # 5 x 4 x 5 x 4 = 400 points, then growth
# length1 and ratio axes are spaced by 2, so their cells reach these bounds (half a step either side of the end values:
# 0.6 / sqrt 2 = 0.424 .. 4.8 sqrt 2 = 6.79; ratio 2 / sqrt 2 = 1.414 .. 16 sqrt 2 = 22.6): no prior mass is left uncovered.
# length2 = length1 * ratio is capped at max_length2 = 30 px, the single-kernel hyperprior's bound (capped_posterior)
two_bounds = dict(sigma1=(0.03, 8.), length1=(0.6/2**0.5, 4.8*2**0.5), sigma2=(0.03, 8.), ratio=(2/2**0.5, 16*2**0.5))
max_length2 = 30.
order_pairs = [(1., 1.), (2., 1.)]   # (short kernel, long kernel); add (2., 2.), (1., 2.) to test the shape of each
orders = [1., 2.]           # the single-kernel DCT comparison (order 1 was imlup-gp.py's best; the pair (2,1) has an order-2 short kernel)
recheck_top = 3             # cold restarts of the best points, the same for the single- and two-kernel grids
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
outer_grid = [1/4, 1/2, 1., 2., 4.]   # local grid about the current optimum after each refit (factors)

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


class GPDiscFit(MultiGPResolvedFit):
    """MultiGPResolvedFit (K-kernel Matern prior on log10 of the disc) with imlup-adi's ResolvedFit keys."""
    PARAM_KEYS = MultiGPResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
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
        # a second kernel of negligible variance leaves the spectrum prior0's to ~1e-9 relative
        exps0 = set_hyper(exposures, sigmas=[1., 1e-8], lengths=[1.6, 12.8], orders=[2., 2.], level_prior=lp_of(False))
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
# %%
# ---------------------------------------------------------------- D: evidence grids (single kernel vs two kernels)
ABBREV = TWO_ABBREV
single_bounds = dict(sigma=(0.03, 8.), length=(0.1, 30.))   # the single-kernel hyperprior (length <= 30 px)
base_fp = fingerprint(resp_fp, beta, level_mu[False], level_sd, n_iter, code_fp)
singles, singles_post = {}, {}
with timed("D: single-kernel grids"):
    for order in orders:
        tag = f"o{order:g}_t0_" + base_fp
        # gp_evidence_grid_nd directly (not the 2-D wrapper) so that recheck_top matches the two-kernel grids
        g1 = gp_evidence_grid_nd(resp, {"sigma": sigmas0, "length": lengths0},
                                 lambda sigma, length, order=order: gp_prior(resolved_wid, sigma, length, order, None,
                                                                             lp_of(False)),
                                 beta=beta, outdir=f"{out}/grid", tag=tag, abbrev={"sigma": "s", "length": "l"},
                                 bounds=single_bounds, n_iter=n_iter, theta0=noise["theta_s"], recheck_top=recheck_top)
        p1 = gp_hyper_posterior(g1, box={"sigma": sigmas0, "length": lengths0})
        singles[order], singles_post[order] = g1, p1
        bs = gp_best(g1)
        print(f"[single o{order:g}] best converged logZ {converged_max(g1):.2f} at {bs}; log_evidence_bounds "
              f"{p1['log_evidence_bounds']:.2f}; edge {p1['edge_weight']:.3f}")

grids, posts, collapsed = {}, {}, {}
with timed("D: two-kernel grids"):
    for pair in order_pairs:
        tag = f"two_o{pair[0]:g}-{pair[1]:g}_" + base_fp
        g2 = gp_evidence_grid_nd(resp, two0, two_kernel_prior_factory(resolved_wid, pair, None, lp_of(False)), beta=beta,
                                 outdir=f"{out}/grid", tag=tag, abbrev=ABBREV, bounds=two_bounds, n_iter=n_iter,
                                 theta0=noise["theta_s"], max_growth=3, recheck_top=recheck_top)
        p2 = capped_posterior(g2, max_length2)   # log evidence on the hyperprior restricted to length2 <= max_length2
        p2["edge_weight"] = gp_hyper_posterior(g2)["edge_weight"]   # of the unmasked posterior (truncation check)
        idx2, b = capped_best(g2, p2["allowed"])
        grids[pair], posts[pair] = g2, p2 | dict(best_index=idx2, best=b)
        _save_npz(f"{out}/grid-two-o{pair[0]:g}-{pair[1]:g}.npz", logZ=g2["logZ"], converged=g2["converged"],
                  gamma=g2["gamma"], best=numpy.array(idx2), weights=p2["weights"], allowed=p2["allowed"],
                  log_evidence_bounds=p2["log_evidence_bounds"], edge_weight=p2["edge_weight"],
                  **{f"axis_{k}": v for k, v in g2["axes"].items()})
        print(f"[two {pair}] best converged logZ {float(g2['logZ'][idx2]):.2f} (length2 <= {max_length2:g}) at sigma1 "
              f"{b['sigma1']:.3g}, length1 {b['length1']:.3g}, sigma2 {b['sigma2']:.3g}, length2 "
              f"{b['length1']*b['ratio']:.3g} (ratio {b['ratio']:.3g}); log_evidence_bounds "
              f"{p2['log_evidence_bounds']:.2f}; edge weight {p2['edge_weight']:.3f}; unconverged "
              f"{int((~g2['converged'].astype(bool)).sum())}")
        # collapse: a kernel's sigma at its grid's lowest value (itself within 2 x the floor), i.e. that kernel has
        # vanished; the merged-kernel case is the ratio at its lowest grid value
        floor = lambda k: g2["axes"][k][0] <= 2*two_bounds[k][0]
        col = dict(sigma1=bool(idx2[0] == 0 and floor("sigma1")), sigma2=bool(idx2[2] == 0 and floor("sigma2")),
                   ratio_at_lower_face=bool(idx2[3] == 0))
        collapsed[pair] = col
        if any(col.values()):
            print(f"NOTE {pair}: collapse flags {col}: one kernel (or two merged kernels) suffices")

# the two-kernel model is only carried forward if it beats the best single kernel (log evidence under the stated,
# matched hyperpriors: sigma (0.03, 8) both; length <= 30 px both, ratio / length1 boxes in two_bounds) and has not collapsed
best_pair = max(posts, key=lambda p: posts[p]["log_evidence_bounds"])
best_single = max(singles_post, key=lambda o: singles_post[o]["log_evidence_bounds"])
dlogZ_max = float(grids[best_pair]["logZ"][posts[best_pair]["best_index"]]
                  - max(converged_max(g) for g in singles.values()))
dlogE = float(posts[best_pair]["log_evidence_bounds"] - singles_post[best_single]["log_evidence_bounds"])
use_two = dlogE > 0 and not any(collapsed[best_pair].values())
print(f"two-kernel {best_pair} vs single-kernel o{best_single:g}: max converged logZ difference {dlogZ_max:+.2f} nats, "
      f"difference of log evidence under the matched hyperpriors {dlogE:+.2f} nats (the extra hyperparameters are charged "
      "by the hyperprior volume; the Laplace log Z is biased up by an amount growing with sigma, gp.py: validate by "
      f"injection). Use the two-kernel model for E/F: {use_two}")

# spectral diagnostic of both at the reference response: does the two-kernel spectrum fix the mismatch?
g2 = grids[best_pair]
b2 = posts[best_pair]["best"]
prior_2 = two_kernel_prior_factory(resolved_wid, best_pair, None, lp_of(False))(**b2)
theta_2 = np.asarray(g2["thetas"][posts[best_pair]["best_index"]])
g1 = singles[best_single]
b1 = gp_best(g1)
prior_1 = gp_prior(resolved_wid, b1["sigma"], b1["length"], best_single, None, lp_of(False))
theta_1 = np.asarray(g1["thetas"][g1["best"]])
spec_1 = gp_spectral_diagnostic(prior_1, theta_1, gp_laplace(resp, prior_1, theta_1, beta))
spec_2 = gp_spectral_diagnostic(prior_2, theta_2, gp_laplace(resp, prior_2, theta_2, beta))
numpy.savez(f"{out}/spectral-diagnostic-D.npz", **{f"single_{k}": v for k, v in spec_1.items()},
            **{f"two_{k}": v for k, v in spec_2.items()})
for name, s in (("single", spec_1), ("two-kernel", spec_2)):
    print(f"spectral diagnostic ({name}): <z^2> per bin {numpy.round(s['power'][:12], 2)} ... max |log10| deviation "
          f"{numpy.max(numpy.abs(numpy.log10(s['power']))):.2f}")
fig, ax = plt.subplots(figsize=(11, 9))
for name, s in (("single kernel", spec_1), ("two kernels", spec_2)):
    ax.errorbar(s["freq"], s["power"], s["power_err"], fmt="o-", label=name)
ax.axhline(1, color="k", ls="--")
ax.set(xscale="log", yscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="<z^2> (prior: 1)")
ax.legend()
fig.savefig(f"{out}/spectral-diagnostic-D.png")
plt.close(fig)

comparison = dict(
    best_single_order=best_single, best_two_order_pair=list(best_pair), use_two_kernel=use_two,
    max_converged_logZ_difference_two_minus_single=dlogZ_max, log_evidence_difference_matched_hyperpriors=dlogE,
    collapsed={f"{p[0]:g}-{p[1]:g}": c for p, c in collapsed.items()},
    two_kernel_log_evidence={f"{p[0]:g}-{p[1]:g}": q["log_evidence_bounds"] for p, q in posts.items()},
    two_kernel_best={f"{p[0]:g}-{p[1]:g}": q["best"] for p, q in posts.items()},
    single_log_evidence={f"o{o:g}": q["log_evidence_bounds"] for o, q in singles_post.items()},
    # both hyperpriors are log-uniform: sigma (0.03, 8); single: length (0.1, 30); two: length1, ratio boxes in two_bounds
    # restricted to length2 <= max_length2 (renormalised); the grids' unmasked edge weights are in grid-two-*.npz
    hyperpriors=dict(single=single_bounds, two=two_bounds, max_length2=max_length2),
    spectral_max_abs_log10_single=float(numpy.max(numpy.abs(numpy.log10(spec_1["power"])))),
    spectral_max_abs_log10_two=float(numpy.max(numpy.abs(numpy.log10(spec_2["power"])))))
with open(f"{out}/comparison-D.json", "w") as f:
    json.dump(comparison, f, indent=2, default=float)
if not use_two:
    print("the two-kernel prior does not beat the single kernel under the matched hyperpriors (or has collapsed): "
          "stages E/F are not run; use imlup-gp.py's single-kernel pipeline. See comparison-D.json")
    sys.exit(0)

# %%
# ---------------------------------------------------------------- E: outer iteration (two-kernel, best order pair)
g = 5e-2
things_gp = {
    "primary_opd": adam(2e-2, 50),
    "spectrum": sgd(g*1, 0), "primary_tilt": sgd(g*1., 0), "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1., 0), "defocus1": sgd(g*1, 0), "defocus2": sgd(g*1, 0), "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0), "cold_mask_rot": sgd(g*1., 0.), "primary_rot": sgd(g*3., 0),
    "primary_low": sgd(g*1, 0), "cold_mask_scale": sgd(g*1, 0), "cold_mask_distortion": sgd(g*3, 0),
    "occulter_radius": sgd(g*1., 0), "occulter_ellipticity": sgd(g*3, 300), "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0), "primary_spider": sgd(g*1., 0), "primary_secondary": sgd(g*1., 0),
    "secondary_shift": sgd(g*1, 0), "primary_distortion": sgd(g*3, 0), "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0), "distortion": sgd(g*1, 0),
    # (primary_opd and resolved are replaced by small Adam rates from step 0 in the first-order stage of refit())
    "resolved": adam(1e-2, 0), "resolved_level": sgd(g*1, 0),
}
check_gp_setup(ref_params, things_gp)


def refit(opt_params, model_b, exps_b, things, log=print):
    """Hybrid refit of opt_params (the things keys): fo_epochs of first-order steps (C = 1 on the disc), then one L-BFGS
    call from there (C = gn_probe on everything). Returns (params, losses, info): losses = the first-order losses (before
    each step) followed by the L-BFGS end loss; info = optimise_optimistix's info dict."""
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
    log(f"[refit] L-BFGS: {info['result']} after {info['num_steps']} iterations, loss {info['loss_start']:.1f} -> "
        f"{info['loss_end']:.1f} ({time.time() - t0:.0f} s)")
    if not all(bool(numpy.all(numpy.isfinite(numpy.asarray(l)))) for l in jax.tree.leaves(x2)):
        raise RuntimeError("L-BFGS returned non-finite parameters")
    return to_host(x2), numpy.append(losses, info["loss_end"]), info


mean_b, lp_b, pair_b = None, lp_of(False), best_pair
make_b = lambda: two_kernel_prior_factory(resolved_wid, pair_b, mean_b, lp_b)
hyp_b = dict(b2)                                    # {sigma1, length1, sigma2, ratio}
steps_b = dict(sigma1=2., length1=2., sigma2=2., ratio=2.)
outer_steps = {k: v**0.5 for k, v in steps_b.items()}   # local grid: 3 values per axis (n=1) at sqrt(2) spacing
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=g2,
               theta=theta_2, stage="D")
outer_converged = False
for it in range(1, n_outer + 1):
    prior_b = make_b()(**hyp_b)
    sigmas_b, lengths_b = two_kernel_hypers(hyp_b)
    h = fingerprint(it, pair_b, list(hyp_b.values()), current["s"], current["resp_fp"], current["theta"], fo_epochs,
                    fo_lr_opd, fo_lr_disc, optx_max_steps, lp_b, repr(sorted(precond_kwargs.items())),
                    repr(sorted(things_gp)), code_fp)
    with timed(f"E{it}: refit"):
        exps_b = scale_errors(set_hyper(exposures, sigmas=sigmas_b, lengths=lengths_b, orders=list(pair_b),
                                        mean=numpy.zeros((resolved_wid, resolved_wid)), level_prior=lp_b), current["s"])
        path = f"{out}/outer{it}-{h}-params.npy"
        if os.path.exists(path):
            p_new = numpy.load(path, allow_pickle=True).item()
        else:
            z_b, l_b = prior_b.params(current["theta"])
            start = dict(current["params"]) | {"resolved": {key: z_b}, "resolved_level": {key: l_b}}
            model_b = set_array(ModelParams(start).inject(model_single))
            opt_params = set_array({k: start[k] for k in things_gp})
            x_new, losses, optx_info = refit(opt_params, model_b, exps_b, things_gp)
            fig = plt.figure(figsize=(10, 10))
            plt.plot(losses[:-1])
            plt.plot([fo_epochs], losses[-1:], "o", label="after L-BFGS")   # the end of the one optimistix call
            plt.legend()
            fig.savefig(f"{out}/outer{it}-losses.png")
            plt.close(fig)
            if not (numpy.all(numpy.isfinite(losses)) and losses[-1] <= losses[0]):   # never worse than the start
                raise RuntimeError(f"outer refit {it}: loss {losses[0]:.1f} -> {losses[-1]:.1f} (non-finite or worse "
                                   "than the start); not saving it")
            p_new = start | x_new
            save_npy(f"{out}/outer{it}-{h}-losses.npy", losses)
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
        tag = f"outer{it}_two_o{pair_b[0]:g}-{pair_b[1]:g}_" + fingerprint(resp_new_fp, beta_new, lp_b, n_iter, code_fp)
        grid_new = gp_evidence_grid_nd(resp_new, local_axes(hyp_b, outer_steps, n=1), make_b(), beta=beta_new,
                                       outdir=f"{out}/grid", tag=tag, abbrev=ABBREV, bounds=two_bounds, n_iter=n_iter,
                                       theta0=theta_new, max_growth=2, recheck_top=3)
        post_new = capped_posterior(grid_new, max_length2)
        idx_new, new = capped_best(grid_new, post_new["allowed"])
        print(f"outer {it}: s {numpy.round(numpy.asarray(s_new), 4)}; best {new} (was {hyp_b})")
    moved = not all(numpy.isclose(new[k], hyp_b[k], rtol=1e-6) for k in hyp_b)
    hyp_b = new
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=np.asarray(grid_new["thetas"][idx_new]), stage=f"E{it}")
    if not moved:
        outer_converged = True
        break
if not outer_converged:
    print(f"WARNING: the hyperparameters still moved at outer iteration {n_outer}: the optics were refitted at the "
          "previous hyperparameters; raise n_outer")

# %%
# ---------------------------------------------------------------- F: products
with timed("F: products"):
    resp_f, beta_f, grid_f = current["resp"], current["beta"], current["grid"]
    prior_f = make_b()(**hyp_b)
    theta_f = current["theta"]
    sigmas_f, lengths_f = two_kernel_hypers(hyp_b)
    ev_f = to_host(gp_log_evidence(resp_f, prior_f, theta_f, beta_f))
    lap = gp_laplace(resp_f, prior_f, theta_f, beta_f)
    log_map = numpy.asarray(prior_f.log10_image(theta_f))
    log_std = numpy.asarray(gp_log10_std(prior_f, lap))
    ds = jax.vmap(prior_f.distribution)(gp_laplace_samples(theta_f, lap, jr.PRNGKey(1), n=200))
    d_mean, d_std = numpy.asarray(ds.mean(0)), numpy.asarray(ds.std(0))

    # scale decomposition of the MAP field: E[f_k | f] = idctn(S_k / S * amp * z), exactly the conditional mean of
    # kernel k's component given the total field (the two sum to the MAP field)
    z_f, l_f = prior_f.params(theta_f)
    S_k = [matern_spectrum((resolved_wid,)*2, l, s, o) for s, l, o in zip(sigmas_f, lengths_f, pair_b)]
    S_tot = sum(S_k)
    comps = [numpy.asarray(gp_field(np.where(S_tot > 0, S/np.where(S_tot > 0, S_tot, 1.), 0.)*z_f, prior_f.amp))
             for S in S_k]
    assert numpy.allclose(sum(comps), numpy.asarray(prior_f.field(theta_f[:-1]).reshape(resolved_wid, -1)), atol=1e-9)

    # posterior mixture over the final 4-D grid, cells drawn in proportion to their evidence weight
    post_f = gp_hyper_posterior(grid_f)
    post_c = capped_posterior(grid_f, max_length2)       # weights on length2 <= max_length2
    flat_w = post_c["weights"].ravel()
    top = numpy.argsort(flat_w)[::-1][:20]               # at most 20 cells (each costs an eigh): renormalised
    w_top = numpy.zeros_like(flat_w)
    w_top[top] = flat_w[top]/flat_w[top].sum()
    print(f"mixture: top 20 cells carry {flat_w[top].sum():.3f} of the posterior weight")
    draws, n_mix = [], 400
    for idx, w in numpy.ndenumerate(w_top.reshape(post_c["weights"].shape)):
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

    numpy.save(f"{out}/gp-disc-log10.npy", log_map)
    numpy.save(f"{out}/gp-disc-log10-std.npy", log_std)
    numpy.save(f"{out}/gp-components-log10.npy", numpy.stack(comps))
    numpy.save(f"{out}/gp-mixture-mean.npy", mix_mean)
    numpy.save(f"{out}/gp-mixture-std.npy", mix_std)
    numpy.savez(f"{out}/spectral-diagnostic.npz", **spec)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    lo, hi = numpy.percentile(log_map, 1), log_map.max()
    panels = [(numpy.log10(numpy.asarray(d_ref)), "L1/TSV reference (log10)", (lo, hi)),
              (log_map, "two-kernel GP MAP (log10)", (lo, hi)),
              (comps[0], f"short kernel (length {lengths_f[0]:.3g} px) component", (None, None)),
              (comps[1], f"long kernel (length {lengths_f[1]:.3g} px) component", (None, None)),
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
           title="Posterior power / prior (two-kernel)")
    fig.savefig(f"{out}/spectral-diagnostic.png")
    plt.close(fig)

    exps_f = scale_errors(set_hyper(exposures, sigmas=sigmas_f, lengths=lengths_f, orders=list(pair_b),
                                    mean=numpy.zeros((resolved_wid, resolved_wid)), level_prior=lp_b), current["s"])
    model_f = set_array(ModelParams(p_final).inject(model_single))
    plot_comparison_detailed(model_f, ModelParams(set_array(p_final)), exps_f, quadrature=False, wf_size=wf_wid,
                             percentile=99, save=f"{out}/comparison")

    n_good = numpy.asarray(resp_f.n_good)
    summary = dict(
        order_pair=list(pair_b), hyper=hyp_b, sigmas=list(map(float, sigmas_f)), lengths=list(map(float, lengths_f)),
        final_stage=current["stage"], outer_converged=outer_converged, data_dir=ddir,
        noise_scale=dict(zip(resp_f.keys, map(float, current["s"]))),
        chi2_per_pixel_raw=dict(zip(resp_f.keys, map(float, ev_f["chi2"]/n_good))),
        logZ=float(ev_f["logZ"]), gamma=float(ev_f["gamma"]), newton_decrement=float(ev_f["newton_decrement"]),
        final_grid=dict(quantiles={k: list(map(float, v)) for k, v in post_f["quantiles"].items()},
                        edge_weight=post_f["edge_weight"], unconverged=int((~grid_f["converged"].astype(bool)).sum())),
        comparison_at_stage_D=comparison, mixture_top20_weight=float(flat_w[top].sum()),
        gradient_ratio=float(noise["g_ratio"]), verify_max_abs=float(v["abs_diff"].max()),
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
