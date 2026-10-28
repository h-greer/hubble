# %%
"""IM Lup (SZ-82, NIC2 F160W; 2 rolls x 2 exposures) with a Gaussian-process disc prior whose hyperparameters are chosen
by the Laplace evidence: phases 0-2 of notebooks/gp-plan.md. Needs imlup-adi's results (imlup-adi/intermediate-params.npy
and params_amp.npy) as the reference optics, star and L1/TSV disc. Stages, saving to imlup-gp/:
  A  exposures (GPResolvedFit on an InterpolatedResolvedSource, 60 px box), model and reference params
  B  LinearResponse at the reference optics (response.npz), checked against the full model (verify.npz)
  C  MAP at (1 dex, 1.6 px), full-model gradient check at it, per-exposure noise scales (noise-*.npz)
  D  evidence grids over (sigma, length) for order 1, 2, 3, without and with the L1/TSV template (grid/, grid-*.npz)
  E  outer iteration: full refit of optics + disc at the chosen hyperparameters, response rebuilt, local grid (outer*)
  F  products: MAP disc, Laplace std / SNR, posterior-mixture mean, spectral diagnostic, comparisons (summary.json)
Resumable: every result is saved atomically under a name carrying a fingerprint of what it depends on (the response,
noise scales, prior settings, gp.py), and reloaded on a rerun; anything stale is recomputed, never reused. Delete
response.npz (and response-outer*.npz) to rebuild the responses. The injection tests (imlup-gp-inject.sh) only need
response.npz, so they can be submitted as soon as stage B has finished.
"""
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
from gp import _save_npz

out = "imlup-gp"
ref_dir = "imlup-adi"
os.makedirs(out, exist_ok=True)
os.makedirs("imlup-gp-inject", exist_ok=True)   # imlup-gp-inject.sh's SLURM logs go there


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


with open("../gp.py", "rb") as f:
    code_fp = hashlib.sha256(f.read()).hexdigest()[:12]   # results depend on gp.py's maths

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
orders = [1., 2., 3.]
templates = [False, True]   # flat prior mean, or the L1/TSV disc as a log10 template
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
outer_epochs = 1000
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


class GPDiscFit(GPResolvedFit):
    """GPResolvedFit (InterpolatedResolvedSource, Matern prior on log10 of the disc) with imlup-adi's ResolvedFit keys."""
    PARAM_KEYS = GPResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
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
print(f"reference level (dex): flat mean {level_mu[False]:.3f}, template {level_mu[True]:.3f}")

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
    if os.path.exists(f"{out}/response.npz"):
        resp = LinearResponse.load(f"{out}/response.npz")
    else:
        resp = build_linear_response(model_single, exposures, batch_size=response_batch)
        resp.save(f"{out}/response.npz")
    resp_fp = response_fingerprint(resp)
print(f"response ready ({out}/response.npz, fingerprint {resp_fp}): imlup-gp-inject.sh can be submitted now")

with timed("B: verify"):
    vpath = f"{out}/verify-{resp_fp}.npz"
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
prior0 = gp_prior(resolved_wid, 1., 1.6, 2., None, lp_of(False))

with timed("C: noise"):
    npath = f"{out}/noise-{fingerprint(resp_fp, level_mu[False], level_sd, n_iter, code_fp)}.npz"
    if not os.path.exists(npath):
        theta1, hist1 = gp_map(resp, prior0, n_iter=n_iter)
        ev1 = to_host(gp_log_evidence(resp, prior0, theta1))

        # The full model's gradient in z at the surrogate MAP (optics fixed) should vanish, relative to its value at
        # z = 0; and its loss should equal the surrogate's (including the prior)
        exps0 = set_hyper(exposures, sigma=1., length=1.6, order=2., level_prior=lp_of(False))
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
# ---------------------------------------------------------------- D: evidence grids
print("NOTE: the template variants use a template estimated from these data (the L1/TSV disc), so their evidence is "
      "optimistic (empirical Bayes on the template too); they are reported but not used for E/F unless "
      "final_allows_template")
grids, posts = {}, {}
with timed("D: grids"):
    for order in orders:
        for template in templates:
            tag = f"o{order:g}_t{int(template)}_" + fingerprint(resp_fp, beta, level_mu[template], level_sd, n_iter,
                                                                 template_floor, code_fp)
            grid = gp_evidence_grid(resp, sigmas0, lengths0, order, mean_of(template), lp_of(template), beta,
                                    outdir=f"{out}/grid", tag=tag, n_iter=n_iter,
                                    theta0=None if template else noise["theta_s"])
            # variants are compared on the starting grid's cells (every grid contains them), so that a grid that grew
            # further does not carry a different hyperprior
            post = gp_hyper_posterior(grid, box=(sigmas0, lengths0))
            grids[(order, template)], posts[(order, template)] = grid, post
            _save_npz(f"{out}/grid-o{order:g}_t{int(template)}.npz", **{k: v for k, v in grid.items() if k != "tag"},
                      weights=post["weights"], sigma_q=post["sigma_q"], length_q=post["length_q"],
                      log_evidence=post["log_evidence"], log_evidence_box=post["log_evidence_box"],
                      edge_weight=post["edge_weight"])
            i, j = grid["best"]
            print(f"[o{order:g}_t{int(template)}] log evidence (starting grid) {post['log_evidence_box']:.2f}; best sigma "
                  f"{grid['sigmas'][i]:.3g} dex, length {grid['lengths'][j]:.3g} px; sigma 16/50/84% "
                  f"{numpy.round(post['sigma_q'], 3)}, length {numpy.round(post['length_q'], 3)}; edge weight "
                  f"{post['edge_weight']:.3f}; unconverged points {int((~grid['converged'].astype(bool)).sum())}")

candidates = [v for v in posts if final_allows_template or not v[1]]
variant = max(candidates, key=lambda v: posts[v]["log_evidence_box"])
order_b, template_b = variant
print(f"variant for E/F: order {order_b:g}, template {template_b}; best overall: "
      f"{max(posts, key=lambda v: posts[v]['log_evidence_box'])}")

# evidence surfaces, relative to the best point over all variants
zmax = max(g["logZ"].max() for g in grids.values())
fig, axs = plt.subplots(len(templates), len(orders), figsize=(10*len(orders), 9*len(templates)), squeeze=False)
for (order, template), g in grids.items():
    ax = axs[templates.index(template), orders.index(order)]
    im = ax.imshow(numpy.clip(g["logZ"] - zmax, -50, 0), cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(g["lengths"])), [f"{l:.3g}" for l in g["lengths"]], rotation=45)
    ax.set_yticks(range(len(g["sigmas"])), [f"{s:.3g}" for s in g["sigmas"]])
    ax.set(xlabel="length (px)", ylabel="sigma (dex)", title=f"order {order:g}, template {template}")
    ax.plot(g["best"][1], g["best"][0], "r*", ms=20)
    plt.colorbar(im, ax=ax, label="log Z - max")
fig.tight_layout()
fig.savefig(f"{out}/evidence.png")
plt.close(fig)

# %%
# ---------------------------------------------------------------- E: outer iteration
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
    # the disc starts at the surrogate MAP, so it is released at once (C = 1 for z, the prior's scale; Adam)
    "resolved": adam(1e-2, 0), "resolved_level": sgd(g*1, 0),
}
check_gp_setup(ref_params, things_gp)

mean_b, lp_b = mean_of(template_b), lp_of(template_b)
grid_b = grids[variant]
sig_b, ell_b = float(grid_b["sigmas"][grid_b["best"][0]]), float(grid_b["lengths"][grid_b["best"][1]])
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=grid_b,
               theta=np.asarray(grid_b["thetas"][grid_b["best"]]), stage="D")
outer_converged = False
for it in range(1, n_outer + 1):
    prior_b = gp_prior(resolved_wid, sig_b, ell_b, order_b, mean_b, lp_b)
    h = fingerprint(it, order_b, template_b, sig_b, ell_b, current["s"], current["resp_fp"], current["theta"],
                    outer_epochs, code_fp)
    with timed(f"E{it}: refit"):
        exps_b = scale_errors(set_hyper(exposures, sigma=sig_b, length=ell_b, order=order_b,
                                        mean=numpy.zeros((resolved_wid, resolved_wid)) if mean_b is None else mean_b,
                                        level_prior=lp_b), current["s"])
        path = f"{out}/outer{it}-{h}-params.npy"
        if os.path.exists(path):
            p_new = numpy.load(path, allow_pickle=True).item()
        else:
            z_b, l_b = prior_b.params(current["theta"])
            start = dict(current["params"]) | {"resolved": {key: z_b}, "resolved_level": {key: l_b}}
            model_b = set_array(ModelParams(start).inject(model_single))
            opt_params = set_array({k: start[k] for k in things_gp})
            losses, history = optimise_new_resolved(opt_params, model_b, exps_b, things_gp, outer_epochs,
                                                    precond_method="gn_probe", precond_kwargs=precond_kwargs,
                                                    mapped=True)
            fig = plt.figure(figsize=(10, 10))
            plt.plot(losses)
            fig.savefig(f"{out}/outer{it}-losses.png")
            plt.close(fig)
            if len(losses) < outer_epochs:
                raise RuntimeError(f"outer refit {it} stopped after {len(losses)}/{outer_epochs} steps (non-finite); "
                                   "not saving it")
            p_new = start | to_host(history[-1])
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
        # noise scales re-estimated at the refitted optics, then a local grid about the current optimum
        npath = f"{out}/noise-outer{it}-{fingerprint(resp_new_fp, h)}.npz"
        if os.path.exists(npath):
            f = numpy.load(npath)
            s_new, theta_new = f["s"], np.asarray(f["theta"])
        else:
            theta_new = prior_b.theta(np.asarray(p_new["resolved"][key]), np.asarray(p_new["resolved_level"][key]))
            s_new, theta_new, _ = gp_error_scale(resp_new, prior_b, theta_new)
            _save_npz(npath, s=s_new, theta=theta_new)
        beta_new = 1/np.asarray(s_new)**2
        tag = f"outer{it}_o{order_b:g}_t{int(template_b)}_" + fingerprint(resp_new_fp, beta_new, lp_b, n_iter, code_fp)
        grid_new = gp_evidence_grid(resp_new, [sig_b*x for x in outer_grid], [ell_b*x for x in outer_grid], order_b,
                                    mean_b, lp_b, beta_new, outdir=f"{out}/grid", tag=tag, n_iter=n_iter,
                                    theta0=theta_new, max_growth=2)
        i, j = grid_new["best"]
        sig_new, ell_new = float(grid_new["sigmas"][i]), float(grid_new["lengths"][j])
        print(f"outer {it}: s {numpy.round(numpy.asarray(s_new), 4)}; best sigma {sig_new:.3g} dex, "
              f"length {ell_new:.3g} px (was {sig_b:.3g}, {ell_b:.3g})")
    moved = not (numpy.isclose(sig_new, sig_b) and numpy.isclose(ell_new, ell_b))
    sig_b, ell_b = sig_new, ell_new
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=np.asarray(grid_new["thetas"][i, j]), stage=f"E{it}")
    if not moved:
        outer_converged = True
        break
if not outer_converged:
    print(f"WARNING: the hyperparameters still moved at outer iteration {n_outer}: the optics were refitted at the "
          "previous (sigma, length); raise n_outer")

# %%
# ---------------------------------------------------------------- F: products
with timed("F: products"):
    resp_f, beta_f, grid_f = current["resp"], current["beta"], current["grid"]
    prior_f = gp_prior(resolved_wid, sig_b, ell_b, order_b, mean_b, lp_b)
    theta_f = current["theta"]
    ev_f = to_host(gp_log_evidence(resp_f, prior_f, theta_f, beta_f))
    lap = gp_laplace(resp_f, prior_f, theta_f, beta_f)
    log_map = numpy.asarray(prior_f.log10_image(theta_f))
    log_std = numpy.asarray(gp_log10_std(prior_f, lap))
    ds = jax.vmap(prior_f.distribution)(gp_laplace_samples(theta_f, lap, jr.PRNGKey(1), n=200))
    d_mean, d_std = numpy.asarray(ds.mean(0)), numpy.asarray(ds.std(0))

    # posterior mixture over the final grid: cells drawn in proportion to their evidence weight
    post_f = gp_hyper_posterior(grid_f)
    draws, n_mix = [], 400
    for (i, j), w in numpy.ndenumerate(post_f["weights"]):
        m = int(round(n_mix*w))
        if m == 0:
            continue
        prior_c = gp_prior(resolved_wid, grid_f["sigmas"][i], grid_f["lengths"][j], order_b, mean_b, lp_b)
        th = np.asarray(grid_f["thetas"][i, j])
        draws.append(jax.vmap(prior_c.distribution)(
            gp_laplace_samples(th, gp_laplace(resp_f, prior_c, th, beta_f), jr.PRNGKey(10 + i*100 + j), n=m)))
    mix = np.concatenate(draws)
    mix_mean, mix_std = numpy.asarray(mix.mean(0)), numpy.asarray(mix.std(0))
    spec = gp_spectral_diagnostic(prior_f, theta_f, lap)

    numpy.save(f"{out}/gp-disc-log10.npy", log_map)
    numpy.save(f"{out}/gp-disc-log10-std.npy", log_std)
    numpy.save(f"{out}/gp-mixture-mean.npy", mix_mean)
    numpy.save(f"{out}/gp-mixture-std.npy", mix_std)
    numpy.savez(f"{out}/spectral-diagnostic.npz", **spec)
    z_f, l_f = prior_f.params(theta_f)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    # discs: L1/TSV reference, GP MAP, posterior mean, uncertainty, SNR
    lo = numpy.percentile(log_map, 1)
    hi = log_map.max()
    panels = [(numpy.log10(numpy.asarray(d_ref)), "L1/TSV reference (log10)", (lo, hi)),
              (log_map, "GP MAP (log10)", (lo, hi)),
              (numpy.log10(mix_mean), "GP posterior-mixture mean (log10)", (lo, hi)),
              (log_std, "Laplace std of log10 I (dex)", (None, None)),
              (d_mean/d_std, "Laplace mean / std", (None, None))]
    fig, axs = plt.subplots(1, len(panels), figsize=(12*len(panels), 11))
    for ax, (im, title, (vmin, vmax)) in zip(axs, panels):
        c = ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out}/discs.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(22, 9))
    axs[0].errorbar(spec["freq"], spec["power"], spec["power_err"], fmt="o")
    axs[0].axhline(1, color="k", ls="--")
    axs[0].set(xscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="<z^2> (prior: 1)",
               title="Posterior power / prior")
    axs[1].plot(spec["freq"], spec["var"], "o")
    axs[1].set(xscale="log", xlabel="DCT wavenumber (cycles / px)", ylabel="posterior variance (prior: 1)",
               title="Data constraint per mode")
    fig.tight_layout()
    fig.savefig(f"{out}/spectral-diagnostic.png")
    plt.close(fig)

    # prior draws at the chosen hyperparameters (level at its prior mean)
    prior_draws = jr.normal(jr.PRNGKey(2), (4, n + 1)).at[:, -1].set(0.)
    fig, axs = plt.subplots(1, 4, figsize=(44, 11))
    for ax, th in zip(axs, prior_draws):
        c = ax.imshow(prior_f.log10_image(th))
        plt.colorbar(c, ax=ax)
    axs[0].set_title(f"prior draws (log10): sigma {sig_b:.3g} dex, length {ell_b:.3g} px, order {order_b:g}")
    fig.tight_layout()
    fig.savefig(f"{out}/prior-draws.png")
    plt.close(fig)

    exps_f = scale_errors(set_hyper(exposures, sigma=sig_b, length=ell_b, order=order_b,
                                    mean=numpy.zeros((resolved_wid, resolved_wid)) if mean_b is None else mean_b,
                                    level_prior=lp_b), current["s"])
    model_f = set_array(ModelParams(p_final).inject(model_single))
    plot_comparison_detailed(model_f, ModelParams(set_array(p_final)), exps_f, quadrature=False, wf_size=wf_wid,
                             percentile=99, save=f"{out}/comparison")

    n_good = numpy.asarray(resp_f.n_good)
    summary = dict(
        variant=dict(order=order_b, template=template_b), sigma=sig_b, length=ell_b, final_stage=current["stage"],
        outer_converged=outer_converged, data_dir=ddir,
        noise_scale=dict(zip(resp_f.keys, map(float, current["s"]))),
        # chi2 / n_good with the errors as given (~ s^2: the goodness of fit), and with the rescaled errors
        # (~ 1 - gamma_e / n_good by construction of the noise scale, not a test of the fit)
        chi2_per_pixel_raw=dict(zip(resp_f.keys, map(float, ev_f["chi2"]/n_good))),
        chi2_per_pixel_rescaled=dict(zip(resp_f.keys, map(float, ev_f["chi2"]*numpy.asarray(beta_f)/n_good))),
        gamma_per_exposure=dict(zip(resp_f.keys, map(float, ev_f["gamma_e"]))),
        logZ=float(ev_f["logZ"]), gamma=float(ev_f["gamma"]), newton_decrement=float(ev_f["newton_decrement"]),
        final_grid=dict(sigma_q=list(map(float, post_f["sigma_q"])), length_q=list(map(float, post_f["length_q"])),
                        edge_weight=post_f["edge_weight"],
                        unconverged=int((~grid_f["converged"].astype(bool)).sum())),
        stage_d=dict(sigma_q=list(map(float, posts[variant]["sigma_q"])),
                     length_q=list(map(float, posts[variant]["length_q"])),
                     edge_weight=posts[variant]["edge_weight"]),
        # log evidence of each variant under a log-uniform hyperprior on the starting grid (sigmas0 x lengths0); the
        # Laplace approximation's bias (gp.py) is not corrected
        variant_log_evidence={f"o{o:g}_t{int(t)}": p["log_evidence_box"] for (o, t), p in posts.items()},
        variant_unconverged={f"o{o:g}_t{int(t)}": int((~grids[(o, t)]["converged"].astype(bool)).sum())
                             for (o, t) in grids},
        verify_max_abs=float(v["abs_diff"].max()), verify_max_rel_change=float(v["rel_change"].max()),
        gradient_ratio=float(noise["g_ratio"]),
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
