# %%
"""imlup-gp-polar.py (imlup-gp-polar.sh): IM Lup (SZ-82, NIC2 F160W; 2 rolls x 2 exposures) with a Gaussian-process
disc prior whose hyperparameters are chosen by the Laplace evidence: phases 0-2 of notebooks/gp-plan.md, with the polar
(multipole) field after virgil's MultipoleField (option A: polar only). The previous run's DCT Matern prior (order 1,
flat mean) is rerun on the same response as a baseline for the evidence comparison. Needs imlup-adi's results
(imlup-adi/intermediate-params.npy and params_amp.npy) as the reference optics, star and L1/TSV disc. Stages, saving
to imlup-gp/:
  A  exposures (PolarGPResolvedFit on an InterpolatedResolvedSource, 60 px box; the polar grid fixed in the disc plane),
     model and reference params
  B  LinearResponse at the reference optics (response.npz), checked against the full model (verify.npz)
  C  at a provisional prior (mid-grid shape, cos_inc 0.7, pa 0): the full-model gradient check and the per-exposure
     noise scales (noise-*.npz), which every evidence below uses
  D  the hyperparameters by the evidence: rounds of a coarse geometry grid (cos_inc x pa) at the current shape and the
     shape grid (sigma, length_r, length_theta) at its best geometry, until the geometry repeats (at most 2), then a
     joint local grid over all five, re-centred on its best, which gives the posterior and the evidence; the DCT
     baseline over (sigma, length) (grid/, grid-*.npz)
The geometry (inclination, position angle, optionally the centre) is a hyperparameter of the polar prior like sigma and
the lengths, so it is chosen the same way, by the Laplace evidence on the linear response; nothing is taken from a fit
to a reference image or from the literature, so the same script applies to any (not edge-on) disc.
  E  outer iteration: full refit of optics + disc at the chosen hyperparameters, response rebuilt, local grid (outer*)
  F  products: MAP disc, Laplace std / SNR, posterior-mixture mean, polar diagnostic, comparisons (summary.json)
This script shares imlup-gp/ with the DCT-prior script imlup-gp.py (out = "imlup-gp", unchanged), so it reuses that
run's response.npz and verify-*.npz (pixel space, independent of the prior). Before the first polar run, move the DCT
run's other products out of imlup-gp/ (to imlup-gp/dct-run/): nothing stale is reused (every name carries a
fingerprint), but the old PNGs and grids would sit beside the new ones.
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
import itertools
import gp
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
from gp import _save_npz   # gp_evidence_grid_nd, PolarGrid, polar_prior, ... come with `import *`

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

# Polar (multipole) prior: a log10 field on a polar grid in the deprojected disc plane (gp.PolarGrid). The radial grid
# is fixed in the disc plane, whatever the geometry: r_max = the box's half-diagonal deprojected at cover_inc, so every
# pixel is inside it for inclinations up to cover_inc (beyond, the outermost ring's value extends: a few % of pixels at
# the cos_inc bound below). 49 x 128 latents: radial spacing ~1.7 disc-plane px, arc spacing ~1.5 px at 30 px;
# nu = 3/2 radially and azimuthally (virgil's default). More inclined discs need a larger cover_inc and n_r (cost ~n^3)
n_r, n_theta, cover_inc = 49, 128, 60.
nu_r = nu_theta = 1.5
# Hyperparameters, all chosen by the evidence; their (uniform) hyperpriors' bounds are fixed here, before the data:
#   sigma (dex, pointwise rms of log10 I at the nodes), length_r (disc-plane grid px; lambda/D at F160W = 1.6 px; lower
#   bound: the node spacing, set once the grid is built), length_theta (rad): log-uniform;
#   cos_inc: uniform on [cos 65 deg, 1] (isotropic orientations; the radial grid covers the box to ~60 deg, and a thin
#   disc's deprojection fails towards edge on);
#   pa (deg, the projected major axis from +y = rows towards +x = columns, in the disc grid's frame): uniform over its
#   period, 180 deg (a flat disc's prior is unchanged by pa + 180, for even n_theta);
#   dx, dy (grid px, the disc centre's offset from the star): uniform within the bounds, searched if fit_offset
bounds = dict(sigma=(0.03, 8.), length_r=(None, 64.), length_theta=(0.03, 6.),
              cos_inc=(float(numpy.cos(numpy.radians(65.))), 1.), dx=(-2., 2.), dy=(-2., 2.))
scales = dict(sigma="log", length_r="log", length_theta="log", cos_inc="linear", pa="periodic", dx="linear", dy="linear")
periods = dict(pa=180.)
abbrev = dict(sigma="s", length_r="lr", length_theta="lt", cos_inc="ci", pa="pa", dx="dx", dy="dy")
fit_offset = False
# Search (stage D): the geometry depends on the shape and vice versa, so rounds of a coarse geometry grid (cos_inc
# 0.45 .. 0.95, pa every 15 deg) at the current shape and the shape grid (axes spaced by 2, grown while posterior mass
# is on a face) at the best geometry, until the geometry repeats; then the joint local grid (3 per axis, joint_steps;
# no growth: re-centred on its best while that is on a face) for the posterior, the intervals and the evidence
shape0 = dict(sigma=0.5, length_r=4., length_theta=0.5)
geometry0 = dict(cos_inc=0.7, pa=0.)        # stage C's provisional geometry (the noise scales barely depend on it)
geometry_axes0 = dict(cos_inc=[0.45 + 0.1*i for i in range(6)], pa=[15.*i for i in range(12)])
shape_axes0 = dict(sigma=[0.25, 0.5, 1., 2.], length_r=[2., 4., 8., 16., 32.], length_theta=[0.125, 0.25, 0.5, 1., 2.])
joint_steps = dict(sigma=2., length_r=2., length_theta=2., cos_inc=0.05, pa=7.5, dx=0.5, dy=0.5)
n_rounds = 2
max_recentre = 3
# DCT baseline: the previous run's choice (order 1, flat mean), for the polar vs DCT evidence comparison
dct_order = 1.
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [0.4, 0.8, 1.6, 3.2, 6.4]
dct_bounds = dict(sigma=(0.03, 8.), length=(0.1, 30.))   # gp_evidence_grid's default bounds
level_sd = 1.               # dex, Normal prior on resolved_level about the reference disc's level
n_iter = 200                # at most this many Levenberg-Marquardt steps per MAP (warm-started; stops when converged)
response_batch = 16         # disc-pixel jvps per batch when building the response

# Outer iteration (stage E): full refit at the chosen hyperparameters, as calibrator-joint / betapic-sparco
# (mapped=True: lax.map over exposures; per_block=False: one preconditioner compile)
precond_kwargs = dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32)
outer_epochs = 2000                   # the previous run's 1000-epoch refits were still improving at the end
n_outer = 2
# after each refit: the joint local grid about the optimum (joint_steps, re-centred at most max_recentre times)

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


class PolarDiscFit(PolarGPResolvedFit):
    """PolarGPResolvedFit (InterpolatedResolvedSource, polar prior on log10 of the disc) with imlup-adi's ResolvedFit
    keys."""
    PARAM_KEYS = PolarGPResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                  "primary_opd": "global", "cold_mask_opd": "exposure",
                                                  "defocus3": "exposure"}


# imlup-adi's stage-1 (intermediate) and stage-2 (params_amp) results
reference = {}
for f in ("intermediate-params.npy", "params_amp.npy"):
    path = f"{ref_dir}/{f}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run imlup-adi.py first (its fit is the reference optics and disc)")
    reference |= numpy.load(path, allow_pickle=True).item()
assert len(reference["resolved"]) == 1, f"one reference disc expected, got {list(reference['resolved'])}"
key, log_ref = next(iter(reference.pop("resolved").items()))
d_ref = 10**np.asarray(log_ref)                                # the L1/TSV disc (the level prior's centre, plots)
assert d_ref.shape == (resolved_wid, resolved_wid)

grid = PolarGrid.build(resolved_wid, n_r=n_r, n_theta=n_theta, cover_inc=cover_inc, nu_r=nu_r, nu_theta=nu_theta)
bounds["length_r"] = (grid.spacing, bounds["length_r"][1])
# the shape's start and grid axes within the bounds (now all set): fail here, not deep in a grid
for k, v in shape0.items():
    assert bounds[k][0] <= v <= bounds[k][1], f"shape0 {k} = {v} outside its bounds {bounds[k]}"
for k, vals in shape_axes0.items():
    keep = [v for v in vals if bounds[k][0] <= v <= bounds[k][1]]
    if len(keep) < len(vals):
        print(f"shape_axes0 {k}: {sorted(set(vals) - set(keep))} outside {bounds[k]} dropped")
    assert keep, f"shape_axes0 {k}: no value within {bounds[k]}"
    shape_axes0[k] = keep
print(f"polar grid {grid.shape}: r_max {grid.r_max:.1f} disc-plane px, spacing {grid.spacing:.2f} px")
level_mu = reference_level(d_ref)
lp = (level_mu, level_sd)
print(f"reference level (dex): {level_mu:.3f}")

# n_share: all four exposures share the target-keyed disc, so each carries 1/4 of its prior
exposures = with_n_share([
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits', PolarDiscFit(spectrum_basis, "F160W", wid=resolved_wid, grid=grid,
                                                                       level_prior=lp),
                       crop=wid, extra_bad=None, hole_margin=0.3)
    for root in roots
])
check_n_share(exposures)
assert exposures[0].fit.get_key(exposures[0], "resolved") == key, "the reference disc's key is not this target's"

# imlup-adi's initial values, then the reference
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
params |= reference

params["resolved"] = {key: np.zeros(grid.shape)}
params["resolved_level"] = {key: np.asarray(level_mu)}
ref_params = params
check_gp_setup(ref_params)
# the reference must be keyed as PolarDiscFit keys its params (a KeyError deep inside a jit is much harder to read)
for exp in exposures:
    for name, v in ref_params.items():
        if isinstance(v, dict):
            assert exp.fit.get_key(exp, name) in v, f"params[{name!r}] has no key {exp.fit.get_key(exp, name)!r}"
model_single = set_array(NICMOSModel(exposures, ref_params, optics, detector))
n = resolved_wid**2


def polar_at(sigma, length_r, length_theta, cos_inc, pa, dx=0., dy=0.):
    return polar_prior(grid, sigma, length_r, length_theta, cos_inc, pa, dx, dy, None, lp)


def at_best(g):
    """(hyperparameters, MAP theta) at the grid's best converged point (gp.gp_best)."""
    hyper = gp_best(g)
    idx = tuple(int(numpy.argmin(numpy.abs(numpy.asarray(g["axes"][k]) - v))) for k, v in hyper.items())
    return hyper, np.asarray(g["thetas"][idx])


def same_cell(a, b):
    return all(numpy.isclose(a[k], b[k]) for k in a)


def joint(center, resp_, beta_, tag, theta0=None):
    """The joint local grid about `center` (3 values per axis, steps joint_steps; pa as a linear axis within +-90 deg of
    the centre, so its cells keep the uniform prior over the period), re-centred on its best converged point while that
    is not the centre, at most max_recentre times (points shared with an earlier grid are reloaded, not recomputed)."""
    for k in range(max_recentre + 1):
        sc = {n: ("linear" if scales[n] == "periodic" else scales[n]) for n in center}
        b = {n: ((center[n] - periods[n]/2, center[n] + periods[n]/2) if scales[n] == "periodic" else bounds[n])
             for n in center}
        g = gp_evidence_grid_nd(resp_, local_axes(center, joint_steps, sc), polar_at, beta_, outdir=f"{out}/grid",
                                tag=tag, abbrev=abbrev, bounds=b, scales=sc, n_iter=n_iter, theta0=theta0, max_growth=0)
        best = gp_best(g)
        if same_cell(best, center):
            return g
        center = best
    print(f"WARNING: the joint grid {tag} still moved after {max_recentre} re-centrings")
    return g


def pa_interval(q):
    """pa quantiles shifted by a multiple of 180 so that the median is in [0, 180) (monotone, possibly beyond 180)."""
    q = numpy.asarray(q, dtype=float)
    return q - 180.*numpy.floor(q[1]/180.)

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
# ---------------------------------------------------------------- C: gradient check, noise scales (provisional prior)
mid = shape0 | geometry0
prior0 = polar_at(**mid)

with timed("C: noise"):
    npath = f"{out}/noise-polar-{fingerprint(resp_fp, *mid.values(), level_mu, level_sd, n_iter, grid.digest(), code_fp)}.npz"
    if not os.path.exists(npath):
        theta1, hist1 = gp_map(resp, prior0, n_iter=n_iter)
        ev1 = to_host(gp_log_evidence(resp, prior0, theta1))

        # The full model's gradient in z at the surrogate MAP (optics fixed) should vanish, relative to its value at
        # z = 0; and its loss should equal the surrogate's (including the prior)
        exps0 = set_hyper(exposures, **mid, level_prior=lp)
        batch0 = make_batch(exps0, model_single)

        def full_grad(z, level):
            sub = set_array({"resolved": {key: z}, "resolved_level": {key: level}})
            return _loss_grad(sub, jax.tree.map(np.ones_like, sub), batch0, model_single)

        loss_0, g_0 = full_grad(np.zeros(grid.shape), np.asarray(level_mu))
        z1, l1 = prior0.params(theta1)
        loss_1, g_1 = full_grad(z1, l1)
        # the full model's z-gradient against the surrogate's at the same point (both include the prior), relative to
        # the gradient at z = 0: tests the surrogate itself, whatever the MAP's convergence
        G1, b1, _ = gp._weighted(resp, np.ones(len(resp.keys)))
        g_sur = gp._grad(G1, b1, prior0.distribution(theta1).ravel(), prior0, theta1)[:-1].reshape(grid.shape)
        g_ratio = float(np.linalg.norm(g_1["resolved"][key] - g_sur)/np.linalg.norm(g_0["resolved"][key]))
        loss_sur = float(ev1["nll"].sum() + ev1["prior"])
        print(f"|full - surrogate grad z| at the surrogate MAP / |full grad z| at z = 0: {g_ratio:.2e} (Newton decrement "
              f"{float(ev1['newton_decrement']):.1e}); full loss {float(loss_1):.6f} vs surrogate {loss_sur:.6f}")

        s, theta_s, ev_s = gp_error_scale(resp, prior0, theta1, verbose=True)
        _save_npz(npath, s=s, theta1=theta1, theta_s=theta_s, g_ratio=g_ratio, loss_full=loss_1,
                  loss_surrogate=loss_sur, chi2_1=ev1["chi2"], n_good=resp.n_good, gamma_e=ev_s["gamma_e"])
    noise = numpy.load(npath)
    s_noise = noise["s"]
    print("noise scales per exposure:", dict(zip(resp.keys, numpy.round(s_noise, 4))))
    if noise["g_ratio"] > 1e-3:
        print(f"WARNING: the full-model and surrogate gradients differ ({float(noise['g_ratio']):.1e} of the gradient "
              "at z = 0): the surrogate is not the full model")
    if numpy.any(s_noise > 3):
        print("WARNING: s >> 1: the model does not fit those exposures; rescaling the errors hides a model failure")
beta = 1/np.asarray(s_noise)**2

# %%
# ---------------------------------------------------------------- D: evidence grids
with timed("D: grids"):
    fp_d = fingerprint(resp_fp, beta, level_mu, level_sd, n_iter, grid.digest(), code_fp)
    shape, geo, rounds = dict(shape0), None, []
    for rnd in range(1, n_rounds + 1):
        # the geometry at the current shape: the full coarse grid (periodic pa), its best converged point
        grid_geo = gp_evidence_grid_nd(resp, geometry_axes0, lambda cos_inc, pa: polar_at(**shape, cos_inc=cos_inc, pa=pa),
                                       beta, outdir=f"{out}/grid", tag=f"geo{rnd}_" + fingerprint(fp_d, *shape.values()),
                                       abbrev=abbrev, bounds=bounds, scales=scales, periods=periods, n_iter=n_iter)
        geo_new = gp_best(grid_geo)
        print(f"round {rnd}: geometry {geo_new} (inc {numpy.degrees(numpy.arccos(geo_new['cos_inc'])):.1f} deg) at "
              f"shape {shape}")
        if geo is not None and same_cell(geo_new, geo):
            break
        geo = geo_new
        # the shape at that geometry
        grid_shape = gp_evidence_grid_nd(resp, shape_axes0, lambda **h: polar_at(**h, **geo), beta, outdir=f"{out}/grid",
                                         tag=f"shape{rnd}_" + fingerprint(fp_d, *geo.values()), abbrev=abbrev,
                                         bounds=bounds, n_iter=n_iter, theta0=noise["theta_s"])
        shape = gp_best(grid_shape)
        rounds.append(dict(geometry=geo, shape=shape))
        print(f"round {rnd}: shape {shape}")
    # the joint local grid about (shape, geometry): the posterior over all hyperparameters and the evidence
    center = shape | geo | (dict(dx=0., dy=0.) if fit_offset else {})
    grid_polar = joint(center, resp, beta, "joint_" + fp_d)
    post_polar = gp_hyper_posterior(grid_polar)
    print(f"joint best {gp_best(grid_polar)}")
    tag_dct = f"o{dct_order:g}_t0_" + fingerprint(resp_fp, beta, level_mu, level_sd, n_iter, code_fp)
    grid_dct = gp_evidence_grid(resp, sigmas0, lengths0, dct_order, None, lp, beta, outdir=f"{out}/grid", tag=tag_dct,
                                n_iter=n_iter, bounds=(dct_bounds["sigma"], dct_bounds["length"]))
    grid_dct = grid_dct | dict(names=["sigma", "length"], axes=dict(sigma=grid_dct["sigmas"], length=grid_dct["lengths"]))
    post_dct = gp_hyper_posterior(grid_dct, box=(sigmas0, lengths0), bounds=dct_bounds)
    for name, g, post in (("polar", grid_polar, post_polar), (f"DCT order {dct_order:g}", grid_dct, post_dct)):
        best = gp_best(g)
        _save_npz(f"{out}/grid-{name.split()[0].lower()}.npz", logZ=g["logZ"], thetas=g["thetas"],
                  converged=g["converged"], weights=post["weights"],
                  **{f"axis_{k}": v for k, v in g["axes"].items()}, **{f"q_{k}": v for k, v in post["quantiles"].items()},
                  log_evidence=post["log_evidence"], log_evidence_bounds=post["log_evidence_bounds"],
                  edge_weight=post["edge_weight"])
        print(f"[{name}] log evidence (log-uniform on the bounds) {post['log_evidence_bounds']:.2f}, max log Z "
              f"{g['logZ'].max():.2f}; "
              f"best {best}; 16/50/84% " + "; ".join(f"{k} {numpy.round(q, 3)}" for k, q in post["quantiles"].items())
              + f"; edge weight {post['edge_weight']:.3f}; unconverged {int((~g['converged'].astype(bool)).sum())} "
              f"(dropped; at most {post['unconverged_weight']:.3f} of the weight)")
    valid = max(post_polar["edge_weight"], post_dct["edge_weight"]) < 0.1
    print(("" if valid else "NOT VALID (posterior weight on a grid's edge): ") +
          f"polar - DCT log evidence: {post_polar['log_evidence_bounds'] - post_dct['log_evidence_bounds']:.1f} nats "
          f"(each under its uniform hyperprior on fixed bounds, the polar one's geometry included; the same noise "
          f"scales; valid if both edge "
          f"weights are small: {post_polar['edge_weight']:.3f}, {post_dct['edge_weight']:.3f}; max log Z difference "
          f"{grid_polar['logZ'].max() - grid_dct['logZ'].max():.1f}; the Laplace bias is not corrected)")

# evidence surfaces relative to each grid's best point: the coarse geometry map, and for each pair of the joint grid's
# axes the maximum over the others; the star marks the best converged point (grid["best"], as gp_best / at_best),
# crosses the unconverged ones
def evidence_panel(ax, img, yv, xv, yname, xname, best, title, clip=50., unconverged=None):
    im = ax.imshow(img if clip is None else numpy.clip(img, -clip, 0), cmap="viridis", origin="lower", aspect="auto")
    ax.set_yticks(range(len(yv)), [f"{v:.3g}" for v in yv])
    ax.set_xticks(range(len(xv)), [f"{v:.3g}" for v in xv], rotation=45)
    ax.set(ylabel=yname, xlabel=xname, title=title)
    if unconverged is not None:
        yy, xx = numpy.nonzero(unconverged)
        ax.plot(xx, yy, "wx", ms=10)
    ax.plot(best[1], best[0], "r*", ms=20)
    plt.colorbar(im, ax=ax, label="log Z - max")


names = grid_polar["names"]
pairs = list(itertools.combinations(range(len(names)), 2))
ncol = 4
fig, axs = plt.subplots((len(pairs) + 1 + ncol - 1)//ncol, ncol, figsize=(10*ncol, 9*((len(pairs) + ncol)//ncol)),
                        squeeze=False)
axs = axs.ravel()
evidence_panel(axs[0], grid_geo["logZ"] - grid_geo["logZ"].max(), grid_geo["axes"]["cos_inc"], grid_geo["axes"]["pa"],
               "cos_inc", "pa (deg)", grid_geo["best"], "coarse geometry (last round)",
               clip=float(numpy.percentile(grid_geo["logZ"].max() - grid_geo["logZ"], 75)),
               unconverged=~grid_geo["converged"].astype(bool))
for ax, (a, b) in zip(axs[1:], pairs):
    other = tuple(i for i in range(len(names)) if i not in (a, b))
    img = grid_polar["logZ"].max(axis=other) - grid_polar["logZ"].max()
    evidence_panel(ax, img, grid_polar["axes"][names[a]], grid_polar["axes"][names[b]], names[a], names[b],
                   (grid_polar["best"][a], grid_polar["best"][b]), "joint grid, max over the others",
                   unconverged=(~grid_polar["converged"].astype(bool)).all(axis=other))
for ax in axs[1 + len(pairs):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{out}/evidence-polar.png")
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

hyper_b, theta_b = at_best(grid_polar)
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=grid_polar,
               theta=theta_b, stage="D")
outer_converged = False
for it in range(1, n_outer + 1):
    prior_b = polar_at(**hyper_b)
    h = fingerprint(it, *hyper_b.values(), current["s"], current["resp_fp"], current["theta"], outer_epochs, lp,
                    repr(sorted(precond_kwargs.items())), repr(sorted(things_gp)), grid.digest(), code_fp)
    with timed(f"E{it}: refit"):
        exps_b = scale_errors(set_hyper(exposures, **hyper_b, level_prior=lp), current["s"])
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
        tag = f"outer{it}_polar_" + fingerprint(resp_new_fp, beta_new, lp, n_iter, *hyper_b.values(), grid.digest(),
                                                 code_fp)
        grid_new = joint(hyper_b, resp_new, beta_new, tag, theta0=theta_new)
        hyper_new, theta_new_b = at_best(grid_new)
        print(f"outer {it}: s {numpy.round(numpy.asarray(s_new), 4)}; best {hyper_new} (was {hyper_b})")
    moved = not same_cell(hyper_new, hyper_b)
    hyper_b = hyper_new    # pa is not wrapped here: pa and pa + 180 give the same prior but rotated latents
    current = dict(params=p_new, resp=resp_new, resp_fp=resp_new_fp, beta=beta_new, s=numpy.asarray(s_new),
                   grid=grid_new, theta=theta_new_b, stage=f"E{it}")
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
    prior_f = polar_at(**hyper_b)
    theta_f = current["theta"]
    ev_f = to_host(gp_log_evidence(resp_f, prior_f, theta_f, beta_f))
    lap = gp_laplace(resp_f, prior_f, theta_f, beta_f)
    log_map = numpy.asarray(prior_f.log10_image(theta_f))
    log_std = numpy.asarray(gp_log10_std(prior_f, lap))
    ds = jax.vmap(prior_f.distribution)(gp_laplace_samples(theta_f, lap, jr.PRNGKey(1), n=200))
    d_mean, d_std = numpy.asarray(ds.mean(0)), numpy.asarray(ds.std(0))

    # posterior mixture over the final grid: converged cells drawn in proportion to their evidence weight
    post_f = gp_hyper_posterior(grid_f)
    draws, n_mix = [], 400
    for cell, w in numpy.ndenumerate(post_f["weights"]):
        m = int(round(n_mix*w))
        if m == 0 or not bool(grid_f["converged"][cell]):   # an unconverged MAP is not a posterior mode
            continue
        prior_c = polar_at(**{k: grid_f["axes"][k][i] for k, i in zip(grid_f["names"], cell)})
        th = np.asarray(grid_f["thetas"][cell])
        draws.append(jax.vmap(prior_c.distribution)(
            gp_laplace_samples(th, gp_laplace(resp_f, prior_c, th, beta_f), jr.PRNGKey(10 + len(draws)), n=m)))
    if not draws:   # every cell below 1 / (2 n_mix): the MAP cell alone
        draws.append(ds)
    mix = np.concatenate(draws)
    if post_f["edge_weight"] > 0.1:
        print(f"WARNING: {post_f['edge_weight']:.2f} of the final grid's posterior weight is on its edge: the mixture and "
              "the summary's intervals are truncated (raise max_recentre or widen joint_steps)")
    mix_mean, mix_std = numpy.asarray(mix.mean(0)), numpy.asarray(mix.std(0))
    diag = gp_polar_diagnostic(prior_f, theta_f, lap)

    numpy.save(f"{out}/gp-disc-log10.npy", log_map)
    numpy.save(f"{out}/gp-disc-log10-std.npy", log_std)
    numpy.save(f"{out}/gp-mixture-mean.npy", mix_mean)
    numpy.save(f"{out}/gp-mixture-std.npy", mix_std)
    numpy.savez(f"{out}/polar-diagnostic.npz", **diag)
    z_f, l_f = prior_f.params(theta_f)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    # discs (sqrt stretch): L1/TSV reference, polar MAP, posterior-mixture mean, uncertainty, SNR
    from matplotlib.colors import PowerNorm
    pitch = 0.0432*2
    ext = numpy.array([-1, 1, -1, 1])*resolved_wid*pitch/2
    sq = lambda im: dict(norm=PowerNorm(0.5, 0, numpy.percentile(im, 99.5)))   # each panel its own stretch
    panels = [(numpy.asarray(d_ref), "L1/TSV reference", sq(numpy.asarray(d_ref))),
              (10**log_map, "polar GP MAP", sq(10**log_map)),
              (mix_mean, "polar GP posterior-mixture mean", sq(10**log_map)),   # the MAP's stretch: robust to outliers
              (log_std, "Laplace std of log10 I (dex)", dict(cmap="viridis")),
              (d_mean/d_std, "Laplace mean / std", dict(cmap="viridis"))]
    fig, axs = plt.subplots(1, len(panels), figsize=(12*len(panels), 11))
    for ax, (im, title, kw) in zip(axs, panels):
        c = ax.imshow(im, extent=ext, **({"cmap": "inferno"} | kw))
        ax.set_title(title)
        plt.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out}/discs.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(33, 9))
    axs[0].errorbar(diag["m"], diag["power_m"], diag["power_m_err"], fmt="o")
    axs[0].axhline(1, color="k", ls="--")
    axs[0].set(xlabel="azimuthal order |m|", ylabel="<|z_m|^2> (prior: 1)", title="Power per order")
    axs[1].plot(diag["m"], diag["var_m"], "o")
    axs[1].set(xlabel="azimuthal order |m|", ylabel="posterior variance (prior: 1)", title="Data constraint per order")
    axs[2].plot(diag["r"]*pitch, diag["power_r"], "o", label="<z^2>")
    axs[2].plot(diag["r"]*pitch, diag["var_r"], "s", label="posterior variance")
    axs[2].axhline(1, color="k", ls="--")
    axs[2].set(xlabel='disc-plane radius (")', title="Per radial latent row")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/polar-diagnostic.png")
    plt.close(fig)

    # prior draws at the chosen hyperparameters (level at its prior mean), with the grid's geometry
    prior_draws = jr.normal(jr.PRNGKey(2), (4, prior_f.n_latent + 1)).at[:, -1].set(0.)
    fig, axs = plt.subplots(1, 4, figsize=(44, 11))
    for ax, th in zip(axs, prior_draws):
        c = ax.imshow(prior_f.log10_image(th), extent=ext)
        plt.colorbar(c, ax=ax)
    fig.suptitle("prior draws (log10): " + ", ".join(f"{k} {v:.3g}" for k, v in hyper_b.items()))
    fig.tight_layout()
    fig.savefig(f"{out}/prior-draws.png")
    plt.close(fig)

    exps_f = scale_errors(set_hyper(exposures, **hyper_b, level_prior=lp), current["s"])
    model_f = set_array(ModelParams(p_final).inject(model_single))
    plot_comparison_detailed(model_f, ModelParams(set_array(p_final)), exps_f, quadrature=False, wf_size=wf_wid,
                             percentile=99, save=f"{out}/comparison")

    n_good = numpy.asarray(resp_f.n_good)
    summary = dict(
        prior="polar",
        # the geometry's posterior (final grid): inc from cos_inc's quantiles (reversed), pa modulo 180 in the disc
        # grid's frame; the coarse (stage C) and refined (D1) searches' bests; the fraction of disc pixels beyond r_max
        geometry=dict(inc_deg=float(numpy.degrees(numpy.arccos(hyper_b["cos_inc"]))), pa_deg=hyper_b["pa"] % 180.,
                      inc_q=list(map(float, numpy.degrees(numpy.arccos(numpy.clip(
                          post_f["quantiles"]["cos_inc"][::-1], -1., 1.))))),
                      pa_q=list(map(float, pa_interval(post_f["quantiles"]["pa"]))),
                      search_rounds=rounds, frame="disc grid (gp.PolarGrid.coordinates)",
                      fraction_beyond_r_max=float(grid.fraction_outside(prior_f.geometry))),
        polar_grid=dict(n_r=n_r, n_theta=n_theta, nu_r=nu_r, nu_theta=nu_theta, spacing_px=grid.spacing,
                        r_max_px=grid.r_max, cover_inc=cover_inc),
        hyper=hyper_b, final_stage=current["stage"], outer_converged=outer_converged, data_dir=ddir,
        noise_scale=dict(zip(resp_f.keys, map(float, current["s"]))),
        # chi2 / n_good with the errors as given (~ s^2: the goodness of fit), and with the rescaled errors
        # (~ 1 - gamma_e / n_good by construction of the noise scale, not a test of the fit)
        chi2_per_pixel_raw=dict(zip(resp_f.keys, map(float, ev_f["chi2"]/n_good))),
        chi2_per_pixel_rescaled=dict(zip(resp_f.keys, map(float, ev_f["chi2"]*numpy.asarray(beta_f)/n_good))),
        gamma_per_exposure=dict(zip(resp_f.keys, map(float, ev_f["gamma_e"]))),
        logZ=float(ev_f["logZ"]), gamma=float(ev_f["gamma"]), newton_decrement=float(ev_f["newton_decrement"]),
        final_grid=dict(quantiles={k: list(map(float, q)) for k, q in post_f["quantiles"].items()},
                        edge_weight=post_f["edge_weight"],
                        unconverged=int((~grid_f["converged"].astype(bool)).sum())),
        stage_d=dict(quantiles={k: list(map(float, q)) for k, q in post_polar["quantiles"].items()},
                     edge_weight=post_polar["edge_weight"], log_evidence_bounds=post_polar["log_evidence_bounds"],
                     shape_grid_best=shape),
        # at the reference optics (stage D), each under a log-uniform hyperprior on its fixed bounds, with the same
        # noise scales; the Laplace approximation's bias (gp.py) is not corrected
        polar_minus_dct_log_evidence=post_polar["log_evidence_bounds"] - post_dct["log_evidence_bounds"],
        evidence_edge_weights=dict(polar=post_polar["edge_weight"], dct=post_dct["edge_weight"]),
        dct_baseline=dict(order=dct_order, log_evidence_bounds=post_dct["log_evidence_bounds"],
                          best=gp_best(grid_dct)),
        verify_max_abs=float(v["abs_diff"].max()), verify_max_rel_change=float(v["rel_change"].max()),
        gradient_ratio=float(noise["g_ratio"]),
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
