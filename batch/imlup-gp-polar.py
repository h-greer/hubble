# %%
"""imlup-gp-polar.py (imlup-gp-polar.sh): IM Lup (SZ-82, NIC2 F160W; 2 rolls x 2 exposures) with the two-component
Gaussian-process disc prior

    log10 I = level + mean + field_polar(z_p) + field_dct(z_d)

a smooth polar (multipole) field, low azimuthal order (hard cutoff m_max, length_theta >= 1 rad), PLUS a stationary DCT
Matern field for the small-scale / non-axisymmetric structure (largely PSF residual). Phases 0-2 of notebooks/gp-plan.md,
with the hyperparameters chosen by the Laplace evidence. The polar field is after virgil's MultipoleField; the prior
(gp.SumPrior) and the fit (gp.PolarDCTResolvedFit) are in gp.py. This replaces the polar-only version of this script
(its prior lost to the DCT prior by 292 nats and made radial spokes); the polar-only run's outputs remain in imlup-gp/
(read, never written, by this script). Needs imlup-adi's results (imlup-adi/intermediate-params.npy and params_amp.npy)
as the reference optics, star and L1/TSV disc. Stages, saving to imlup-gp-polardct/ (response.npz, verify-*.npz and, if
present, the polar run's stage-C noise scales are read from imlup-gp/, so every evidence here, the DCT baseline's and the
polar-only run's are on the same response and noise scales):
  A  exposures (PolarDCTResolvedFit on an InterpolatedResolvedSource, 60 px box; the polar grid fixed in the disc plane),
     model, reference params (ONE "resolved" parameter: the flat latents (z_p, z_d))
  B  LinearResponse at the reference optics (imlup-gp/response.npz; built here only if missing), checked against the
     full model (verify)
  C  at a provisional prior (the previous runs' optima): the full-model gradient check of the sum-prior surrogate; the
     noise scales of the polar run are kept (so the evidences are comparable), the sum prior's own are computed and
     reported (noise-pd-*.npz)
  D  the search (below), then the DCT-only baseline grid and the polar-only run's evidence (grid/, grid-*.npz)
  E  outer iteration: hybrid refit of optics + disc at the chosen hyperparameters (a short first-order stage, then one
     L-BFGS call, as batch/calibrator-joint.py; the disc is whitened with the gn_probe C too), response rebuilt, local
     grid (outer*)
  F  products: MAP disc and its two components, Laplace std / SNR, posterior-mixture mean, component correlation,
     polar and DCT diagnostics, comparisons (summary.json)
Search (D): start from the DCT-only best (sigma_d, length_d) and the polar-only run's geometry / length_r (read from
imlup-gp/ with fallbacks). Block rounds (at most n_rounds): polar shape (sigma_p, length_r, length_theta; 3^3 local grid,
may grow) -> DCT (sigma_d, length_d; 3^2) -> geometry (cos_inc, pa; 3^2), until all three repeat; then the joint local
grid over the shape axes (sigma_p, length_r, length_theta, sigma_d, length_d: 3^5 = 243) with the geometry fixed
(joint_geometry = False) or over all seven (3^7 = 2187, joint_geometry = True, the default: the evidence is then
marginal over the geometry as the polar-only run's was, which the sum - polar comparison needs; the outer loop (E) then
holds the geometry fixed at the joint's best).
The geometry (inclination, position angle, optionally the centre) is a hyperparameter like the others, chosen by the
Laplace evidence on the linear response; nothing is taken from a fit to a reference image or from the literature.
Identifiability (two fields can trade): the polar part is kept smooth by length_theta >= 1 rad (bound), sigma_p's prior
and the cutoff m <= m_max (mainly a saving of latents, see the gp.py section); the DCT's length_d ranges as widely as
the DCT-only run's (up to 60 px; that run chose 25.6), so sum contains DCT-only (sigma_p -> its floor, same length_d
range); the Laplace posterior correlation between the components is reported (sum_components).
Resumable: every result is saved atomically under a name carrying a fingerprint of what it depends on (the response,
noise scales, prior settings, gp.py), and reloaded on a rerun; anything stale is recomputed, never reused. Delete
imlup-gp/response.npz (and response-outer*.npz) to rebuild the responses.
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
import glob
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
from gp import _save_npz   # gp_evidence_grid_nd, PolarGrid, ... come with `import *`

out = "imlup-gp-polardct"
prev = "imlup-gp"        # the previous runs (DCT, polar-only), read only: response.npz, verify, noise scales, grids
ref_dir = "imlup-adi"
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


with open("../gp.py", "rb") as f:
    code_fp = hashlib.sha256(f.read()).hexdigest()[:12]   # results depend on gp.py's maths; gp.py changed (for both new priors: the
                                                          # multi-kernel DCT and the polar + DCT sum), so no earlier-hash grids are
                                                          # reused (only the response / verify files and beta are shared)

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
resolved_wid = 60
roots = ["n8zu11epq", "n8zu11eqq", "n8zu12exq", "n8zu12eyq"]

# Polar part: as imlup-gp-polar.py's grid (49 x 128 nodes), but the latents are the low-m real-Fourier coordinates of each
# ring, orders 0..m_max (2 m_max + 1 per ring: m_max = 4: 49 x 9 = 441 polar latents instead of 6272; None: all 6272).
# Total latents with the DCT's 60 x 60 = 3600: 4041 (m_max 4), 4433 (m_max 8), 9872 (None)
n_r, n_theta, cover_inc = 49, 128, 60.
nu_r = nu_theta = 1.5
m_max = 4
# Hyperparameters (7 searched + order_d fixed). Hyperprior bounds are fixed here, before the data:
#   polar: sigma_p (dex, rms of the retained polar field), length_r (disc-plane px; lower bound: the node spacing),
#   length_theta (rad; LOWER BOUND 1: this, with sigma_p's prior, keeps the polar part smooth (no spokes);
#   the m cutoff mainly saves latents), cos_inc, pa, (dx, dy) as before;
#   DCT: sigma_d (dex), length_d (px; up to 60, as wide as the DCT-only run's range: it chose 25.6; no tighter cap, so that
#   sum contains DCT-only; identifiability comes from the polar part's bounds)
bounds = dict(sigma_p=(0.03, 4.), length_r=(None, 64.), length_theta=(1., 6.),
              cos_inc=(float(numpy.cos(numpy.radians(65.))), 1.), dx=(-2., 2.), dy=(-2., 2.),
              sigma_d=(0.03, 4.), length_d=(0.1, 60.))
scales = dict(sigma_p="log", length_r="log", length_theta="log", cos_inc="linear", pa="periodic", dx="linear", dy="linear",
              sigma_d="log", length_d="log")
periods = dict(pa=180.)
abbrev = dict(sigma_p="sp", length_r="lr", length_theta="lt", cos_inc="ci", pa="pa", dx="dx", dy="dy", sigma_d="sd",
              length_d="ld")
fit_offset = False
dct_order = 1.              # the previous DCT run's order; fixed (not searched) for the DCT part
shape_names = ("sigma_p", "length_r", "length_theta", "sigma_d", "length_d")
geometry_names = ("cos_inc", "pa")
joint_geometry = True       # the stage-D joint grid also spans cos_inc, pa: 3^7 = 2187 points (+ re-centrings), so that the
                            # evidence is marginal over the geometry like the polar-only run's. Cost: at ~0.5-1 s/point on an
                            # A100 about 20-40 min (+ up to ~3 re-centrings, each ~700-1500 new points); at 3 s/point 2 h.
                            # False: 243 points, geometry conditional on the round search (profile, not marginal)
# Starting point: the previous runs' optima (read from prev/, with fallbacks), see previous_starts()
fallback = dict(sigma_d=0.5, length_d=1.6, length_r=8., cos_inc=0.7, pa=0.)
poly_start = dict(sigma_p=0.3, length_theta=2.)      # the polar-smooth start (length_r from the polar run)
# steps of the local grids (ratio for log axes, difference for linear): 3 values per axis
joint_steps = dict(sigma_p=2., length_r=2., length_theta=2., cos_inc=0.05, pa=7.5, dx=0.5, dy=0.5, sigma_d=2., length_d=1.6)
geo_steps = dict(cos_inc=0.1, pa=15.)
n_rounds = 2
max_recentre = 3
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]   # the box reaches the previous run's optimum (25.6)
dct_bounds = dict(sigma=(0.03, 8.), length=bounds["length_d"])   # the baseline over the SAME length range as the sum's DCT part (the previous run's 30 px cap was below
# where it wanted to go): its log evidence is not the previous run's; rerun here (cheap)
level_sd = 1.
n_iter = 200
response_batch = 16

precond_kwargs = dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32)
# The refit is a hybrid, as batch/calibrator-joint.py: (a) a short first-order stage (the old adam/sgd schedule, but
# without the delayed adam(2e-2, 50) kick on primary_opd, whose first Adam step threw the loss up by ~250-300 at epoch
# 50; small Adam rates from step 0 instead) to move the disc and optics off the surrogate-MAP start, then (b) ONE
# optimise_optimistix call (L-BFGS in u = (x - x0)/sqrt(C), BestSoFarMinimiser: never accepts a worse point) with C =
# gn_probe for ALL parameters including the disc (the identity initial inverse Hessian of optimistix's LBFGS needs
# unit-scale coordinates). The old 2000-epoch first-order run was still improving at the end (slow power-law tail).
fo_epochs = 200         # first-order epochs (~1.15 s each at production size)
fo_lr_opd = 5e-3        # Adam rate on primary_opd from step 0 (the old one: 2e-2 from epoch 50)
fo_lr_disc = 3e-3       # Adam rate on the disc latents z, C = 1 (the old one: 1e-2, which raised the loss by ~470)
optx_max_steps = 800    # L-BFGS iterations, accepted AND rejected line-search trials (each costs one loss+grad)
n_outer = 2

if os.path.isdir('../data/data/'):
    ddir = '../data/data/'
else:
    ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'
    print(f"WARNING: ../data/data/ not found; using {ddir}")
print(f"data directory: {ddir}")

# %%
# ---------------------------------------------------------------- A: exposures, model, reference
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20, pad=pad)
detector = NICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_F160W_{nwavels}.npy")[:, :npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))


class PolarDCTDiscFit(PolarDCTResolvedFit):
    """PolarDCTResolvedFit (InterpolatedResolvedSource, polar + DCT prior on log10 of the disc; ONE "resolved" parameter
    holding the flat (z_p, z_d)) with imlup-adi's ResolvedFit keys."""
    PARAM_KEYS = PolarDCTResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
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
n_p = grid.shape[0]*(n_theta if m_max is None else 2*m_max + 1)
n_latent = n_p + resolved_wid**2
print(f"polar grid {grid.shape}: r_max {grid.r_max:.1f} disc-plane px, spacing {grid.spacing:.2f} px; m_max {m_max}: "
      f"{n_p} polar + {resolved_wid**2} DCT = {n_latent} latents")
level_mu = reference_level(d_ref)
lp = (level_mu, level_sd)
print(f"reference level (dex): {level_mu:.3f}")

# n_share: all four exposures share the target-keyed disc, so each carries 1/4 of its prior
exposures = with_n_share([
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits', PolarDCTDiscFit(spectrum_basis, "F160W", wid=resolved_wid,
                                                                          grid=grid, m_max=m_max, order_d=dct_order,
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

params["resolved"] = {key: np.zeros(n_latent)}   # flat (z_p, z_d)
params["resolved_level"] = {key: np.asarray(level_mu)}
ref_params = params
check_gp_setup(ref_params)
# the reference must be keyed as PolarDCTDiscFit keys its params (a KeyError deep inside a jit is much harder to read)
for exp in exposures:
    for name, v in ref_params.items():
        if isinstance(v, dict):
            assert exp.fit.get_key(exp, name) in v, f"params[{name!r}] has no key {exp.fit.get_key(exp, name)!r}"
model_single = set_array(NICMOSModel(exposures, ref_params, optics, detector))
n = resolved_wid**2


def previous_starts():
    """The previous runs' optima: the DCT-only best (sigma_d, length_d) from prev/grid-dct.npz (gp_best's rule) and the polar
    run's length_r, cos_inc, pa from prev/summary.json (its final hyperparameters); the fallbacks if either is missing."""
    s = dict(fallback)
    try:
        g = numpy.load(f"{prev}/grid-dct.npz")
        logZ = numpy.where(g["converged"].astype(bool), g["logZ"], -numpy.inf)
        i, j = numpy.unravel_index(numpy.argmax(logZ), logZ.shape)
        s |= dict(sigma_d=float(g["axis_sigma"][i]), length_d=float(g["axis_length"][j]))
    except Exception as e:      # missing / differently shaped: keep the fallbacks
        print(f"previous DCT best not read ({type(e).__name__}: {e}); fallbacks for sigma_d, length_d")
    try:
        h = json.load(open(f"{prev}/summary.json"))["hyper"]
        s |= {k: float(h[k]) for k in ("length_r", "cos_inc", "pa")}
    except Exception as e:
        print(f"previous polar geometry not read ({type(e).__name__}: {e}); fallbacks for length_r, cos_inc, pa")
    s["cos_inc"] = min(max(s["cos_inc"], bounds["cos_inc"][0]), bounds["cos_inc"][1])
    s["length_d"] = min(max(s["length_d"], bounds["length_d"][0]), bounds["length_d"][1])
    s["length_r"] = min(max(s["length_r"], bounds["length_r"][0]), bounds["length_r"][1])
    return s


start = previous_starts() | poly_start
print("starting point:", start)
fixed = {}                 # hyperparameters held fixed in sum_at (the geometry while it is not in the joint grid)


def sum_at(sigma_p, length_r, length_theta, sigma_d, length_d, cos_inc, pa, dx=0., dy=0.):
    return sum_prior(grid, m_max, sigma_p, length_r, length_theta, cos_inc, pa, sigma_d, length_d, dct_order, dx, dy, None,
                     lp)


def make_at(**fix):
    """sum_at with some hyperparameters fixed: the grid engines call make_prior(**searched)."""
    return lambda **h: sum_at(**(fix | h))


def at_best(g):
    """(searched hyperparameters, MAP theta) at the grid's best converged point (gp.gp_best)."""
    hyper = gp_best(g)
    idx = tuple(int(numpy.argmin(numpy.abs(numpy.asarray(g["axes"][k]) - v))) for k, v in hyper.items())
    return hyper, np.asarray(g["thetas"][idx])


def same_cell(a, b):
    return all(numpy.isclose(a[k], b[k]) for k in a)


def joint(center, resp_, beta_, tag, theta0=None, fix=None):
    """The joint local grid about `center` (3 values per axis, steps joint_steps; pa as a linear axis within +-90 deg of
    the centre, so its cells keep the uniform prior over the period), re-centred on its best converged point while that
    is not the centre, at most max_recentre times (points shared with an earlier grid are reloaded, not recomputed)."""
    for k in range(max_recentre + 1):
        sc = {n: ("linear" if scales[n] == "periodic" else scales[n]) for n in center}
        b = {n: ((center[n] - periods[n]/2, center[n] + periods[n]/2) if scales[n] == "periodic" else bounds[n])
             for n in center}
        g = gp_evidence_grid_nd(resp_, local_axes(center, joint_steps, sc), make_at(**(fix or {})), beta_, outdir=f"{out}/grid",
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
    if os.path.exists(f"{prev}/response.npz"):     # pixel space, independent of the prior: shared with the DCT / polar runs
        resp = LinearResponse.load(f"{prev}/response.npz")
    elif os.path.exists(f"{out}/response.npz"):
        resp = LinearResponse.load(f"{out}/response.npz")
    else:
        resp = build_linear_response(model_single, exposures, batch_size=response_batch)
        resp.save(f"{out}/response.npz")
    resp_fp = response_fingerprint(resp)
print(f"response ready (fingerprint {resp_fp})")

with timed("B: verify"):
    vpath = f"{prev}/verify-{resp_fp}.npz" if os.path.exists(f"{prev}/verify-{resp_fp}.npz") else f"{out}/verify-{resp_fp}.npz"
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
mid = start | dict(cos_inc=start["cos_inc"], pa=start["pa"])
prior0 = sum_at(**mid)

with timed("C: noise"):
    npath = f"{out}/noise-pd-{fingerprint(resp_fp, *mid.values(), level_mu, level_sd, n_iter, grid.digest(), m_max, code_fp)}.npz"
    if not os.path.exists(npath):
        theta1, hist1 = gp_map(resp, prior0, n_iter=n_iter)
        ev1 = to_host(gp_log_evidence(resp, prior0, theta1))

        # The full model's gradient in z at the surrogate MAP (optics fixed) should vanish, relative to its value at
        # z = 0; and its loss should equal the surrogate's (including the prior)
        exps0 = set_hyper(exposures, **mid, level_prior=lp)   # sigma_p ... length_d, cos_inc, pa: PolarDCTResolvedFit.HYPER
        batch0 = make_batch(exps0, model_single)

        def full_grad(z, level):
            sub = set_array({"resolved": {key: z}, "resolved_level": {key: level}})
            return _loss_grad(sub, jax.tree.map(np.ones_like, sub), batch0, model_single)

        loss_0, g_0 = full_grad(np.zeros(n_latent), np.asarray(level_mu))
        z1, l1 = prior0.params(theta1)
        loss_1, g_1 = full_grad(z1, l1)
        # the full model's z-gradient against the surrogate's at the same point (both include the prior), relative to
        # the gradient at z = 0: tests the surrogate itself, whatever the MAP's convergence
        G1, b1, _ = gp._weighted(resp, np.ones(len(resp.keys)))
        g_sur = gp._grad(G1, b1, prior0.distribution(theta1).ravel(), prior0, theta1)[:-1]
        g_ratio = float(np.linalg.norm(g_1["resolved"][key] - g_sur)/np.linalg.norm(g_0["resolved"][key]))
        loss_sur = float(ev1["nll"].sum() + ev1["prior"])
        print(f"|full - surrogate grad z| at the surrogate MAP / |full grad z| at z = 0: {g_ratio:.2e} (Newton decrement "
              f"{float(ev1['newton_decrement']):.1e}); full loss {float(loss_1):.6f} vs surrogate {loss_sur:.6f}")

        s, theta_s, ev_s = gp_error_scale(resp, prior0, theta1, verbose=True)
        _save_npz(npath, s=s, theta1=theta1, theta_s=theta_s, g_ratio=g_ratio, loss_full=loss_1,
                  loss_surrogate=loss_sur, chi2_1=ev1["chi2"], n_good=resp.n_good, gamma_e=ev_s["gamma_e"])
    noise = numpy.load(npath)
    print("noise scales of the sum prior (information only):", dict(zip(resp.keys, numpy.round(noise["s"], 4))))
    # the scales every evidence here uses: the previous polar run's stage C (so the DCT, polar-only and sum evidences,
    # share one beta); the sum prior's own if that run is missing
    old = [f for f in sorted(glob.glob(f"{prev}/noise-polar-*.npz"), key=os.path.getmtime)
           if numpy.load(f)["s"].shape == (len(resp.keys),)]
    s_noise = numpy.load(old[-1])["s"] if old else noise["s"]
    print("noise scales used:", dict(zip(resp.keys, numpy.round(s_noise, 4))), f"({old[-1] if old else 'own'})")
    if noise["g_ratio"] > 1e-3:
        print(f"WARNING: the full-model and surrogate gradients differ ({float(noise['g_ratio']):.1e} of the gradient "
              "at z = 0): the surrogate is not the full model")
    if numpy.any(s_noise > 3):
        print("WARNING: s >> 1: the model does not fit those exposures; rescaling the errors hides a model failure")
beta = 1/np.asarray(s_noise)**2
assert len(s_noise) == len(resp.keys)

# %%
# ---------------------------------------------------------------- D: evidence grids
def local(center, names, steps, n=1):
    """Local axes about `center` for `names` (pa as a linear axis within +-90 deg of the centre), with scales, bounds."""
    sc = {k: ("linear" if scales[k] == "periodic" else scales[k]) for k in names}
    b = {k: ((center[k] - periods[k]/2, center[k] + periods[k]/2) if scales[k] == "periodic" else bounds[k]) for k in names}
    return local_axes({k: center[k] for k in names}, steps, sc, n), sc, b


with timed("D: grids"):
    fp_d = fingerprint(resp_fp, beta, level_mu, level_sd, n_iter, grid.digest(), m_max, code_fp)
    poly = {k: start[k] for k in ("sigma_p", "length_r", "length_theta")}
    dct = {k: start[k] for k in ("sigma_d", "length_d")}
    geo = {k: start[k] for k in geometry_names}
    rounds = []
    for rnd in range(1, n_rounds + 1):
        old = (dict(poly), dict(dct), dict(geo))
        # (1) polar shape at the current DCT and geometry (the DCT part fixed: the polar part claims the smooth disc)
        ax, sc, b = local(poly, poly, joint_steps)
        g1 = gp_evidence_grid_nd(resp, ax, make_at(**dct, **geo), beta, outdir=f"{out}/grid",
                                 tag=f"polar{rnd}_" + fingerprint(fp_d, *dct.values(), *geo.values()), abbrev=abbrev,
                                 bounds=b, scales=sc, n_iter=n_iter, max_growth=2, theta0=None)
        poly = gp_best(g1)
        clear_basis_cache()
        # (2) the DCT at the new polar part
        ax, sc, b = local(dct, dct, joint_steps)
        g2 = gp_evidence_grid_nd(resp, ax, make_at(**poly, **geo), beta, outdir=f"{out}/grid",
                                 tag=f"dct{rnd}_" + fingerprint(fp_d, *poly.values(), *geo.values()), abbrev=abbrev,
                                 bounds=b, scales=sc, n_iter=n_iter, max_growth=2)
        dct = gp_best(g2)
        # (3) the geometry at both
        ax, sc, b = local(geo, geo, geo_steps)
        g3 = gp_evidence_grid_nd(resp, ax, make_at(**poly, **dct), beta, outdir=f"{out}/grid",
                                 tag=f"geo{rnd}_" + fingerprint(fp_d, *poly.values(), *dct.values()), abbrev=abbrev,
                                 bounds=b, scales=sc, n_iter=n_iter, max_growth=1)
        geo = gp_best(g3)
        clear_basis_cache()
        rounds.append(dict(polar=poly, dct=dct, geometry=geo))
        print(f"round {rnd}: polar {poly}, DCT {dct}, geometry {geo} (inc {numpy.degrees(numpy.arccos(geo['cos_inc'])):.1f} deg)")
        if same_cell(poly, old[0]) and same_cell(dct, old[1]) and same_cell(geo, old[2]):
            break
    # the joint local grid: the posterior over the searched hyperparameters and the evidence
    searched_d = shape_names + (geometry_names if joint_geometry else ()) + (("dx", "dy") if fit_offset else ())
    fixed = {k: v for k, v in (poly | dct | geo).items() if k not in searched_d}
    center = {k: v for k, v in (poly | dct | geo).items() if k in searched_d}
    grid_sum = joint(center, resp, beta, "joint_" + fp_d, fix=fixed)
    clear_basis_cache()
    # from here (stages E, F) the geometry is held at the joint's best and only the shape axes are searched
    searched = shape_names + (("dx", "dy") if fit_offset else ())
    fixed = {k: v for k, v in (poly | dct | geo | gp_best(grid_sum)).items() if k not in searched}
    post_sum = gp_hyper_posterior(grid_sum)
    best_sum = gp_best(grid_sum)
    print(f"joint best {best_sum} (fixed in D: {sorted(set(shape_names + geometry_names) - set(searched_d))})")
    lo, hi = bounds["length_d"]
    ld = best_sum["length_d"]
    print(f"best length_d {ld:.3g} px: " + ("ON/NEAR the upper bound " + str(hi) if hi/ld < joint_steps["length_d"] else
          "ON/NEAR the lower bound " + str(lo) if ld/lo < joint_steps["length_d"] else f"interior (bounds {lo}-{hi})"))
    tag_dct = f"o{dct_order:g}_t0_" + fingerprint(resp_fp, beta, level_mu, level_sd, n_iter, code_fp)
    grid_dct = gp_evidence_grid(resp, sigmas0, lengths0, dct_order, None, lp, beta, outdir=f"{out}/grid", tag=tag_dct,
                                n_iter=n_iter, bounds=(dct_bounds["sigma"], dct_bounds["length"]))
    grid_dct = grid_dct | dict(names=["sigma", "length"], axes=dict(sigma=grid_dct["sigmas"], length=grid_dct["lengths"]))
    post_dct = gp_hyper_posterior(grid_dct, box=(sigmas0, lengths0), bounds=dct_bounds)
    clear_basis_cache()
    hd0 = gp_best(grid_dct)
    print(f"DCT-only best {hd0}; length on the bound: {dct_bounds['length'][1]/hd0['length'] < 2.}")
    # the polar-only run's evidence, same response and noise scales (its saved summary), if present
    polar_only = None
    try:
        f = numpy.load(f"{prev}/grid-polar.npz")
        polar_only = dict(log_evidence_bounds=float(f["log_evidence_bounds"]), edge_weight=float(f["edge_weight"]),
                          max_logZ=float(numpy.max(f["logZ"])))
    except Exception as e:
        print(f"polar-only evidence not read ({type(e).__name__}: {e}); no sum - polar comparison")
    for name, g, post in (("sum", grid_sum, post_sum), (f"DCT order {dct_order:g}", grid_dct, post_dct)):
        best = gp_best(g)
        _save_npz(f"{out}/grid-{name.split()[0].lower()}.npz", logZ=g["logZ"], thetas=g["thetas"],
                  converged=g["converged"], weights=post["weights"],
                  **{f"axis_{k}": v for k, v in g["axes"].items()}, **{f"q_{k}": v for k, v in post["quantiles"].items()},
                  log_evidence=post["log_evidence"], log_evidence_bounds=post["log_evidence_bounds"],
                  edge_weight=post["edge_weight"])
        print(f"[{name}] log evidence (log-uniform on the bounds) {post['log_evidence_bounds']:.2f}, max log Z "
              f"{g['logZ'].max():.2f}; best {best}; 16/50/84% "
              + "; ".join(f"{k} {numpy.round(q, 3)}" for k, q in post["quantiles"].items())
              + f"; edge weight {post['edge_weight']:.3f}; unconverged {int((~g['converged'].astype(bool)).sum())}")
    valid = max(post_sum["edge_weight"], post_dct["edge_weight"]) < 0.1
    print(("" if valid else "NOT VALID (posterior weight on a grid's edge): ")
          + f"sum - DCT log evidence: {post_sum['log_evidence_bounds'] - post_dct['log_evidence_bounds']:.1f} nats; max log Z "
          f"difference {grid_sum['logZ'].max() - grid_dct['logZ'].max():.1f} (the sum prior's geometry is "
          f"{'marginalised' if joint_geometry else 'conditional on the round search (profile): compare max log Z too'})")
    if polar_only is not None:
        print(f"sum - polar-only log evidence: {post_sum['log_evidence_bounds'] - polar_only['log_evidence_bounds']:.1f} nats; "
              f"max log Z difference {grid_sum['logZ'].max() - polar_only['max_logZ']:.1f}")
    # sanity (nesting): the sum prior with sigma_p at its floor is the DCT prior plus 441 near-dead latents: its log Z
    # at the DCT-only best should match the DCT grid's best to within a few tenths of a nat (more: the grids' MAPs differ)
    hd = gp_best(grid_dct)
    pr_n = sum_at(**(poly | dict(sigma_p=bounds["sigma_p"][0], sigma_d=hd["sigma"], length_d=hd["length"]) | geo))
    th_n, _ = gp_map(resp, pr_n, None, beta, n_iter)
    print(f"nesting check: log Z(sum, sigma_p at floor, DCT best) {float(gp_log_evidence(resp, pr_n, th_n, beta)['logZ']):.2f}"
          f" vs DCT grid max {grid_dct['logZ'].max():.2f}")

# evidence surfaces for each pair of the joint grid's axes (maximum over the others); the star marks the best converged point
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


names = grid_sum["names"]
pairs = list(itertools.combinations(range(len(names)), 2))
ncol = 4
fig, axs = plt.subplots((len(pairs) + ncol - 1)//ncol, ncol, figsize=(10*ncol, 9*((len(pairs) + ncol - 1)//ncol)), squeeze=False)
axs = axs.ravel()
for ax, (a, b) in zip(axs, pairs):
    other = tuple(i for i in range(len(names)) if i not in (a, b))
    img = grid_sum["logZ"].max(axis=other) - grid_sum["logZ"].max()
    evidence_panel(ax, img, grid_sum["axes"][names[a]], grid_sum["axes"][names[b]], names[a], names[b],
                   (grid_sum["best"][a], grid_sum["best"][b]), "joint grid, max over the others",
                   unconverged=(~grid_sum["converged"].astype(bool)).all(axis=other))
for ax in axs[len(pairs):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{out}/evidence-sum.png")
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


hyper_b, theta_b = at_best(grid_sum)
hyper_b = fixed | hyper_b     # the searched hyperparameters plus the fixed ones (geometry)
current = dict(params=ref_params, resp=resp, resp_fp=resp_fp, beta=beta, s=numpy.asarray(s_noise), grid=grid_sum,
               theta=theta_b, stage="D")
outer_converged = False
for it in range(1, n_outer + 1):
    prior_b = sum_at(**hyper_b)
    h = fingerprint(it, *hyper_b.values(), current["s"], current["resp_fp"], current["theta"], fo_epochs, fo_lr_opd, fo_lr_disc, optx_max_steps, lp,
                    repr(sorted(precond_kwargs.items())), repr(sorted(things_gp)), grid.digest(), m_max, code_fp)
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
            x_new, losses, optx_info = refit(opt_params, model_b, exps_b, things_gp)
            fig = plt.figure(figsize=(10, 10))
            plt.plot(losses[:-1])
            plt.plot([fo_epochs], losses[-1:], "o", label="after L-BFGS")   # the end of the one optimisation call
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
        tag = f"outer{it}_sum_" + fingerprint(resp_new_fp, beta_new, lp, n_iter, *hyper_b.values(), grid.digest(), m_max,
                                               code_fp)
        grid_new = joint({k: v for k, v in hyper_b.items() if k in searched}, resp_new, beta_new, tag, theta0=theta_new,
                         fix={k: v for k, v in hyper_b.items() if k not in searched})
        hyper_new, theta_new_b = at_best(grid_new)
        hyper_new = {k: v for k, v in hyper_b.items() if k not in searched} | hyper_new
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
    prior_f = sum_at(**hyper_b)
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
        prior_c = sum_at(**(hyper_b | {k: grid_f["axes"][k][i] for k, i in zip(grid_f["names"], cell)}))
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
    comp = to_host(sum_components(prior_f, theta_f, lap))
    diag_p = polar_orders_diagnostic(prior_f, theta_f, lap) if m_max is not None else None
    sub, th_sub, lap_sub = dct_view(prior_f, theta_f, lap)
    diag_d = gp_spectral_diagnostic(sub, th_sub, lap_sub)

    numpy.save(f"{out}/gp-disc-log10.npy", log_map)
    numpy.save(f"{out}/gp-disc-log10-std.npy", log_std)
    numpy.save(f"{out}/gp-mixture-mean.npy", mix_mean)
    numpy.save(f"{out}/gp-mixture-std.npy", mix_std)
    numpy.savez(f"{out}/components.npz", **comp)
    numpy.savez(f"{out}/dct-diagnostic.npz", **diag_d)
    if diag_p is not None:
        numpy.savez(f"{out}/polar-diagnostic.npz", **diag_p)
    z_f, l_f = prior_f.params(theta_f)
    p_final = dict(current["params"]) | {"resolved": {key: z_f}, "resolved_level": {key: l_f}}
    save_npy(f"{out}/gp-params.npy", to_host(p_final))

    # discs (sqrt stretch): L1/TSV reference, polar MAP, posterior-mixture mean, uncertainty, SNR
    from matplotlib.colors import PowerNorm
    pitch = 0.0432*2
    ext = numpy.array([-1, 1, -1, 1])*resolved_wid*pitch/2
    sq = lambda im: dict(norm=PowerNorm(0.5, 0, numpy.percentile(im, 99.5)))   # each panel its own stretch
    panels = [(numpy.asarray(d_ref), "L1/TSV reference", sq(numpy.asarray(d_ref))),
              (10**log_map, "polar + DCT GP MAP", sq(10**log_map)),
              (mix_mean, "posterior-mixture mean", sq(10**log_map)),   # the MAP's stretch: robust to outliers
              (comp["f_p"], "polar component (log10, MAP)", dict(cmap="RdBu_r")),
              (comp["f_d"], "DCT component (log10, MAP)", dict(cmap="RdBu_r")),
              (comp["corr"], "posterior corr(polar, DCT) per pixel", dict(cmap="RdBu_r", vmin=-1, vmax=1)),
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
    if diag_p is not None:
        axs[0].plot(diag_p["m"], diag_p["power_m"], "o", label="<z^2>")
        axs[0].plot(diag_p["m"], diag_p["var_m"], "s", label="posterior variance")
        axs[0].axhline(1, color="k", ls="--")
        axs[0].set(xlabel="azimuthal order m", title="Polar latents per order (prior: 1)")
        axs[0].legend()
    axs[1].errorbar(diag_d["freq"], diag_d["power"], diag_d["power_err"], fmt="o")
    axs[1].axhline(1, color="k", ls="--")
    axs[1].set(xlabel="cycles / px", ylabel="<z_k^2> (prior: 1)", title="DCT component's spectral power", xscale="log")
    axs[2].plot(diag_d["freq"], diag_d["var"], "o")
    axs[2].set(xlabel="cycles / px", ylabel="posterior variance (prior: 1)", title="DCT data constraint", xscale="log")
    fig.tight_layout()
    fig.savefig(f"{out}/diagnostics.png")
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
        prior="polar+DCT",
        # the geometry's posterior (final grid): inc from cos_inc's quantiles (reversed), pa modulo 180 in the disc
        # grid's frame; the coarse (stage C) and refined (D1) searches' bests; the fraction of disc pixels beyond r_max
        geometry=dict(inc_deg=float(numpy.degrees(numpy.arccos(hyper_b["cos_inc"]))), pa_deg=hyper_b["pa"] % 180.,
                      inc_q=(list(map(float, numpy.degrees(numpy.arccos(numpy.clip(
                          post_f["quantiles"]["cos_inc"][::-1], -1., 1.))))) if "cos_inc" in post_f["quantiles"] else None),
                      pa_q=(list(map(float, pa_interval(post_f["quantiles"]["pa"]))) if "pa" in post_f["quantiles"] else None),
                      search_rounds=rounds, frame="disc grid (gp.PolarGrid.coordinates)",
                      fraction_beyond_r_max=float(grid.fraction_outside(prior_f.geometry))),
        polar_grid=dict(n_r=n_r, n_theta=n_theta, nu_r=nu_r, nu_theta=nu_theta, spacing_px=grid.spacing,
                        r_max_px=grid.r_max, cover_inc=cover_inc, m_max=m_max, n_latent=n_latent),
        components=dict(rms_polar=float(comp["rms_p"]), rms_dct=float(comp["rms_d"]), var_share_polar=float(comp["var_share"]),
                        mean_corr=float(numpy.mean(comp["corr"])), min_corr=float(numpy.min(comp["corr"]))),
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
        stage_d=dict(quantiles={k: list(map(float, q)) for k, q in post_sum["quantiles"].items()},
                     edge_weight=post_sum["edge_weight"], log_evidence_bounds=post_sum["log_evidence_bounds"],
                     rounds=rounds),
        # at the reference optics (stage D), each under a log-uniform hyperprior on its fixed bounds, with the same
        # noise scales; the Laplace approximation's bias (gp.py) is not corrected
        sum_minus_dct_log_evidence=post_sum["log_evidence_bounds"] - post_dct["log_evidence_bounds"],
        sum_minus_polar_log_evidence=(None if polar_only is None else
                                      post_sum["log_evidence_bounds"] - polar_only["log_evidence_bounds"]),
        polar_only=polar_only, joint_geometry=joint_geometry,
        sum_vs_polar_basis=("sum: marginal over geometry (joint 7-axis grid) vs the polar run's marginal" if joint_geometry else
                            "sum: CONDITIONAL on the searched geometry (profile, optimistic) vs the polar run's marginal"),
        evidence_edge_weights=dict(sum=post_sum["edge_weight"], dct=post_dct["edge_weight"]),
        dct_baseline=dict(order=dct_order, log_evidence_bounds=post_dct["log_evidence_bounds"],
                          best=gp_best(grid_dct)),
        verify_max_abs=float(v["abs_diff"].max()), verify_max_rel_change=float(v["rel_change"].max()),
        gradient_ratio=float(noise["g_ratio"]),
    )
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

