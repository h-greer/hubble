# %%
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax
import numpy
import os
import time

jax.config.update("jax_enable_x64", True)

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
from spectra import *
import precond
import jax.random as jr
import interpax as ipx
from interpax._fd_derivs import approx_df

out = "gmaur-wpca-trace"
os.makedirs(out, exist_ok=True)

# Profiling copy of gmaur-wpca. Each stage first runs a traced probe of n_trace epochs from the stage's starting point
# (trace in PROFILE_DIR/stage1, /stage2), then the full stage untraced from the same point, so the fit is unchanged.
# PROFILE_DIR is set by gmaur-wpca-trace.sh on /aphid/scratch-3month, as is JAX_COMPILATION_CACHE_DIR (read by JAX from
# the environment), so the full stage reuses the probe's compiled programs. Image rendering and plots are not traced.
# Every compilation is logged with its duration
prof_dir = os.environ.get("PROFILE_DIR", f"{out}/profile")
os.makedirs(prof_dir, exist_ok=True)
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)   # cache every compile, incl. the ~1000 small eager ones
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

n_stage1, n_stage2 = 1000, 500
n_trace = 5

prof_opts = jax.profiler.ProfileOptions()
prof_opts.host_tracer_level = 2     # XLA compile/execute events on the host
prof_opts.python_tracer_level = 0   # off: the Python tracer buffered ~GBs in host RAM and the 20-epoch probe segfaulted


def traced_probe(name, optimise, opt_params, model, optimisers, ones=()):
    """The preconditioner and n_trace epochs of `optimise` under the profiler, timed; the epochs are discarded and C
    is returned for the full stage (same starting point, so the same C)"""
    jax.profiler.start_trace(f"{prof_dir}/{name}", profiler_options=prof_opts)
    t0 = time.perf_counter()
    with jax.profiler.TraceAnnotation(name):
        C = precond_C(opt_params, model, exposures_single, ones=ones)
        jax.block_until_ready(C)
        print(f"{name} preconditioner: {time.perf_counter() - t0:.1f} s (incl. compilation)")
        _, hist = optimise(opt_params, model, exposures_single, optimisers, n_trace, use_c=C)
        jax.block_until_ready(hist[-1])
    jax.profiler.stop_trace()
    print(f"{name} probe: {time.perf_counter() - t0:.1f} s for {n_trace} epochs (traced, incl. compilation)")
    return C


# ----------------------------------------------------------------------------------------------------------------------
# Script-local copies of the detector and preconditioner with two speed-ups (the core modules are unchanged):
#
# 1. FastResample: detectors.Resample with the same cubic2 (not-a-knot) interpolation, but the spline derivatives
#    fx, fy, fxy come from a dense derivative matrix D (D @ f = df/dx on the pixel grid; linear in f, so D is the
#    cubic2 derivative of the identity), built once, instead of lineax tridiagonal solves (56k tiny cuSPARSE launches,
#    ~20% of GPU time in the profile). Identical to round-off (1e-16 on the image, 4e-15 on its gradient at 272 px).
#
# 2. _gn_core_lin / precond_C: precond._gn_core with one jax.linearize per exposure, reused for the Hutchinson vjps
#    (linear_transpose) and the exact jvps, so the forward model is traced once per exposure rather than once for the
#    vjp and again per exact group (~20-40% less tracing and compilation on a tiny config; identical to 1e-15).
#    precond_C replaces precond_cache.get_precond(..., per_block=False): one call on all params, no disk cache, and C = 1
#    for the params in `ones`; it is passed to the optimisers as use_c.

def cubic2_derivative_matrix(n):
    c = dlu.pixel_coords(n, 2)
    return np.asarray(approx_df(c[1][:, 0], np.eye(n), "cubic2", 0))


class FastResample(Resample):
    D: np.ndarray

    def __init__(self, D, rotation=0.0, anisotropy=1.00, order=3):
        super().__init__(rotation, anisotropy, order)
        self.D = D

    def apply(self, PSF):
        coords = dlu.pixel_coords(PSF.data.shape[0], 2)
        point = lambda p: self.transform(p[:, None, None])[:, 0, 0]
        jac = jax.vmap(jax.jacfwd(point))(coords.reshape(2, -1).T)
        det = np.abs(np.linalg.det(jac)).reshape(coords.shape[1:])
        f = PSF.data
        fx = self.D @ f
        fy = f @ self.D.T
        fxy = fx @ self.D.T
        xs, ys = coords
        xp, yp = self.transform(coords).reshape(2, -1)
        out = ipx.interp2d(yp, xp, ys[:, 0], xs[0], f, method="cubic2", extrap=0.0, fx=fx, fy=fy, fxy=fxy)
        return PSF.set("data", det*out.reshape(coords[0].shape))


class FastNICMOSDetector(dl.LayeredDetector):
    """detectors.NICMOSDetector with FastResample; the same layer names and order"""
    def __init__(self, oversample, wid, pad=0):
        layers = [
            ("resample", FastResample(cubic2_derivative_matrix((wid + 2*pad)*oversample), anisotropy=1.)),
            ("downsample", dl.layers.Downsample(oversample)),
            ("bias", dl.layers.AddConstant(value=0.0)),
        ]
        if pad:
            layers.append(("crop", dl.layers.Resize(wid)))
        super().__init__(layers)


@eqx.filter_jit
def _gn_core_lin(params, exposures, model, key, exact_groups, quad_idx, n_probes, probe_batch, chunk):
    """precond._gn_core with the forward model linearised once per exposure"""
    params = precond._f64(params)
    x, unravel = precond.ravel_pytree(params)
    fs = precond._weighted_images(x, unravel, exposures, model)
    gn = np.zeros_like(x)
    lins = [jax.linearize(f, x) for f in fs]   # (out, f_lin) per exposure

    if n_probes > 0:  # Hutchinson: GN_k = E_z[(J^T (z sqrt(w)))_k^2], z Rademacher over pixels
        for ei, (out, f_lin) in enumerate(lins):
            vjp = jax.linear_transpose(f_lin, x)

            def probe(k, vjp=vjp, shape=out.shape):
                return vjp(jr.rademacher(k, shape, dtype=x.dtype))[0] ** 2

            keys = jr.split(jr.fold_in(key, ei), n_probes)
            gn = gn + jax.lax.map(probe, keys, batch_size=min(int(probe_batch), n_probes)).mean(0)

    for users, idx in exact_groups:  # exact GN_kk: one jvp per k, only through the exposures that read it
        gn_k = lambda v, i, users=users: sum(np.sum(lins[e][1](v) ** 2) for e in users)
        gn = gn.at[idx].set(precond._map_tangents(gn_k, x, idx, chunk))

    if quad_idx.shape[0] > 0:  # non-quadratic in the error model: |H_kk| instead
        h = precond._hessian_entries(x, unravel, exposures, model, quad_idx, chunk)
        gn = gn.at[quad_idx].set(np.abs(h))

    reg, _ = precond.ravel_pytree(precond._regulariser_curvature(params, exposures))
    return precond._finish(gn + reg, unravel)


def precond_C(params, model, exposures, ones=(), n_probes=32, probe_batch=4, chunk=4, exact_max_size=32,
              exact_blocks=("cold_mask_opd",), key=jr.key(0), damping=1e-3):
    """Diagonal preconditioner C (structure of params) from _gn_core_lin, with C = 1 for the params in `ones`"""
    sub = {k: v for k, v in params.items() if k not in ones}
    model = ModelParams(params).inject(model)      # the current values, as precond_cache.get_diag does
    exact, quad = precond._indices(sub, exposures, tuple(exact_blocks), int(exact_max_size))
    diag = _gn_core_lin(sub, exposures, model, key, exact, quad, int(n_probes), int(probe_batch), int(chunk))
    C = dict(precond.precond_from_diag(diag, damping))
    C.update({k: jax.tree.map(np.ones_like, params[k]) for k in ones if k in params})
    return {k: C[k] for k in params}

# %%
wf_wid = 512
wid = 64
oversample = 4
pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

filt = "F110W"
nwavels = 200
npoly = 8

# Wavefront PCA basis of the calibrator-joint fits (notebooks/calibrator-pca.ipynb): the mean high-order wavefront plus
# n_wpca unit-norm modes (in nm per unit coefficient), as Fourier coefficients on the fits' grid, with the primary_low span removed. The high-order
# wavefront is restricted to this basis (primary_opd stays zero and is not fitted). The basis was fitted in F160W on a
# 48 x 48 grid (k_max = 24), so in F110W it reaches only ~1.5" (0.9 um) to ~1.8" (1.1 um), short of the crop edges (~2.6")
wpca_file = "../data/wavefront_pca.npz"
wpca = np.load(wpca_file)
n_wpca = wpca["modes"].shape[0]
n_modes = int(round(wpca["mean"].size**0.5))
n_zernikes = 50

resolved_wid = 56   # 56 x 0.0864" = 4.84", the 64 px (4.84") crop
regulariser = np.array([0.02, 0.4])

# The calibrator-joint model: Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter,
# Hermite interpolation between 20 knots across the 200-wavelength spectrum, no pupil amplitude map; plus the wavefront PCA.
# The disc (one wavelength, the filter centre, propagated directly) is an InterpolatedResolvedSource: exact PSFs within
# 0.6" and 36 anchor PSFs beyond (rings 0.6-2.2", 6 angles), 184 renders instead of 56^2 = 3136
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20,
                                  wavefront_pca=wpca_file, pad=pad)
detector = FastNICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_{filt}_{nwavels}.npy")[:, :npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'


class WavefrontPCAResolvedFit(PointResolvedFit):
    """Star plus resolved disc, with calibrator-joint's per-exposure keys (cold mask shift, tilt and OPD, detector
    defocus); the wavefront PCA coefficients are per visit, like primary_opd in calibrator-joint. One stellar spectrum
    for all exposures (global), as is the disc (keyed by target)."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                                "primary_wpca": "visit", "cold_mask_opd": "exposure",
                                                "defocus3": "exposure", "spectrum": "global"}


# GM Aur F110W, 48 s, programme 10852 (DD2): two rolls 58 deg apart, two exposures each. The disc is shared by both
# rolls, so roll diversity separates it from the pupil-fixed speckles, alongside the wavefront PCA basis
#   n9p511 2006-09-17, ORIENTAT  43.6 deg
#   n9p510 2006-11-22, ORIENTAT -14.4 deg
# (The 1998 F110W visit n4q813 (DD1, 640 s, -145.5 deg) is not used.)
exposures_single = [
    exposure_from_file(ddir + f'{root}_m_clc_calf.fits',
                       WavefrontPCAResolvedFit(spectrum_basis, filt, wid=resolved_wid, regulariser=regulariser,
                                               resolved_source="interp"),
                       crop=wid, extra_bad=None, hole_margin=0.1)
    for root in [
        "n9p511wzq", 
        "n9p511x0q", 
        "n9p510kxq", 
        "n9p510kyq"
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

p0 = params.params
t0 = time.perf_counter()
star = images(p0 | {"resolved": jax.tree.map(lambda r: np.full_like(r, -30.), p0["resolved"])})   # disc off
disc = images(p0 | {"spectrum": jax.tree.map(np.zeros_like, p0["spectrum"])})                      # star off (eps flux)
jax.block_until_ready((star, disc))
print(f"initial render: {time.perf_counter() - t0:.1f} s")

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
C1 = traced_probe("stage1", optimise_new, opt_params, model_single, things_start)

t0 = time.perf_counter()
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, n_stage1, use_c=C1)
jax.block_until_ready(params_history[-1])
print(f"stage 1: {time.perf_counter() - t0:.1f} s for {n_stage1} epochs")

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

C2 = traced_probe("stage2", optimise_new_resolved, opt_params, model_single, things, ones=("resolved",))   # C = 1 for the disc, as optimise_new_resolved
jax.profiler.save_device_memory_profile(f"{prof_dir}/memory.prof")   # device memory after the stage-2 probe
print("traces saved to", prof_dir)

t0 = time.perf_counter()
losses, params_history = optimise_new_resolved(opt_params, model_single, exposures_single, things, n_stage2, use_c=C2)
jax.block_until_ready(params_history[-1])
print(f"stage 2: {time.perf_counter() - t0:.1f} s for {n_stage2} epochs")

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
np.save(f"{out}/gmaur.npy", disc)

for name, im in (("gmaur", 10**disc), ("gmaur-log", disc)):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.colorbar()
    plt.savefig(f"{out}/{name}.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])


