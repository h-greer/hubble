# %%
# Occulter-glint fit of ONE calibrator: PSF-5-K0 (V* LQ Hya, F160W), its two 224 s exposures at one roll angle.
# Warm-started from this star's calibrator-joint fit (job 192), then the glint parameters are released stage by stage
# (see README.md). Each stage is saved; a resubmit resumes after the last saved stage.
# FIT_DRY_RUN=1: tiny sizes / epochs for a local smoke test, outputs under $FIT_DRY_RUN_OUT (default: the system temp
# dir, never next to this script)
import sys
sys.path.insert(0, '..')

# %%
import jax.numpy as np
import jax
import numpy as onp
import os
import csv
import json
import tempfile

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache (directory from JAX_COMPILATION_CACHE_DIR, set in the .sh); cache every compile
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

import equinox as eqx
import optax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["image.origin"] = 'lower'

from detectors import *
from apertures import *
from models import *
from fitting import *
import fitting
from spectra import *
from occulters import glint_log_prior, edge_profile, logamp_guard, LAMBDA_F_REF, EDGE_M0
from batching import make_batch

# %%
TARGET = "PSF-5-K0"
ROOTNAMES = ["n8zu85ndq", "n8zu85neq"]      # both 224 s, ORIENTAT 56.84 (the same roll)
FILTER = "F160W"

# "climb" (Fourier edge + rim) or "image" (the measured hole outline + rim): switching is this one line
OCCULTER = "climb"
EDGE = "soft_binarise"     # the edge model the warm start was fitted with
EDGE_ORDER = 5             # climb: free edge orders m = 3..5
RIM_K, RIM_H = 3, 5        # rim radial scales (0.25, 0.5, 1 lambda F#) x azimuthal harmonics (1 + 2 (m_rim - 1))

# Glint prior (occulters.glint_log_prior): sigmas on the resulting rim OPD (nm) and log-amplitude, edge coefficients.
# SIGMA_AMP is used in every stage, but it only acts from s4 (rim_amp is zero and frozen before that)
SIGMA_OPD, SIGMA_AMP, SIGMA_EDGE, SIGMA_POS_AMP = 30., 0.01, 0.01, 3e-3
RIM_OPD_MAX_STEP = 5.      # s3-s4: per-step clip of rim_opd updates (nm). A rate limit only, NOT a bound on the total
                           # (600 steps x 5 nm); how far the linear chromatic mode is off is recorded in summary.json

# Warm start: the calibrator-joint L-BFGS result; intermediate-params.npy only fills the 5 entries it alone holds
# (cold_mask_shear, primary_shear, outer_radius, primary_outer, primary_pad)
WARM = ["calibrator-joint/192-PSF-5-K0/params_amp_optx.npy", "calibrator-joint/192-PSF-5-K0/intermediate-params.npy"]

DRY_RUN = os.environ.get("FIT_DRY_RUN") == "1"
STOP_AFTER = os.environ.get("FIT_STOP_AFTER")       # stage name: exit after it (dry run: one stage per process)
INJECT_NAN = os.environ.get("FIT_INJECT_NAN")       # test: stage name whose start gets fnumber = NaN

if DRY_RUN:
    wf_wid, wid, oversample, pad = 32, 16, 2, 2
    nwavels, npoly, n_modes, n_zernikes, n_nodes = 3, 2, 4, 4, 2
    n_probes, epochs, polish_steps = 1, [3, 3, 3, 3, 3], 8
    out = os.path.join(os.environ.get("FIT_DRY_RUN_OUT", os.path.join(tempfile.gettempdir(), "fit_psf5k0_dryrun")), OCCULTER)
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir", os.path.join(os.path.dirname(out), "jax_cache"))
else:
    # as calibrator-joint.py
    wf_wid, wid, oversample, pad = 512, 64, 4, 2
    nwavels, npoly, n_modes, n_zernikes, n_nodes = 200, 8, 48, 50, 20
    n_probes, epochs, polish_steps = 32, [500, 2000, 400, 300, 300], 512
    out = f"psf5k0/{OCCULTER}"
os.makedirs(out, exist_ok=True)

# %%
vects = onp.load("../data/iterative_spectrum_basis_F160W_200.npy")[:, :npoly]
vects = vects[onp.linspace(0, 199, nwavels).astype(int)]      # all 200 in production
spectrum_basis = np.asarray(vects/onp.sqrt(onp.mean(vects**2, axis=0)))

# The two exposures (preferred copies in data/calibrators.csv); fail at startup if either is unusable
rows = {r["rootname"]: r for r in csv.DictReader(open("../data/calibrators.csv"))
        if r["rootname"] in ROOTNAMES and r["preferred_copy"] == "True"}
paths = []
for root in ROOTNAMES:
    r = rows[root]
    p = "../" + r["file_path"]
    assert r["target"] == TARGET and r["filter"] == FILTER and os.path.exists(p), f"{root}: {r['target']} {r['filter']} {p}"
    assert fits.getheader(p, 2)["NAXIS"] > 0, f"{root}: empty error extension"
    paths.append(p)


# As calibrator-joint.py. Every occulter_* parameter is shared by both exposures
class PerExposureColdMaskFit(SinglePointFit):
    PARAM_KEYS = SinglePointFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                              "primary_opd": "exposure", "cold_mask_opd": "exposure",
                                              "defocus1": "exposure", "defocus2": "exposure", "defocus3": "exposure"}


exposures = [exposure_from_file(p, PerExposureColdMaskFit(spectrum_basis, FILTER), crop=wid, extra_bad=None, hole_margin=0.1)
             for p in paths]
print(TARGET, [(e.key, float(e.orient), e.exptime) for e in exposures])

m_rim = (RIM_H - 1)//2 + 1
n_edge = EDGE_ORDER - EDGE_M0 + 1
rim_scales = onp.array([0.25, 0.5, 1.])*LAMBDA_F_REF
assert len(rim_scales) == RIM_K


def make_model(params, chromatic):
    kw = dict(edge=EDGE, rim_scales=rim_scales, rim_amp=onp.zeros((RIM_K, RIM_H)), rim_opd=onp.zeros((RIM_K, RIM_H)))
    if OCCULTER == "climb":
        kw |= dict(m_edge=EDGE_ORDER, m_rim=m_rim, edge_coeffs=onp.zeros((n_edge, 2)))
    else:
        kw |= dict(m_max=m_rim, mirror=False, rotation=0.)
    optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=n_nodes, pad=pad,
                                      occulter=OCCULTER, occulter_kwargs=kw, chromatic=chromatic)
    return set_array(NICMOSModel(exposures, dict(params), optics, NICMOSDetector(oversample, wid, pad=pad)))


# %%
# calibrator-joint's initial parameters plus the glint leaves (all zero: the elliptical hole)
params = {
    "spectrum": {}, "primary_opd": {}, "primary_low": {}, "primary_tilt": {},
    "cold_mask_opd": {}, "cold_mask_tilt": {}, "cold_mask_shift": {},
    "cold_mask_rot": {}, "cold_mask_shear": {}, "cold_mask_scale": {}, "cold_mask_distortion": {},
    "defocus1": {}, "defocus2": {}, "defocus3": {},
    "primary_rot": {}, "primary_shear": {}, "bias": {},
    "occulter_radius": 0.8, "occulter_ellipticity": np.zeros(2),
    "occulter_rim_amp": np.zeros((RIM_K, RIM_H)), "occulter_rim_opd": np.zeros((RIM_K, RIM_H)),   # rim_opd in nm
    "fnumber": 45.7, "anisotropy": 1. + 1e-5,
    "outer_radius": 1.2*0.9768, "secondary_radius": 0.365*1.2, "spider_width": 0.072*1.2, "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2, "primary_outer": 1.2, "primary_pad": 0.065*1.2,
    "secondary_shift": np.zeros(2), "primary_distortion": np.zeros((2, 7)), "distortion": np.zeros((2, 7)),
}
if OCCULTER == "climb":
    params["occulter_edge"] = np.zeros((n_edge, 2))
else:
    params["occulter_rotation"] = np.asarray(0.)

for exp in exposures:
    key = lambda k: exp.fit.get_key(exp, k)
    params["spectrum"][key("spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*100)
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

# Warm start by exact key (primary_opd: calibrator-joint stored one wavefront per visit, key = rootname[:6]); a later file
# only fills entries the earlier ones lack. defocus3 (not fitted there) and the glint leaves keep their initial values
GLINT_KEYS = ("occulter_rim_amp", "occulter_rim_opd", "occulter_edge", "occulter_rotation")
seeded = {}
for f in WARM:
    src = onp.load(f, allow_pickle=True).item()
    for k, v in params.items():
        if k in GLINT_KEYS or k == "defocus3" or k not in src:
            continue
        for kk in (v if isinstance(v, dict) else [None]):
            tag = k if kk is None else f"{k}[{kk}]"
            s = src[k] if kk is None else src[k].get(kk, src[k].get(kk[:6]) if k == "primary_opd" else None)
            if tag not in seeded and s is not None and onp.shape(s) == onp.shape(v if kk is None else v[kk]):
                if kk is None:
                    params[k] = np.asarray(s, float)
                else:
                    v[kk] = np.asarray(s, float)
                seeded[tag] = f
n_tags = sum(len(v) if isinstance(v, dict) else 1 for k, v in params.items() if k not in GLINT_KEYS and k != "defocus3")
print(f"warm start: seeded {len(seeded)}/{n_tags};", "from", WARM[1], sorted(t for t, f in seeded.items() if f == WARM[1]))
assert DRY_RUN or len(seeded) == n_tags == 39, sorted(seeded)      # dry run: sizes differ, some shapes do not match

# %%
g = 5e-2


def sgd_clip(lr, delay, step):
    """sgd with an elementwise clip of each update (parameter units)"""
    return optax.chain(sgd(lr, delay), optax.clip(step))


# calibrator-joint's things_start and things, with the occulter frozen at the warm start (released in s2)
things_start = {
    "spectrum": sgd(g*3, 30.), "primary_tilt": sgd(g*5., 15.), "cold_mask_tilt": sgd(g*1., 0), "cold_mask_opd": sgd(g*0.1, 40),
    "defocus1": sgd(g*1, 60), "defocus2": sgd(g*10, 60), "bias": sgd(g*3, 50), "cold_mask_shift": sgd(g*0.3, 70),
    "cold_mask_rot": sgd(g*0.1, 85), "primary_rot": sgd(g*0.03, 85), "primary_low": sgd(g*2., 150),
    "cold_mask_shear": sgd(g*0.1, 100), "cold_mask_distortion": sgd(g*0.1, 150), "primary_shear": sgd(g*1, 100),
    "outer_radius": sgd(g*0.1, 120), "secondary_radius": sgd(g*0.1, 120), "spider_width": sgd(g*0.1, 120),
    "primary_spider": sgd(g*0.1, 120), "primary_secondary": sgd(g*0.1, 120), "primary_pad": sgd(g*0.1, 120),
    "primary_outer": sgd(g*0.1, 120), "secondary_shift": sgd(g*0.3, 120), "primary_distortion": sgd(g*0.1, 150),
    "fnumber": sgd(g*1., 200), "anisotropy": sgd(g*2., 200),
}
things = {
    "primary_opd": adam(2e-2, 50),
    "spectrum": sgd(g*1, 0), "primary_tilt": sgd(g*1., 0), "cold_mask_tilt": sgd(g*1, 0), "cold_mask_opd": sgd(g*1., 0),
    "defocus1": sgd(g*1, 0), "defocus2": sgd(g*1, 0),
    "bias": sgd(g*3, 0), "cold_mask_shift": sgd(g*1, 0), "cold_mask_rot": sgd(g*1., 0.), "primary_rot": sgd(g*3., 0),
    "primary_low": sgd(g*1, 0), "cold_mask_scale": sgd(g*1, 0), "cold_mask_distortion": sgd(g*3, 0),
    "secondary_radius": sgd(g*1., 0), "spider_width": sgd(g*1., 0), "primary_spider": sgd(g*1., 0),
    "primary_secondary": sgd(g*1., 0), "secondary_shift": sgd(g*1, 0), "primary_distortion": sgd(g*3, 0),
    "fnumber": sgd(g*1., 0), "anisotropy": sgd(g*1., 0), "distortion": sgd(g*1, 0),
}
late = things | {"primary_opd": adam(2e-2, 0)}        # after s1b the wavefront is converged: no delay
# Occulter geometry: the edge (climb) / rotation (image) starts after a third of s2
geom = {"occulter_radius": sgd(g*1., 0), "occulter_ellipticity": sgd(g*3, 0),
        ("occulter_edge" if OCCULTER == "climb" else "occulter_rotation"): sgd(g*0.3, epochs[2]//3)}
rim_opd = {"occulter_rim_opd": sgd_clip(g*1., 0, RIM_OPD_MAX_STEP)}
rim_amp = {"occulter_rim_amp": sgd(g*0.3, 0)}

# (name, optimisers, epochs (s5: L-BFGS max steps), chromatic mode). r, e and the edge are
# frozen in s3 and s4 (they trade against the rim leaves) and released again in the s5 polish
stages = [
    ("s1a", things_start, epochs[0], None),
    ("s1b", things, epochs[1], None),
    ("s2", late | geom, epochs[2], None),
    ("s3", late | rim_opd, epochs[3], "linear"),
    ("s4", late | rim_opd | rim_amp, epochs[4], "linear"),
    ("s5", late | geom | rim_opd | rim_amp, polish_steps, "exact"),
]
precond_kwargs = dict(n_probes=n_probes, chunk=4, per_block=False, exact_max_size=32,
                      cache_dir=os.path.abspath(f"{out}/precond_cache"))

# %%
# The glint prior: fitting has no prior hook, so fitting.loss_fn (used by optimise_new and optimise_optimistix) is
# replaced by loss - log prior; clear_caches so no jitted loss traced before keeps the old one
_loss_fn = fitting.loss_fn


def install_prior(model):
    scales = model.optics.get("occulter.layers.occulter.rim_scales")
    occ0 = model.optics.get("occulter.layers.occulter")

    def loss_with_prior(params, exposures, model):
        merged = {**model.params, **params.params}
        gp = {k: merged[k] for k in GLINT_KEYS if k in merged}
        occ = eqx.tree_at(lambda o: o.edge_coeffs, occ0, gp["occulter_edge"]) if "occulter_edge" in gp else None
        lp = glint_log_prior(gp, scales, sigma_opd=SIGMA_OPD, sigma_amp=SIGMA_AMP, sigma_edge=SIGMA_EDGE,
                             sigma_pos_amp=SIGMA_POS_AMP, occulter=occ)
        return _loss_fn(params, exposures, model) - lp
    fitting.loss_fn = loss_with_prior
    jax.clear_caches()


def save(path, obj):
    """atomic np.save: a job killed mid-write never leaves a truncated stage file"""
    with open(path + ".tmp", "wb") as f:
        onp.save(f, obj, allow_pickle=True)
    os.replace(path + ".tmp", path)


def finite(tree):
    return all(onp.all(onp.isfinite(onp.asarray(x))) for x in jax.tree.leaves(tree))


def rim_opd_rms(p):
    """rms (nm) of the rim OPD profile on the ring the prior uses (edge_profile defaults = glint_log_prior's)"""
    return float(np.sqrt(np.mean(edge_profile(p["occulter_rim_opd"], rim_scales)**2)))


# %%
# Resume after the last saved stage (its params file is written last)
names = [s[0] for s in stages]
done = [n for n in names if os.path.exists(f"{out}/{n}_params.npy")]
cur = jax.tree.map(onp.asarray, params)
start = 0
if done:
    cur = onp.load(f"{out}/{done[-1]}_params.npy", allow_pickle=True).item()
    start = names.index(done[-1]) + 1
    print("resuming after", done[-1])

for name, opt, n, chromatic in stages[start:]:
    print(f"=== {name}: {n} steps, chromatic={chromatic}, free {list(opt)}")
    if name == INJECT_NAN:
        cur["fnumber"] = onp.asarray(onp.nan)
    model = make_model(cur, chromatic)
    install_prior(model)
    opt_params = set_array({k: cur[k] for k in opt})

    if name != "s5":
        # each stage's first update is zero by design (zdx.delay(lr, 0) is 0 at count 0), so losses[0] == losses[1]
        losses, history = optimise_new(opt_params, model, exposures, opt, n, mapped=True,
                                       precond_method="gn_probe", precond_kwargs=precond_kwargs)
        new, losses = (history[-1] if len(history) else {}), onp.asarray(losses)
        ok = len(history) == n
        info = {}
    else:
        new, info = optimise_optimistix(opt_params, model, exposures, max_steps=n, return_info=True, mapped=True,
                                        precond_method="gn_probe", precond_kwargs=precond_kwargs)
        losses = onp.asarray([info["loss_start"], info["loss_end"]])
        ok = bool(onp.isfinite(losses).all())
        info = {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in info.items()}
        print("optimistix:", info)

    new = jax.tree.map(onp.asarray, dict(new))
    if not (ok and finite(new) and onp.isfinite(losses).all()):
        print(f"{name}: non-finite loss or parameters after {len(losses)}/{n} steps; NOT saved, stopping")
        sys.exit(1)
    cur = {**cur, **new}
    print(f"{name}: loss {losses[0]:.6g} -> {losses[-1]:.6g}")
    if name in ("s3", "s4"):
        print(f"{name}: rim_opd profile rms {rim_opd_rms(cur):.3g} nm")
    save(f"{out}/{name}_losses.npy", losses)
    if name == "s5":
        save(f"{out}/s5_info.npy", info)
    save(f"{out}/{name}_params.npy", cur)      # last: marks the stage done
    if name == STOP_AFTER:
        sys.exit(0)

# %%
# Final products from the s5 parameters
save(f"{out}/params_final.npy", cur)
model = make_model(cur, stages[-1][3])
stage_losses = {n: onp.load(f"{out}/{n}_losses.npy") for n in names}

fig, axs = plt.subplots(1, len(names), figsize=(4*len(names), 4), layout="compressed")
for ax, n in zip(axs, names):
    ax.plot(stage_losses[n], "o-" if n == "s5" else "-")
    ax.set(title=n, xlabel="step" if n != "s5" else "start / end", yscale="log")
fig.savefig(f"{out}/losses.png")

chi2 = {}
for exp in exposures:
    im = onp.asarray(exp.fit(model, exp))
    z = onp.where(exp.bad, onp.nan, (exp.data - im)/exp.err)
    chi2[exp.key] = float(onp.nanmean(z**2))
    onp.savez(f"{out}/residuals_{exp.key}.npz", data=exp.data, model=im, err=exp.err, bad=exp.bad, z=z)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="compressed")
    v = onp.nanpercentile(exp.data, 99)
    for ax, a, t, kw in ((axs[0], exp.data, "data", dict(vmin=0, vmax=v)), (axs[1], im, "model", dict(vmin=0, vmax=v)),
                         (axs[2], z, "(data - model)/err", dict(vmin=-5, vmax=5, cmap="RdBu_r"))):
        fig.colorbar(ax.imshow(a, **kw), ax=ax)
        ax.set_title(f"{exp.key} {t}")
    fig.savefig(f"{out}/residuals_{exp.key}.png")

# Linear-chromatic loss at the s4 parameters: s5's loss_start is the exact loss at the same point (same prior)
p4 = onp.load(f"{out}/s4_params.npy", allow_pickle=True).item()
m4 = make_model(p4, "linear")
install_prior(m4)
s4_lin = float(eqx.filter_jit(lambda p, b, m: fitting.loss_fn(ModelParams(p), b, m))(
    set_array({"occulter_rim_opd": p4["occulter_rim_opd"]}), make_batch(exposures, m4), m4))
s5_info = onp.load(f"{out}/s5_info.npy", allow_pickle=True).item()

amp = edge_profile(cur["occulter_rim_amp"], rim_scales)
ampg = logamp_guard(amp)       # the log-amplitude the model transmits (positive values are dead)
summary = {
    "exposures": ROOTNAMES, "occulter": OCCULTER, "dry_run": DRY_RUN,
    "loss": {n: {"first": float(l[0]), "last": float(l[-1]), "steps": len(l)} for n, l in stage_losses.items()},
    "polish": s5_info,
    "chi2_red": chi2,
    "rim_opd_rms_nm": rim_opd_rms(cur),
    "rim_opd_rms_nm_after": {n: rim_opd_rms(onp.load(f"{out}/{n}_params.npy", allow_pickle=True).item()) for n in ("s3", "s4")},
    "s4_loss_linear": s4_lin, "s4_loss_exact": float(s5_info["loss_start"]),
    "s4_loss_linear_minus_exact": s4_lin - float(s5_info["loss_start"]),
    "rim_amp_raw_rms": float(np.sqrt(np.mean(amp**2))), "rim_amp_raw_abs_max": float(np.max(np.abs(amp))),
    "rim_amp_frac_positive": float(np.mean(amp > 0)),
    "rim_logamp_guarded_rms": float(np.sqrt(np.mean(ampg**2))), "rim_logamp_guarded_min": float(np.min(ampg)),
    **{k: onp.asarray(cur[k]).tolist() for k in ("occulter_radius", "occulter_ellipticity", "occulter_edge", "occulter_rotation") if k in cur},
}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
