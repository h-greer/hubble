# %%
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import jax
import numpy
import os

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
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chainconsumer as cc

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
from plotting import _signed_panel  # underscore names are not covered by `import *`
from spectra import *

import interpax as ipx

import csv
from datetime import datetime, timedelta

# %%
wf_wid = 512
wid = 64
oversample = 4

pad = 2      # model 2 extra detector pixels on each side, cropped after resampling (see NICMOSDetector)

filt = "F110W"
nwavels = 200
npoly=8

# The 64 px crop (about detector pixel (75, 212)) spans x 43-106, y 180-243; across the calibrators the hole (TARSIAFX/Y)
# spans x 70.6-75.0, y 209.1-212.4, so the farthest crop edge is 35.9 px = 2.73" (x, 0.0760"/px) from the star. Fourier mode k
# of primary_opd puts a speckle at k lambda / 3 m (the 3 m wavefront array): the edge is k = 27.0 at 1.470 um, the red end of
# the grid (bluer wavelengths reach proportionally less far). dLux's even-n grid has sin/cos pairs up to n_modes//2 - 1 = 27
n_modes = 56
n_zernikes = 50

# Fresnel occulter (defocus1/2 at the occulter, defocus3 at the detector), elliptical occulter; the PSF is Hermite-interpolated
# between 20 knot wavelengths across the 200-wavelength spectrum. No pupil amplitude map
optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20, pad=pad)
# optics = NICMOSFresnelCoronagraph(wf_wid, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes, n_nodes=20, amplitude=True, pad=pad)

detector = NICMOSDetector(oversample, wid, pad=pad)

vects = np.load(f"../data/iterative_spectrum_basis_{filt}_{nwavels}.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))


# Calibrator visits: exposures in filt of targets judged suitable in data/calibrators.csv (one copy per
# observation), grouped per SIMBAD object into visits of exposures taken within ~1 day of the previous one
# Skipped: files with an empty error extension (some DD1 data)
rows = [r for r in csv.DictReader(open("../data/calibrators.csv"))
        if r["calibrator"] == "yes" and r["preferred_copy"] == "True" and r["filter"] == filt
        and fits.getheader("../" + r["file_path"], 2)["NAXIS"] > 0]
for r in rows:
    r["time"] = datetime.fromisoformat(r["date_obs"])
rows.sort(key=lambda r: (r["simbad_main_id"], r["time"]))

visits = []
for r in rows:
    if visits and visits[-1][-1]["simbad_main_id"] == r["simbad_main_id"] and r["time"] - visits[-1][-1]["time"] < timedelta(days=1.5):
        visits[-1].append(r)
    else:
        visits.append([r])

# Each roll of a visit is fitted separately
visits = [[r for r in visit if round(float(r["orientat"]), 1) == roll]
          for visit in visits for roll in sorted({round(float(r["orientat"]), 1) for r in visit})]
# At most 4 exposures per run, split as evenly as possible in time order
max_exp = 4
visits = [list(chunk) for visit in visits for chunk in numpy.array_split(visit, -(-len(visit)//max_exp))]
print(f"{len(visits)} calibrator visits")

index = int(sys.argv[1])
visit = visits[index]
target = visit[0]["target"]
print(index, target, visit[0]["simbad_main_id"], visit[0]["date_obs"], [r["rootname"] for r in visit])

out = f"calibrator-joint-{filt}/{index}-{target}"
os.makedirs(out, exist_ok=True)

# Superseded by hole_margin (hole mask from the TARSIAFX/Y header, per exposure, so valid for both DD1 and DD2)
# flatdirs = {
#     "DD1": '../data/NICMOS-LAPL-DD1/archive.stsci.edu/missions/hlsp/laplace/dd1/LAPL/NICMOS-LAPL-DD1/HOLEFLATS/',
#     "DD2": '../data/NICMOS-LAPL-DD2/LAPL_HOLEFLATS_DD2/',
# }

# The cold mask lateral position (shift) and its tilt (a pupil-plane Tilt initialised from the era's
# approximate hole offset, models.hole_tilt), cold-mask OPD and defocus can differ between exposures; rotation, shear, scale and
# distortion stay global. The high-order wavefront (primary_opd) and low-order Zernikes are per exposure
class PerExposureColdMaskFit(SinglePointFit):
    PARAM_KEYS = SinglePointFit.PARAM_KEYS | {"cold_mask_shift": "exposure", "cold_mask_tilt": "exposure",
                                              "primary_opd": "exposure", "cold_mask_opd": "exposure", "defocus3": "exposure"}

# The spectrum is keyed by target, so all exposures of the visit share it
exposures_single = [
    exposure_from_file("../" + r["file_path"], PerExposureColdMaskFit(spectrum_basis, filt), crop=wid, extra_bad=None, hole_margin=0.1)
    for r in visit
]

# %%
params = {
    "spectrum": {},
    "primary_opd": {},
    # "primary_amp": {},
    "primary_low": {},
    "primary_tilt": {},
    "cold_mask_opd": {},
    "cold_mask_tilt": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_shear": {},
    "cold_mask_scale": {},
    "cold_mask_distortion": {},
    "defocus1": {},
    "defocus2": {},
    "defocus3": {},
    "primary_rot": {},
    "primary_shear": {},
    # "jitter": {},

    "bias": {},
    "occulter_radius": 0.8,
    "occulter_ellipticity": np.zeros(2),   # (e1, e2); major axis at atan2(e2, e1)/2 from the occulter-plane x axis
    "fnumber": 45.7,
    "anisotropy": 1.+1e-5,

    "outer_radius": 1.2*0.9768,
    "secondary_radius": 0.365*1.2,
    "spider_width": 0.072*1.2,
    "primary_spider": 0.0256,
    "primary_secondary": 0.330*1.2,
    "primary_outer": 1.2,
    "primary_pad": 0.065*1.2,
    "secondary_shift": np.zeros(2),        # cm, spider-aligned frame; ~1 cm expected from the 7.4' field offset
    "primary_distortion": np.zeros((2, 7)),  # fraction of the 1.2 m radius at the edge
    "distortion": np.zeros((2, 7)),        # detector-plane polynomial distortion
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*100) # matches the data flux at initialisation

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.05, -0.75])*0.075
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = hole_tilt(exp)

    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    # params["primary_amp"][exp.fit.get_key(exp, "primary_amp")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.zeros(16).at[0].set(120.)

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.,10.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 3.2
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -0.6
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.06,-0.06])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    for k in ("defocus1", "defocus2", "defocus3"):
        params[k][exp.fit.get_key(exp, k)] = 0.
    params["cold_mask_distortion"][exp.fit.get_key(exp, "cold_mask_distortion")] = np.zeros((2, 7))
    # params["jitter"][exp.fit.get_key(exp, "jitter")] = 7. # mas; gradient vanishes at 0


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)

# %%
plot_comparison_detailed(model_single, params, exposures_single, percentile=99, wf_size=wf_wid, save=f"{out}/initial-comparison")

# %%
g = 5e-2

things = {
    "primary_opd": adam(2e-2, 50),
    # Delayed until well after the phase (primary_opd starts at 50), so the phase is fitted first
    # and the phase/amplitude degeneracy is avoided
    # "primary_amp": adam(2e-2, 150),

    "spectrum": sgd(g*1, 0),
    "primary_tilt": sgd(g*1., 0),
    "cold_mask_tilt": sgd(g*1, 0),
    "cold_mask_opd": sgd(g*1., 0),

    "defocus1": sgd(g*1, 0),
    "defocus2": sgd(g*1, 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*1, 0),
    "cold_mask_rot": sgd(g*1., 0.),
    "primary_rot": sgd(g*3., 0),

    "primary_low": sgd(g*1, 0),
    # "jitter": sgd(g*1, 100),

    "cold_mask_scale": sgd(g*1, 0),
    "cold_mask_distortion": sgd(g*3, 0),

    "occulter_radius": sgd(g*1., 0),
    "occulter_ellipticity": sgd(g*3, 300),

    "secondary_radius": sgd(g*1., 0),
    "spider_width": sgd(g*1., 0),
    "primary_spider": sgd(g*1., 0),
    "primary_secondary": sgd(g*1., 0),
    "secondary_shift": sgd(g*1, 0),
    "primary_distortion": sgd(g*3, 0),

    "fnumber": sgd(g*1., 0),
    "anisotropy": sgd(g*1., 0),
    "distortion": sgd(g*1, 0),
}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*5., 15.),
    "cold_mask_tilt": sgd(g*1., 0),
    "cold_mask_opd": sgd(g*0.1, 40),

    "defocus1": sgd(g*1, 60),
    "defocus2": sgd(g*10, 60),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*0.3, 70),
    "cold_mask_rot": sgd(g*0.1, 85),
    "primary_rot": sgd(g*0.03, 85),

    "primary_low": sgd(g*2., 150),

    "cold_mask_shear": sgd(g*0.1, 100),
    "cold_mask_distortion": sgd(g*0.1, 150),

    "primary_shear": sgd(g*1, 100),

    "occulter_radius": sgd(g*1., 120),

    "outer_radius": sgd(g*0.1, 120),
    "secondary_radius": sgd(g*0.1, 120),
    "spider_width": sgd(g*0.1, 120),
    "primary_spider": sgd(g*0.1, 120),
    "primary_secondary": sgd(g*0.1, 120),
    "primary_pad": sgd(g*0.1, 120),
    "primary_outer": sgd(g*0.1, 120),
    "secondary_shift": sgd(g*0.3, 120),
    "primary_distortion": sgd(g*0.1, 150),

    "fnumber": sgd(g*1., 200),
    "anisotropy": sgd(g*2., 200),
}

groups = list(things.keys())
# per_block=False: one preconditioner compile for all blocks; mapped=True (below): lax.map over exposures, so compile time
# does not grow with the number of exposures
precond_kwargs = dict(n_probes=32, chunk=4, per_block=False, exact_max_size=32)

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things_start, 500,
    mapped=True, precond_method="gn_probe", precond_kwargs=precond_kwargs)

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/intermediate-losses.png")

plot_params(params_history, list(things_start.keys()), xw = 5, save=f"{out}/intermediate-params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False, wf_size=wf_wid, save=f"{out}/intermediate-comparison")
np.save(f"{out}/intermediate-params.npy", params_history[-1])
losses_start = losses

# %%
# Keep the stage-1 values of parameters that stage 2 does not fit (stage 2 injects into model_single)
model_single = ModelParams(params_history[-1]).inject(model_single)
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

losses, params_history = optimise_new(
    opt_params, model_single, exposures_single, things, 2000,
    mapped=True, precond_method="gn_probe", precond_kwargs=precond_kwargs)

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"{out}/losses.png")
print("final loss", losses[-1])

plot_params(params_history, groups, xw = 5, save=f"{out}/params")
plot_comparison_detailed(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False, wf_size=wf_wid, percentile=99, save=f"{out}/comparison")

# %%
# # Recovered amplitude where both the primary and the cold mask transmit, as in "Pupil x Cold Mask"
# model = ModelParams(params_history[-1]).inject(model_single)
# exp = exposures_single[0]
# optics = exp.fit.update_optics(model, exp)

# coords = dlu.pixel_coords(wf_wid, optics.diameter)
# primary = optics.primary.transmission(coords, 2.4/wf_wid)
# cold_mask = optics.cold_mask.transmission(coords, 2.4/wf_wid)
# amp = 100*(np.exp(optics.primary_amp.eval_basis()) - 1)

# fig, ax = plt.subplots(figsize=(10, 10), layout='compressed')
# _signed_panel(ax, np.where(primary*cold_mask < .5, np.nan, amp), "Pupil x Cold Mask Amplitude", label="Amplitude (%)", percentile=99)
# fig.savefig(f"{out}/amplitude.png")

# %%
# Recovered stellar spectrum (before the filter) and the detected (filtered) spectrum
exp = exposures_single[0]
spectrum = exp.fit.source.spectrum.set("basis_weights", ModelParams(params_history[-1]).get(exp.fit.map_param(exp, "spectrum")))
wavels = 1e6*spectrum.wavelengths

fig, axs = plt.subplots(1, 2, figsize=(20, 8), layout='compressed')
axs[0].plot(wavels, spectrum.spec_weights())
axs[0].set(xlabel="Wavelength (um)", ylabel="Relative flux", title="Stellar spectrum")
axs[1].plot(wavels, spectrum.weights)
axs[1].set(xlabel="Wavelength (um)", ylabel="Normalised weight", title="Detected spectrum")
fig.savefig(f"{out}/spectrum.png")

# %%
np.save(f"{out}/params_amp.npy", params_history[-1])
# print({k: float(np.abs(v)) for k, v in ModelParams(params_history[-1]).get("jitter").items()}, "mas")

# %%
# Machine-readable fit diagnostics (a dict, loaded with np.load(..., allow_pickle=True).item()), for assessing
# fit quality in bulk across the job array
def stats(z):
    z = z[np.isfinite(z)]
    return {"n": int(z.size), "chi2_red": float(np.mean(z**2)), "z_std": float(np.std(z)), "z_mean": float(np.mean(z)),
            "z_mad": float(1.4826*np.median(np.abs(z - np.median(z)))), "z_absmax": float(np.max(np.abs(z)))}

def convergence(l):
    l = np.asarray(l)
    return {"first": float(l[0]), "last": float(l[-1]), "min": float(np.min(l)), "n_epochs": int(l.size),
            "rel_change_last_10pc": float((l[-max(l.size//10, 2)] - l[-1])/abs(l[-1]))}

final = ModelParams(params_history[-1])
model = final.inject(model_single)
diag = {"index": index, "target": target, "simbad_main_id": visit[0]["simbad_main_id"], "date_obs": visit[0]["date_obs"],
        "delivery": visit[0]["delivery"], "n_exposures": len(exposures_single),
        "loss_stage1": convergence(losses_start), "loss_stage2": convergence(losses),
        "finite_params": bool(all(np.all(np.isfinite(x)) for x in jax.tree.leaves(final.params))),
        "spectrum": np.asarray(final.get(exposures_single[0].fit.map_param(exposures_single[0], "spectrum"))),
        "exposures": []}

for exp, r in zip(exposures_single, visit):
    im = exp.fit(model, exp)
    z = np.where(exp.bad, np.nan, (exp.data - im)/exp.err)
    # Halo: pixels where the model exceeds 1% of its peak; wings: the rest
    halo = im > 0.01*np.nanmax(im)
    get = lambda p: np.asarray(final.get(exp.fit.map_param(exp, p)))
    diag["exposures"].append({
        "rootname": exp.key, "orientat": float(exp.orient), "exptime": float(exp.exptime), "proposal": r["proposal"],
        "all": stats(z), "halo": stats(np.where(halo, z, np.nan)), "wings": stats(np.where(halo, np.nan, z)),
        "flux_ratio": float(np.nansum(np.where(exp.bad, np.nan, im))/np.nansum(exp.data)),
        "frac_bad": float(np.mean(exp.bad)),
        # "jitter_mas": float(np.abs(final.get(exp.fit.map_param(exp, "jitter")))),
        **{p: get(p) for p in ["primary_tilt", "cold_mask_tilt", "cold_mask_shift", "bias", "defocus1", "defocus2"]},
        "primary_opd_rms_nm": float(np.std(final.get(exp.fit.map_param(exp, "primary_opd")))),
        "primary_low_rms_nm": float(np.std(final.get(exp.fit.map_param(exp, "primary_low")))),
    })
    np.save(f"{out}/z_{exp.key}.npy", z)

np.save(f"{out}/diagnostics.npy", diag)
np.save(f"{out}/losses.npy", np.asarray(losses))
np.save(f"{out}/losses_stage1.npy", np.asarray(losses_start))

# %%
# optimistix L-BFGS polish (fitting.optimise_optimistix) of the stage-2 fit, over the same fitted parameters. The
# preconditioner is stage 2's (gn_probe), recomputed at the stage-2 point: the per-exposure primary_opd makes exact
# Gauss-Newton columns expensive here. Stops at steps below atol (~sigma units) or after n_optx steps; a blow-up or a
# worse end point falls back to the stage-2 values. Outputs carry an "optx" suffix; params_amp.npy and
# diagnostics.npy above stay the stage-2 results.
n_optx = 2048
x3, optx_info = optimise_optimistix(params_history[-1], model, exposures_single, max_steps=n_optx, return_info=True,
                                    mapped=True, precond_method="gn_probe", precond_kwargs=precond_kwargs)
final_optx = ModelParams(x3)
model_optx = final_optx.inject(model_single)
np.save(f"{out}/params_amp_optx.npy", x3)
plot_comparison_detailed(model, final_optx, exposures_single, quadrature=False, wf_size=wf_wid, percentile=99, save=f"{out}/comparison-optx")

lines = [f"optimistix: {optx_info['result']} after {optx_info['num_steps']} steps; loss {optx_info['loss_start']:.1f} -> "
         f"{optx_info['loss_end']:.1f} (stage 2 final {losses[-1]:.1f})" + (f"; frozen {optx_info['frozen']}" if optx_info["frozen"] else "")]
for exp, d in zip(exposures_single, diag["exposures"]):
    z = np.where(exp.bad, np.nan, (exp.data - exp.fit(model_optx, exp))/exp.err)
    np.save(f"{out}/z_optx_{exp.key}.npy", z)
    lines.append(f"{exp.key}: chi2_red {stats(z)['chi2_red']:.3f} (stage 2 {d['all']['chi2_red']:.3f})")
print("\n".join(lines)); open(f"{out}/optx.txt", "w").write("\n".join(lines) + "\n")
