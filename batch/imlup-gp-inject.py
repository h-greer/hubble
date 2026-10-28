# %%
"""Injection-recovery for the GP disc prior's evidence-chosen hyperparameters (notebooks/gp-plan.md, phase 2), on IM Lup's
linear response (imlup-gp/response.npz, from imlup-gp.py stage B; no optics are evaluated here). Array task
idx = truth * n_seeds + seed simulates data y_e = A_e + R_e d_true + err * N(0, 1) for one of
  0 "gp"      a Matern GP draw at known (sigma_t, length_t, order 2), level at the reference level;
  1 "ref"     imlup-adi's L1/TSV disc (not GP-distributed: a robustness test);
  2 "planet"  the L1/TSV disc plus a point source (one disc pixel) at ~1" with single-pixel SNR planet_snr,
then estimates the noise scales, runs the evidence grid (order 2, flat mean) and records the selected hyperparameters,
their grid-posterior intervals, the Laplace z-scores of the true log10 disc and the planet's recovered flux, in
imlup-gp-inject/{idx}/summary.npz. imlup-gp-summary.py aggregates them.
"""
import sys
sys.path.insert(0, '..')

# %%
import jax.numpy as np
import jax.random as jr
import jax
import numpy
import os
import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from gp import *
from gp import _save_npz

out = "imlup-gp-inject"
resp_path = "imlup-gp/response.npz"
ref_path = "imlup-adi/params_amp.npy"

truths = ["gp", "ref", "planet"]
n_seeds = 8
sigma_t, length_t = 0.5, 1.6       # dex, px: the "gp" truth, inside the starting grid
planet_snr = 10.                   # single-pixel Fisher SNR of the injected point source
planet_radius, planet_pa = 1.0, 45.  # arcsec from the centre, degrees in the grid frame
pitch = 0.0864                       # arcsec / disc pixel

# as imlup-gp.py
sigmas0 = [0.25, 0.5, 1., 2.]
lengths0 = [0.4, 0.8, 1.6, 3.2, 6.4]
level_sd = 1.
n_iter = 200

idx = int(sys.argv[1])
truth, seed = truths[idx // n_seeds], idx % n_seeds
task = f"{out}/{idx}"
os.makedirs(task, exist_ok=True)
if os.path.exists(f"{task}/summary.npz"):
    print(f"{task}/summary.npz exists; nothing to do")
    sys.exit(0)
t0 = time.perf_counter()

# %%
if not os.path.exists(resp_path):
    raise FileNotFoundError(f"{resp_path} missing: run imlup-gp.py (stage B) first")
resp = LinearResponse.load(resp_path)
w = resp.wid
d_ref = 10**numpy.asarray(next(iter(numpy.load(ref_path, allow_pickle=True).item()["resolved"].values())))
assert d_ref.shape == (w, w)
level_mu = reference_level(d_ref)
lp = (level_mu, level_sd)

k_truth, k_noise, k_samples = jr.split(jr.PRNGKey(1000 + idx), 3)
planet_pix, planet_flux = None, 0.
if truth == "gp":
    prior_t = gp_prior(w, sigma_t, length_t, 2., None, lp)
    d_true = prior_t.distribution(jr.normal(k_truth, (w*w + 1,)).at[-1].set(0.))
else:
    d_true = np.asarray(d_ref)
if truth == "planet":
    c = (w - 1)/2
    r = planet_radius/pitch
    row = int(round(c + r*numpy.sin(numpy.deg2rad(planet_pa))))
    col = int(round(c + r*numpy.cos(numpy.deg2rad(planet_pa))))
    planet_pix = (row, col)
    j = row*w + col
    planet_flux = float(planet_snr/np.sqrt(resp.G[:, j, j].sum()))
    d_true = d_true.at[row, col].add(planet_flux)

sim = simulate_response(resp, d_true, k_noise)

# %%
# noise scales (truth: 1), then the evidence grid at those scales
prior0 = gp_prior(w, 1., 1.6, 2., None, lp)
s, theta_s, _ = gp_error_scale(sim, prior0)
beta = 1/s**2
grid = gp_evidence_grid(sim, sigmas0, lengths0, 2., None, lp, beta, outdir=task, tag="grid", n_iter=n_iter,
                        theta0=theta_s, verbose=False)
post = gp_hyper_posterior(grid)
i, j = grid["best"]
sig_b, ell_b = float(grid["sigmas"][i]), float(grid["lengths"][j])

# Laplace posterior at the best point: z-scores of the true log10 disc, where the data constrain it
prior_b = gp_prior(w, sig_b, ell_b, 2., None, lp)
theta_b = np.asarray(grid["thetas"][i, j])
lap = gp_laplace(sim, prior_b, theta_b, beta)
log_std = gp_log10_std(prior_b, lap)
prior_std = gp_log10_std(prior_b, (np.zeros(w*w + 1), np.eye(w*w + 1)))
zscore = numpy.asarray((np.log10(d_true) - prior_b.log10_image(theta_b))/log_std)
constrained = numpy.asarray(log_std < 0.5*prior_std)

out_planet = {}
if planet_pix is not None:
    # 3 x 3 aperture flux about the planet pixel, from Laplace draws (the disc's own flux there included in the truth)
    ds = jax.vmap(prior_b.distribution)(gp_laplace_samples(theta_b, lap, k_samples, n=400))
    sl = (slice(planet_pix[0] - 1, planet_pix[0] + 2), slice(planet_pix[1] - 1, planet_pix[1] + 2))
    ap = numpy.asarray(ds[:, sl[0], sl[1]].sum((1, 2)))
    out_planet = dict(planet_pix=planet_pix, planet_flux=planet_flux,
                      aperture_true=float(d_true[sl].sum()), aperture_ref=float(np.asarray(d_ref)[sl].sum()),
                      aperture_mean=ap.mean(), aperture_std=ap.std())

_save_npz(f"{task}/summary.npz", idx=idx, truth=truth, seed=seed, sigma_t=sigma_t if truth == "gp" else numpy.nan,
            length_t=length_t if truth == "gp" else numpy.nan, s=numpy.asarray(s), sigma_best=sig_b,
            length_best=ell_b, sigma_q=post["sigma_q"], length_q=post["length_q"], edge_weight=post["edge_weight"],
            sigmas=grid["sigmas"], lengths=grid["lengths"], logZ=grid["logZ"], converged=grid["converged"],
            zscore=zscore, constrained=constrained, **out_planet)
print(f"task {idx} ({truth}, seed {seed}): s {numpy.round(numpy.asarray(s), 3)}; best sigma {sig_b:.3g} dex, "
      f"length {ell_b:.3g} px; z-score std (constrained pixels) {zscore[constrained].std():.3f} over "
      f"{constrained.sum()} px; {time.perf_counter() - t0:.0f} s")
