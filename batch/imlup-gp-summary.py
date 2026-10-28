# %%
"""Aggregate imlup-gp-inject's array tasks (no GPU needed: `python imlup-gp-summary.py` in batch/ once they finish) and
check the plan's pass criteria (notebooks/gp-plan.md, phase 2) for the "gp" truth:
  the evidence optimum within one grid cell (a factor 2) of the true (sigma, length) in >= 75% of seeds;
  the 16-84% grid-posterior intervals cover the truth at roughly their nominal rate;
  the Laplace z-scores of the true log10 disc (data-constrained pixels) have std ~ 1;
  the noise scales s ~ 1.
For "ref" and "planet" it reports the same diagnostics, and the planet's recovered aperture flux.
"""
import glob
import os
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = "imlup-gp-inject"
rows = []
for path in sorted(glob.glob(f"{out}/*/summary.npz"), key=lambda p: int(p.split("/")[-2])):
    with numpy.load(path, allow_pickle=True) as f:
        rows.append({k: f[k] for k in f.files})
print(f"{len(rows)} tasks")

for truth in ("gp", "ref", "planet"):
    rs = [r for r in rows if str(r["truth"]) == truth]
    if not rs:
        continue
    print(f"\n== {truth} ({len(rs)} seeds)")
    s = numpy.stack([r["s"] for r in rs])
    zstd = numpy.array([r["zscore"][r["constrained"]].std() for r in rs])
    z1 = numpy.array([numpy.mean(numpy.abs(r["zscore"][r["constrained"]]) < 1) for r in rs])
    print(f"noise scale s: mean {s.mean():.3f}, range {s.min():.3f}-{s.max():.3f}")
    print(f"z-score std (constrained pixels): mean {zstd.mean():.3f}, range {zstd.min():.3f}-{zstd.max():.3f}; "
          f"fraction |z| < 1: {z1.mean():.3f} (Gaussian 0.683)")
    for r in rs:
        print(f"  seed {int(r['seed'])}: best sigma {float(r['sigma_best']):.3g}, length {float(r['length_best']):.3g}; "
              f"sigma 16/84% {numpy.round(r['sigma_q'][[0, 2]], 3)}, length {numpy.round(r['length_q'][[0, 2]], 3)}; "
              f"edge weight {float(r['edge_weight']):.3f}")
    if truth == "gp":
        st, lt = float(rs[0]["sigma_t"]), float(rs[0]["length_t"])
        near = numpy.array([abs(numpy.log2(float(r["sigma_best"])/st)) <= 1 and
                            abs(numpy.log2(float(r["length_best"])/lt)) <= 1 for r in rs])
        cov_s = numpy.mean([r["sigma_q"][0] <= st <= r["sigma_q"][2] for r in rs])
        cov_l = numpy.mean([r["length_q"][0] <= lt <= r["length_q"][2] for r in rs])
        print(f"PASS criteria: within a grid cell {near.mean():.2f} (>= 0.75: {near.mean() >= 0.75}); "
              f"68% coverage sigma {cov_s:.2f}, length {cov_l:.2f} (nominal 0.68); "
              f"z-score std {zstd.mean():.2f} (~1); s {s.mean():.3f} (~1)")
    if truth == "planet":
        for r in rs:
            print(f"  seed {int(r['seed'])}: aperture flux {float(r['aperture_mean']):.4g} +- "
                  f"{float(r['aperture_std']):.2g} (true {float(r['aperture_true']):.4g}, of which disc "
                  f"{float(r['aperture_ref']):.4g}); "
                  f"z {(float(r['aperture_mean']) - float(r['aperture_true']))/float(r['aperture_std']):.2f}")

fig, axs = plt.subplots(1, 2, figsize=(20, 9))
for truth, m in (("gp", "o"), ("ref", "s"), ("planet", "^")):
    rs = [r for r in rows if str(r["truth"]) == truth]
    if rs:
        axs[0].scatter([float(r["length_best"]) for r in rs], [float(r["sigma_best"]) for r in rs], marker=m, s=150,
                       label=truth, alpha=0.6)
        axs[1].hist(numpy.concatenate([r["zscore"][r["constrained"]] for r in rs]), bins=numpy.linspace(-5, 5, 51),
                    histtype="step", density=True, label=truth, lw=2)
gp_rows = [r for r in rows if str(r["truth"]) == "gp"]
if gp_rows:
    axs[0].plot(float(gp_rows[0]["length_t"]), float(gp_rows[0]["sigma_t"]), "k*", ms=30, label="gp truth")
x = numpy.linspace(-5, 5, 201)
axs[1].plot(x, numpy.exp(-x**2/2)/numpy.sqrt(2*numpy.pi), "k--", label="N(0, 1)")
axs[0].set(xscale="log", yscale="log", xlabel="length (px)", ylabel="sigma (dex)", title="Evidence-selected")
axs[1].set(xlabel="z-score of the true log10 disc", title="Laplace calibration (constrained pixels)")
for ax in axs:
    ax.legend()
fig.tight_layout()
fig.savefig(f"{out}/summary.png")
print(f"\nsaved {out}/summary.png")
