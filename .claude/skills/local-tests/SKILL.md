---
name: local-tests
description: How to run tests, benchmarks and any JAX/dLux script for this repo locally without running the laptop out of RAM. Use whenever executing Python from this repo (tests, equivalence checks, notebook cells run as scripts, benchmarks, fits), and before launching subagents that will run code.
---

# Running code locally under a memory budget

This repo's forward model (JAX + dLux, wf 512 x 512, 80 x 4 oversampled PSF, many wavelengths) can easily
allocate tens of GB, especially under `jax.grad`, `jacfwd`, Hessians or large `vmap`s. The machine must
never run out of RAM. **Every Python process that imports JAX or the repo modules must be run through
`run_budget.py`** in this directory. Never call `.venv/bin/python` on such a script directly.

## The runner

```bash
cd /Users/haydengreer/PhD/hst/coron && \
  .venv/bin/python .claude/skills/local-tests/run_budget.py --est-gb N [--timeout S] /abs/path/script.py [args...]
```

- `--est-gb N` (required): the memory reserved for the job. It is also a hard cap: if the job's resident
  memory (including child processes) exceeds N GB it is killed (`exit=137`, `KILLED: RSS ... > reservation`).
- `--timeout S`: wall-clock limit in seconds (default 1800). The job is killed on expiry.
- `--budget-gb B`: total budget shared by all jobs (default and maximum 20).
- The script runs with **cwd = `<repo>/batch`** and the repo's `.venv` python. Repo modules are found
  via `sys.path.insert(0, '..')` (or an absolute path) in the script; data paths like `../data/...` resolve
  from `batch/`. To run as a notebook would (relative `bad_map.npy`, `params.npy`), `os.chdir` to
  `../notebooks` at the top of the script.
- It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and 3 CPU
  threads, so RSS reflects real usage.
- The final line reports `exit=<code> peak_rss=<GB> wall=<s>`; use it to calibrate future estimates.

### Concurrency
Jobs reserve their `--est-gb` in a shared ledger (`$TMPDIR/coron_mem_ledger.json`, file-locked). Jobs
run concurrently while the sum of reservations is <= 20 GB; others wait (`after Ns wait`). Because each
job is killed if it exceeds its own reservation, the total can never exceed the budget. Dead processes
are pruned from the ledger automatically, so a crashed job never leaks its reservation. Several agents
or background jobs can therefore safely run tests at once, as long as each uses the runner.

## Choosing `--est-gb`

Start small and calibrate from the reported `peak_rss`; set the next estimate to ~1.3 x the peak.
Reference peaks measured on this repo:

| Workload | Peak |
|---|---|
| import + tiny model (wf 64-128, n_modes 4-8, 2 wavelengths, 16 px) | 0.6-2 GB |
| plot_comparison_detailed, wf 128 | ~2 GB |
| tiny-model equivalence tests incl. preconditioner / a few optimiser steps | 2.5-4 GB |
| production forward (wf 512, wid 80, oversample 4, 3 wavelengths) | ~2-3 GB |
| forward-mode Jacobian, production, batch_size 2 (~2000-4000 columns) | 3-6.5 GB |
| exact resolved-source PSF stacks / large interpolated-source configs | 4-10 GB |

If a job is killed for memory, do not just raise the estimate: first shrink the problem (below).

## Keep tests small

- Develop and verify on tiny configs first: `NICMOSCoronagraph(64 or 128, 16, 2, n_modes=4-8,
  n_zernikes=4)`, `nwavels` 2-3, one exposure cropped to 16 px, few params. Do a single production-size
  run only when the result depends on it.
- Batch generously: `jax.lax.map(..., batch_size=1-4)` rather than `vmap` over large axes;
  `plotting.model_jacobian(..., batch_size=2)`; `NICMOSCoronagraph(..., wl_batch=k, remat=True)` for
  gradients at production size.
- Prefer short scripts (seconds to a few minutes) and cache expensive intermediates (Jacobians, PSF stacks)
  with `np.save` in a scratch directory so analysis iterations don't recompute them.
- Write test scripts and outputs to a scratch directory, never into the repo (in particular not into
  `batch/`, which is the cwd: always save with absolute paths).

## Correctness checks to include

- Equivalence tests compare old vs new outputs (images, loglike, optics leaves) with exact equality or a
  stated tolerance, not just "it runs". To import a proposed copy of a module, put its directory first on
  `sys.path`, then the repo, and assert `module.__file__` points where you expect.
- JAX 0.10.2 on this machine has two known miscompilations:
  - jitted `jax.jvp` with compile-time-constant tangents is wrong: pass tangents as jit arguments;
  - jitted reverse-mode gradients for `primary_rot`, `primary_shear` and `primary_spider` are wrong:
    check those against forward mode (`plotting.model_jacobian`) or central finite differences.

## Executing notebooks

`nbconvert` (7.17) and `nbclient` are installed in the venv, and the venv's `python3` kernel is the
default, so notebooks can be executed headlessly through the runner (`-m nbconvert` is passed to the
venv python; the kernel is a child process, so its memory counts against `--est-gb`):

```bash
cd /Users/haydengreer/PhD/hst/coron && \
  .venv/bin/python .claude/skills/local-tests/run_budget.py --est-gb N --timeout S \
    -m nbconvert --to notebook --execute /abs/path/notebooks/<name>.ipynb \
    --output-dir /abs/scratch/dir --output <name>_out --ExecutePreprocessor.timeout=-1
```

- Use **absolute paths**: the runner's cwd is `batch/`, so relative notebook paths fail (exit 255).
- The kernel's cwd is the notebook's own directory, so its `sys.path.insert(0, '..')` and relative data
  paths (`bad_map.npy`, `params.npy`, `../data/...`) behave as in Jupyter. A notebook copied elsewhere
  will not find the repo modules.
- **Always write the executed copy elsewhere** with `--output-dir` (never `--inplace`), so the user's
  notebook and its outputs are untouched; read results from the output copy (e.g. extract `image/png`
  outputs with `json` + `base64` and view them).
- Execution stops at the first failing cell (deliberate `stop` cells included); add
  `--allow-errors` to run past them.
- Full production notebooks (fits, Jacobians) take many minutes and several GB: for development, run a
  reduced copy (smaller `wf_wid`, `n_modes`, `nwavels`, epochs) placed in `notebooks/` under a temporary
  name, and delete it afterwards. Alternatively, `exec` selected cells' source from a script (Agg backend,
  `plt.show` replaced by `savefig`) under the runner.

## Common data paths (from `batch/`)
- Calibrator/science data: `../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/` (the scripts'
  `../data/data/` exists only on the cluster).
- Hole flats: `../data/NICMOS-LAPL-DD2/LAPL_HOLEFLATS_DD2/`.
- Best-fit calibrator params: `../notebooks/params.npy` (or `params_amp.npy` for amplitude fits),
  `np.load(..., allow_pickle=True).item()`.
