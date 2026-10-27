# Instructions for agents working in this repo

## Do not modify core files without explicit direction

Do not edit, refactor or "clean up" the following unless the user explicitly asks for that specific change:

- **Core non-notebook files**: every `.py` module at the repo root (e.g. `models.py`, `apertures.py`, `detectors.py`,
  `fitting.py`, `plotting.py`, `precond.py`, `spectra.py`, `stats.py`, `filters.py`) and the batch scripts in `batch/`.
- **Core notebooks**: established analysis notebooks such as `notebooks/eigenbasis.ipynb` (and the other existing
  notebooks the user relies on).

New functionality for a task (a new fit class, a helper, a model variant) goes in the notebook or scratch script you
are writing for that task, not in a core module, unless the user says otherwise. If a change to a core file seems
necessary, propose it and wait for approval.
