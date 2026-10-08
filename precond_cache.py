"""
Disk cache + per-exposure splitting for preconditioner diagonals.

The estimators live in `precond.DIAG_METHODS`; this module decides which exposures each parameter
leaf needs, runs the estimator on the smallest loss that contains them, and caches every block.

For a leaf params[name][k], the "users" are the exposures e with e.fit.get_key(e, name) == k.
One user: the leaf is exposure-specific and is computed from that exposure's loss alone.  Several
users: it is shared and computed on exactly that exposure set.  No users, get_key failing, or a
top-level (non-dict) parameter: shared by all exposures.  Leaves with the same users form a block,
and a block is one cache entry.

Cache key: method + result-affecting kwargs, x64, source fingerprint, library versions, the model
(without params), the block's exposures, and the block / relevant leaf shapes.  Parameter values
are NOT in the key: the entry stores the relevant values it was computed at, and is reused only if
they are within `value_tol` (scale-aware, see `_distance`); otherwise it is recomputed and overwritten.
"""
import os
import re
import json
import copy
import hashlib
import inspect
import tempfile
import importlib.metadata

import numpy as onp
import jax
import jax.numpy as np
import jax.tree_util as jtu

import precond
from models import ModelParams

CACHE_VERSION = 3
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tmp_precond_cache")

# Source files that define the loss: editing any of them invalidates the cache.
_CODE_FILES = ["models.py", "stats.py", "apertures.py", "detectors.py", "spectra.py"]
# Method kwargs that only change memory use, not the result.
_MEMORY_KWARGS = {"batch_size", "chunk", "probe_batch"}

# Length scale for a relevant leaf without a diagonal in the block; only the magnitude matters.
DEFAULT_ATOL = {
    "cold_mask_rot": 0.1, "primary_rot": 0.1, "cold_mask_shift": 0.1, "cold_mask_shear": 0.01,
    "primary_shear": 0.01, "cold_mask_scale": 0.01, "primary_tilt": 0.01, "cold_mask_tilt": 0.01,
    "primary_spider": 0.01, "fnumber": 0.5, "anisotropy": 1e-3, "bias": 1.0,
    "occulter_radius": 0.01, "occulter_coeffs": 0.01, "outer_radius": 0.01,
    "secondary_radius": 0.01, "spider_width": 0.01, "spectrum": 1.0,
}
GENERIC_ATOL = 1e-3


# ------------------------------------------------------------------ fingerprints
def _digest(*trees):
    """sha256 of pytrees: structure plus leaf bytes (memory addresses in reprs are masked)."""
    mask = lambda s: re.sub(r"0x[0-9a-fA-F]+", "0x", str(s))
    h = hashlib.sha256()
    for tree in trees:
        leaves, treedef = jtu.tree_flatten(tree)
        h.update(mask(treedef).encode())
        for leaf in leaves:
            try:
                a = onp.asarray(leaf)
                assert a.dtype != object and not isinstance(leaf, (str, bytes))
                h.update(f"{a.dtype}{a.shape}".encode() + onp.ascontiguousarray(a).tobytes())
            except Exception:  # strings, None, arbitrary static objects
                h.update(mask(repr(leaf)).encode())
    return h.hexdigest()


def _code_fingerprint():
    here = os.path.dirname(os.path.abspath(__file__))
    files = [precond.__file__] + [os.path.join(here, f) for f in _CODE_FILES]
    h = hashlib.sha256()
    for f in files:
        if os.path.exists(f):
            h.update(open(f, "rb").read())
    return h.hexdigest()


def _library_versions():
    out = {}
    for pkg in ("jax", "jaxlib", "dLux", "equinox", "zodiax"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _json_default(x):
    """Canonical form of non-JSON method kwargs (arrays, typed PRNG keys)."""
    if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
    try:
        return onp.asarray(x).tolist()
    except Exception:
        return str(x)


def _method_kwargs(method, kwargs):
    """The method's kwargs with signature defaults applied, minus memory-only ones."""
    ba = inspect.signature(precond.DIAG_METHODS[method]).bind_partial(**kwargs)
    ba.apply_defaults()
    return {k: v for k, v in sorted(ba.arguments.items()) if k not in _MEMORY_KWARGS}


# ------------------------------------------------------------------ block structure
def _leaf(p, name, k):
    return p[name] if k is None else p[name][k]


def _get_key(e, name):
    try:
        return e.fit.get_key(e, name)
    except Exception:
        return None  # unknown -> treated as shared by all


def _group_leaves(p, exposures, per_exposure):
    """{frozenset(user exposure idx): [(name, key or None), ...]} in params order."""
    everyone = frozenset(range(len(exposures)))
    groups = {}
    for name, val in p.items():
        if not isinstance(val, dict):
            groups.setdefault(everyone, []).append((name, None))
            continue
        keys = [_get_key(e, name) for e in exposures]
        for k in val:
            users = frozenset(i for i, kk in enumerate(keys) if kk == k)
            if not per_exposure or None in keys or not users:
                users = everyone
            groups.setdefault(users, []).append((name, k))
    return groups


def _relevant_leaves(mp, es):
    """Leaves of the model parameters `mp` that the exposures `es` read."""
    rel = []
    for name, val in mp.items():
        if not isinstance(val, dict):
            rel.append((name, None))
            continue
        keys = [_get_key(e, name) for e in es]
        rel += [(name, k) for k in val if None in keys or k in keys]
    return rel


def _atol(name, diag=None):
    """Length scale of a leaf: the curvature length 1/sqrt(mean diag) when its diagonal is known."""
    if diag is not None:
        d = onp.asarray(diag, dtype=onp.float64).ravel()
        d = d[onp.isfinite(d) & (d > 0)]
        if d.size:
            return float(1.0 / onp.sqrt(d.mean()))
    return float(DEFAULT_ATOL.get(name, GENERIC_ATOL))


def _distance(x, x0, sizes, atols):
    """Scale-aware distance between current (x) and stored (x0) relevant-parameter vectors:
    max over leaves of rms(x - x0) / (rms(x0) + atol).  Per leaf, so a huge leaf (spectrum ~1e4)
    cannot hide an O(1) one; atol makes it meaningful near 0 (angles, biases) and is a relative
    change for large values.  Non-finite values give inf."""
    if x.shape != x0.shape or not (onp.all(onp.isfinite(x)) and onp.all(onp.isfinite(x0))):
        return onp.inf
    cuts = onp.cumsum(sizes)[:-1]
    worst = 0.0
    for a, a0, atol in zip(onp.split(x, cuts), onp.split(x0, cuts), atols):
        if a.size:
            worst = max(worst, float(onp.sqrt(onp.mean((a - a0) ** 2)) / (onp.sqrt(onp.mean(a0 ** 2)) + atol)))
    return worst


# ------------------------------------------------------------------ disk IO
def _write_entry(path, leaves, arrays, values, sizes, atols):
    # write-then-rename so concurrent readers/writers never see a partial file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"leaf_{i}": a for i, a in enumerate(arrays)}
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            onp.savez(fh, values=values, sizes=sizes, atols=atols, **payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_entry(path, n_leaves):
    try:
        with onp.load(path) as z:
            return [z[f"leaf_{i}"] for i in range(n_leaves)], z["values"], z["sizes"], z["atols"]
    except Exception:
        return None  # missing, corrupt, or from an older format


# ------------------------------------------------------------------ canonicalisation
def _canonical_exposures(es):
    """Copies of `es` with static fields that do not affect the loss replaced by placeholders, so
    jit sees the same static structure for different exposures of equal shape (else every exposure
    recompiles).  `target` -> "t{m}" by first appearance, so shared targets still share get_key keys."""
    targets, out = {}, []
    for j, e in enumerate(es):
        c = copy.copy(e)
        object.__setattr__(c, "filename", f"e{j}")
        object.__setattr__(c, "target", targets.setdefault(e.target, f"t{len(targets)}"))
        for f, v in (("mjd", "0"), ("exptime", "0"), ("wcs", None), ("hdr", None)):
            if hasattr(c, f):
                object.__setattr__(c, f, v)
        out.append(c)
    return out


def _canonical_diag_call(diag_fn, sub, es, model, kw):
    """Run diag_fn on exposures with canonical names and params dict keys renamed to match
    (model.params restricted to what `es` read), then map the result back to the original keys."""
    ces = _canonical_exposures(es)
    table = {}  # name -> {original key: canonical key}
    for name, val in model.params.items():
        if isinstance(val, dict):
            for e, c in zip(es, ces):
                ko, kc = _get_key(e, name), _get_key(c, name)
                if ko is not None and kc is not None:
                    table.setdefault(name, {})[ko] = kc

    def rename(p, table, drop=False):
        out = {}
        for n, v in p.items():
            m = table.get(n)
            out[n] = {m.get(k, k): x for k, x in v.items() if k in m or not drop} if isinstance(v, dict) and m else v
        return out

    cmodel = model.set("params", rename(model.params, table, drop=True))
    d = diag_fn(rename(sub, table), ces, cmodel, **kw)
    return rename(dict(d), {n: {v: k for k, v in m.items()} for n, m in table.items()})


# ------------------------------------------------------------------ public API
def get_diag(params, exposures, model, method="gn_probe", cache_dir=DEFAULT_CACHE_DIR,
             per_exposure=True, refresh=False, match="values", value_tol=0.03, verbose=False,
             **method_kwargs):
    """
    Diagonal preconditioner statistics for `params` (dict or ModelParams, a subset of model.params).

    method        "hessian" | "gn_exact" | "gn_probe" (precond.DIAG_METHODS); kwargs go to it.
    cache_dir     None disables the disk cache.
    per_exposure  split into per-exposure / shared blocks (module doc); False: one block.
    refresh       recompute every block and overwrite its entry.
    match         "values": reuse an entry only if the relevant parameters are within value_tol
                  (`_distance`); "structure": reuse whenever model/data/structure match.
    Returns a dict with the structure of `params`.
    """
    assert match in ("values", "structure") and len(exposures) > 0
    exposures = list(exposures)
    p = dict(params.params if isinstance(params, ModelParams) else params)
    diag_fn = precond.DIAG_METHODS[method]

    # make the current values visible to the model so that block subsets see the right ones
    model = ModelParams(p).inject(model)
    mp = model.params

    if cache_dir is not None:
        base = dict(version=CACHE_VERSION, method=method, kw=_method_kwargs(method, method_kwargs),
                    x64=bool(jax.config.x64_enabled), code=_code_fingerprint(),
                    libs=_library_versions(), model=_digest(model.set("params", {})))
        exposure_fp = [_digest([getattr(e, a, None) for a in
                                ("filename", "target", "filter", "exptime", "mjd", "orient", "pam", "data", "err", "bad")],
                               e.fit) for e in exposures]

    groups = _group_leaves(p, exposures, per_exposure)
    out = {}
    for users in sorted(groups, key=lambda s: (len(s), sorted(s))):
        leaves, es = groups[users], [exposures[i] for i in sorted(users)]
        arrays, status = None, "nocache"
        if cache_dir is not None:
            rel = _relevant_leaves(mp, es)
            key = hashlib.sha256(json.dumps(dict(
                base, exposures=sorted(exposure_fp[i] for i in users),
                leaves=[(n, k, onp.shape(_leaf(p, n, k))) for n, k in sorted(leaves, key=str)],
                rel=[(n, k, onp.shape(_leaf(mp, n, k))) for n, k in sorted(rel, key=str)]),
                sort_keys=True, default=_json_default).encode()).hexdigest()[:32]
            path = os.path.join(cache_dir, key + ".npz")
            chunks = [onp.concatenate([onp.asarray(l, dtype=onp.float64).ravel() for l in jtu.tree_leaves(_leaf(mp, n, k))]
                      or [onp.zeros(0)]) for n, k in rel]
            sizes = onp.array([c.size for c in chunks])
            values = onp.concatenate(chunks) if chunks else onp.zeros(0)
            hit = None if refresh else _read_entry(path, len(leaves))
            status = "refresh" if refresh else "miss"
            if hit is not None:
                arrays, values0, sizes0, atols0 = hit
                dist = 0.0 if match == "structure" else _distance(values, values0, sizes0, atols0)
                status = "hit" if dist <= value_tol else "stale"
                if status == "stale":
                    arrays = None
        if arrays is None:
            sub = {}
            for n, k in leaves:
                if k is None:
                    sub[n] = p[n]
                else:
                    sub.setdefault(n, {})[k] = p[n][k]
            d = _canonical_diag_call(diag_fn, sub, es, model, method_kwargs)
            arrays = [onp.asarray(_leaf(d, n, k), dtype=onp.float64) for n, k in leaves]
            if cache_dir is not None:
                if all(onp.all(onp.isfinite(a)) for a in arrays):
                    diags = dict(zip(leaves, arrays))
                    _write_entry(path, leaves, arrays, values, sizes, [_atol(n, diags.get((n, k))) for n, k in rel])
                else:
                    status += "+nonfinite-not-cached"
        if verbose:
            print(f"[precond_cache] exposures={sorted(users)} {status} leaves={[n for n, _ in leaves]}")
        for (n, k), a in zip(leaves, arrays):
            if k is None:
                out[n] = np.asarray(a)
            else:
                out.setdefault(n, {})[k] = np.asarray(a)

    # restore the order of `params`
    return {n: ({k: out[n][k] for k in v} if isinstance(v, dict) else out[n]) for n, v in p.items()}


def get_precond(params, exposures, model, method="gn_probe", damping=1e-3, **kw):
    """precond.precond_from_diag(get_diag(...), damping): C with the structure of params."""
    return precond.precond_from_diag(get_diag(params, exposures, model, method=method, **kw), damping)

