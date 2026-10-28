# %%
"""Gaussian-process (Matern) priors for the resolved disc, after virgil (benjaminpope/virgil, fields.py), with the
hyperparameters chosen by the Laplace evidence (empirical Bayes) rather than by hand. Plan: notebooks/gp-plan.md.

Prior. The log10 image is a GP in whitened form: one N(0, 1) coefficient z per DCT-II mode, scaled by the square root of
the Matern power spectrum and inverse transformed,

    log10 I = level + mean + idctn(sqrt(S) * z),   S_jk ∝ (1/length^2 + lambda_jk)^(-order),

with lambda_jk the eigenvalues of the reflecting-boundary grid Laplacian (pixel units). S_00 = 0, so the field has zero
spatial mean and `level` is exactly the spatial mean of log10 I - mean: the log of the geometric-mean brightness, not
of the mean brightness. `mean` is an optional log10 template (default 0). Units and meaning:
  sigma   rms of log10 I in dex, averaged over the grid (sigma_ln = ln(10) sigma = 2.303 sigma);
  length  the Matern scale 1/kappa in grid pixels (0.0864"), NOT the correlation range: for order 2 the correlation
          falls to ~0.14 at ~2.8 length. For length <~ 1 px the grid Laplacian saturates and the field is near white;
  order   nu + 1 in 2D (order 2: Matern nu = 1; order 1: nu = 0, whose pointwise variance depends on the pixel size).
The DCT's reflecting edges are part of the prior: the variance rises towards the box edges (~1.7-2.2x the centre's) and
corners (~3-4.6x), so keep a margin of several `length` between the disc and the box edge. Because sigma^2 is the
grid-averaged variance (edges included, constant mode removed), the variance at the centre is below it (~0.65-0.93
sigma^2, less for larger length), which couples sigma and length somewhat.

Hyperparameters. Never fit sigma / length by MAP jointly with z: in whitened coordinates a larger amplitude always lowers
0.5 |z|^2, so the joint mode inflates the amplitude and over-fits modes the data do not constrain (Ensslin & Frommert 2011;
virgil's fitting.py warns likewise). Instead, at fixed optics the detector image is linear in the disc d = 10**log10 I
(InterpolatedResolvedSource, detector resampling) and the noise is Gaussian with fixed errors, so

    -log L(d) = sum_e [ 0.5 (d^T G_e d - 2 b_e^T d + yy_e) + lognorm_e ],   G_e = R_e^T W_e R_e,

exactly (LinearResponse, built once per optics state with build_linear_response and checked against the full model by
verify_linear_response). Everything below then costs dense n x n linear algebra and no optical renders:
  gp_map              Levenberg-Marquardt MAP of theta = (z, u), level = level_mu + level_sd u, at fixed hyperparameters;
  gp_log_evidence     log Z ~ -nll - 0.5|theta|^2 - 0.5 sum log(1 + mu_i), mu_i the eigenvalues of J^T G J (Laplace;
                      the log-normal posterior is skewed in weakly constrained directions, so this overestimates log Z
                      by a fraction of a nat per such direction, more for larger sigma: validate a sigma preference by
                      injection, or by sampling (gp-plan.md phase 3));
  gp_error_scale      per-exposure noise scale s_e (MacKay: stationary point of log Z in beta_e = 1 / s_e^2);
  gp_evidence_grid    log-spaced (sigma, length) grid, warm-started, grown x2 while the optimum is on an edge, resumable;
  gp_hyper_posterior  the grid as quadrature for p(sigma, length | data) under log-uniform hyperpriors (INLA-style);
  gp_laplace / gp_laplace_samples / gp_log10_std   the Laplace posterior N(theta_map, (I + J^T G J)^-1);
  gp_spectral_diagnostic   posterior power per DCT wavenumber vs the prior (1 per whitened mode);
  simulate_response   injection: the response of simulated data, without any optics.
"""
import os
import warnings

import numpy as onp
import jax
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx

from models import ModelFit, ModelParams, PointResolvedFit
from batching import make_batch

LN10 = float(onp.log(10.0))
HYPER_NAMES = ("sigma", "length", "order", "mean", "level_prior")


# ----------------------------------------------------------------------------- prior
def matern_spectrum(shape, length, sigma, order=2.):
    """Matern power spectrum (variance per DCT-II mode) on `shape`, normalised to a mean pixel variance sigma^2, with the
    constant mode removed (it is `level`). length: scalar or (row, col) for an anisotropic field. All arguments may be
    traced, so changing the hyperparameters never recompiles."""
    length = np.broadcast_to(np.asarray(length, dtype=float), (2,))
    lap = [(2*np.sin(np.pi*np.arange(n)/(2*n)))**2 for n in shape]
    S = (1 + length[0]**2*lap[0][:, None] + length[1]**2*lap[1][None, :])**(-np.asarray(order, dtype=float))
    S = S.at[0, 0].set(0.)
    return sigma**2 * S * S.size / S.sum()


def matern_amplitude(shape, length, sigma, order=2.):
    """sqrt(matern_spectrum), with a finite gradient at the removed constant mode."""
    S = matern_spectrum(shape, length, sigma, order)
    pos = S > 0
    return np.where(pos, np.sqrt(np.where(pos, S, 1.)), 0.)


def gp_field(z, amplitude):
    """Whitened coefficients z -> correlated zero-mean field."""
    return jsp.fft.idctn(amplitude*z, type=2, norm="ortho")


def log10_template(T, floor=1e-3):
    """log10 template for GPResolvedFit(mean=...): log10(max(T / max T, floor)), so 0 at the template's peak."""
    T = np.asarray(T, dtype=float)
    return np.log10(np.maximum(T/np.max(T), floor))


def reference_level(d_ref, mean=None):
    """Prior centre for resolved_level from a reference disc d_ref (e.g. an L1/TSV fit): the spatial mean of
    log10 d_ref - mean, which is what `level` is for a zero-mean field."""
    logd = np.log10(np.asarray(d_ref, dtype=float))
    return float(np.mean(logd - (0. if mean is None else mean)))


class GPResolvedFit(PointResolvedFit):
    """PointResolvedFit with the Matern GP prior on log10 of the disc, always on an InterpolatedResolvedSource.

    Parameters: "resolved" (wid x wid whitened coefficients z, start at zeros) and "resolved_level" (scalar, dex; the
    spatial mean of log10 I - mean), both keyed by target. Hyperparameters are fit leaves (not parameters, never
    optimised): sigma (dex), length (grid px, (row, col)), order, mean (log10 template, wid x wid) and level_prior
    (mean, std) of the Normal prior on resolved_level; change them with set_hyper (no recompilation).
    The loss gains the negative log prior 0.5 |z|^2 + 0.5 ((level - mu) / sd)^2, divided by n_share, the number of
    exposures in the loss that share one target-keyed disc, so that it is counted once in total: set it with
    with_n_share(exposures) on the full list of exposures entering the loss.
    d_override (internal): a disc used instead of the parameters (LinearResponse, verify_linear_response)."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"resolved_level": "target"}
    sigma: jax.Array
    length: jax.Array
    order: jax.Array
    mean: jax.Array
    level_prior: jax.Array
    n_share: int
    d_override: jax.Array | None

    def __init__(self, spectrum_basis, filter, wid, sigma=1., length=1.6, order=2., mean=None, level_prior=(0., 1.),
                 n_share=1, resolved_kwargs=None):
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.sigma = np.asarray(sigma, dtype=float)
        self.length = np.broadcast_to(np.asarray(length, dtype=float), (2,))
        self.order = np.asarray(order, dtype=float)
        self.mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
        assert self.mean.shape == (wid, wid), f"mean template shape {self.mean.shape} != ({wid}, {wid})"
        self.level_prior = np.asarray(level_prior, dtype=float)
        self.n_share = int(n_share)
        self.regulariser = None  # the GP prior replaces the L1/TSV weights; precond.py handles it through n_share
        self.d_override = None

    def amplitude(self):
        return matern_amplitude(self.mean.shape, self.length, self.sigma, self.order)

    def get_distribution(self, model, exposure):
        if self.d_override is not None:
            return self.d_override
        z = model.get(self.map_param(exposure, "resolved"))
        level = model.get(self.map_param(exposure, "resolved_level"))
        return 10**(level + self.mean + gp_field(z, self.amplitude()))

    def neg_log_prior(self, model, exposure):
        z = model.get(self.map_param(exposure, "resolved"))
        level = model.get(self.map_param(exposure, "resolved_level"))
        mu, sd = self.level_prior
        return 0.5*np.sum(z**2) + 0.5*((level - mu)/sd)**2

    def loglike(self, model, exposure, per_pix=False, return_im=False):
        ll = ModelFit.loglike(self, model, exposure, per_pix=per_pix, return_im=return_im)
        if return_im or per_pix:
            return ll
        return ll + self.neg_log_prior(model, exposure)/self.n_share


def gp_n_share(exposures):
    """{disc key: number of exposures reading it}."""
    counts = {}
    for e in exposures:
        k = e.fit.get_key(e, "resolved")
        counts[k] = counts.get(k, 0) + 1
    return counts


def with_n_share(exposures):
    """`exposures` with each GPResolvedFit's n_share set to the number of exposures in this list sharing its disc.
    Call it on exactly the list that enters the loss (a subset, e.g. a bootstrap, needs its own call)."""
    counts = gp_n_share(exposures)
    return [eqx.tree_at(lambda e: e.fit.n_share, e, counts[e.fit.get_key(e, "resolved")]) for e in exposures]


def check_n_share(exposures):
    counts = gp_n_share(exposures)
    bad = [(e.key, e.fit.n_share, counts[e.fit.get_key(e, "resolved")]) for e in exposures
           if e.fit.n_share != counts[e.fit.get_key(e, "resolved")]]
    assert not bad, f"GPResolvedFit.n_share does not match the exposures sharing the disc (key, n_share, actual): {bad}"


def set_hyper(exposures, **hyper):
    """`exposures` with GP hyperparameters replaced (sigma, length, order, mean, level_prior; None leaves one alone)."""
    unknown = set(hyper) - set(HYPER_NAMES)
    assert not unknown, f"unknown GP hyperparameters {unknown}; expected some of {HYPER_NAMES}"
    out = []
    for e in exposures:
        fit = e.fit
        for name, v in hyper.items():
            if v is None:
                continue
            v = np.broadcast_to(np.asarray(v, dtype=float), (2,)) if name == "length" else np.asarray(v, dtype=float)
            assert v.shape == getattr(fit, name).shape, f"{name}: shape {v.shape} != {getattr(fit, name).shape}"
            fit = eqx.tree_at(lambda f, name=name: getattr(f, name), fit, v)
        out.append(eqx.tree_at(lambda e: e.fit, e, fit))
    return out


def scale_errors(exposures, s):
    """`exposures` with err multiplied by s (one scale per exposure, e.g. from gp_error_scale)."""
    s = onp.broadcast_to(onp.asarray(s, dtype=float), (len(exposures),))
    return [eqx.tree_at(lambda e: e.err, e, e.err*float(si)) for e, si in zip(exposures, s)]


def check_gp_setup(params, optimisers=None):
    """Raise if resolved_level is missing from the params (or would be silently frozen by an optimiser dict that fits
    "resolved" without it); warn if a hyperparameter is passed to an optimiser (they are not parameters, and their
    joint MAP with z is biased)."""
    p = params.params if isinstance(params, ModelParams) else params
    if "resolved" in p and "resolved_level" not in p:
        raise KeyError("GPResolvedFit needs params['resolved_level'] (a {target key: level} dict)")
    if optimisers is not None:
        if "resolved" in optimisers and "resolved_level" not in optimisers:
            raise KeyError("optimisers fit 'resolved' but not 'resolved_level', which would be silently frozen")
        hyper = [k for k in optimisers if k in HYPER_NAMES]
        if hyper:
            warnings.warn(f"GP hyperparameters {hyper} in the optimisers: their MAP jointly with the disc is biased "
                          "(towards over-fitting); fix them, choose them by the evidence (gp_evidence_grid) or sample them")


def gp_image(params, exposure):
    """The disc 10**log10 I of `exposure` (a GPResolvedFit) at `params` (dict or ModelParams). Use this to plot GP fits:
    params['resolved'] holds the whitened z, not log10 I."""
    p = params.params if isinstance(params, ModelParams) else params
    fit = exposure.fit
    key = fit.get_key(exposure, "resolved")
    return 10**(p["resolved_level"][key] + fit.mean + gp_field(p["resolved"][key], fit.amplitude()))


# ----------------------------------------------------------------------------- surrogate prior (theta = (z, u))
class GPPrior(eqx.Module):
    """The GP prior in the surrogate's coordinates theta = (z.ravel(), u) ~ N(0, I), level = level_mu + level_sd u.
    All leaves are arrays, so jitted functions taking it never recompile when the hyperparameters change."""
    amp: jax.Array
    mean: jax.Array
    level_mu: jax.Array
    level_sd: jax.Array
    sigma: jax.Array
    length: jax.Array
    order: jax.Array

    @property
    def wid(self):
        return self.amp.shape[0]

    def log10_image(self, theta):
        w = self.amp.shape[0]
        return self.level_mu + self.level_sd*theta[-1] + self.mean + gp_field(theta[:-1].reshape(w, w), self.amp)

    def distribution(self, theta):
        return 10**self.log10_image(theta)

    def theta(self, z, level):
        return np.concatenate([np.ravel(z), np.atleast_1d((level - self.level_mu)/self.level_sd)])

    def params(self, theta):
        """(z, level) for GPResolvedFit's "resolved" and "resolved_level"."""
        w = self.amp.shape[0]
        return theta[:-1].reshape(w, w), self.level_mu + self.level_sd*theta[-1]


def gp_prior(wid, sigma, length, order=2., mean=None, level_prior=(0., 1.)):
    length = np.broadcast_to(np.asarray(length, dtype=float), (2,))
    mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    sigma, order = np.asarray(sigma, dtype=float), np.asarray(order, dtype=float)
    return GPPrior(matern_amplitude((wid, wid), length, sigma, order), mean, mu, sd, sigma, length, order)


def gp_prior_from_fit(fit):
    """The GPPrior of a GPResolvedFit's current hyperparameters."""
    return gp_prior(fit.mean.shape[0], fit.sigma, fit.length, fit.order, fit.mean, fit.level_prior)


# ----------------------------------------------------------------------------- linear response
def _save_npz(path, **arrays):
    """Atomic np.savez (safe with concurrent array tasks and against a job killed mid-write)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        onp.savez(fh, **{k: onp.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


class LinearResponse(eqx.Module):
    """The exact quadratic negative log-likelihood of the disc d (flat, wid^2) at fixed optics, per exposure e:

        nll_e(d) = 0.5 beta_e (d^T G_e d - 2 b_e^T d + yy_e) + lognorm_e - 0.5 n_good_e log beta_e,

    G_e = R_e^T W_e R_e, b_e = R_e^T W_e (y_e - A_e), yy_e = (y_e - A_e)^T W_e (y_e - A_e), with R_e the disc response,
    A_e the image at d = 0 (star + bias), W_e = 1 / err^2 on the good pixels and beta_e = 1 / s_e^2 a noise rescaling
    (beta = 1: the errors as given). lognorm_e = sum_good log(err sqrt(2 pi)), so nll_e is ModelFit.loglike exactly."""
    G: jax.Array
    b: jax.Array
    yy: jax.Array
    n_good: jax.Array
    lognorm: jax.Array
    keys: tuple = eqx.field(static=True)

    @property
    def wid(self):
        return int(round(onp.sqrt(self.G.shape[-1])))

    def chi2(self, d):
        """Per-exposure chi^2 (unit beta) of the flat disc d."""
        return self.yy - 2*self.b @ d + np.einsum("eij,j->ei", self.G, d) @ d

    def nll(self, d, beta=None):
        """Per-exposure negative log-likelihood of the flat disc d."""
        beta = np.ones(len(self.keys)) if beta is None else np.asarray(beta, dtype=float)
        return 0.5*beta*self.chi2(d) + self.lognorm - 0.5*self.n_good*np.log(beta)

    def save(self, path):
        _save_npz(path, G=self.G, b=self.b, yy=self.yy, n_good=self.n_good, lognorm=self.lognorm,
                  keys=onp.array(self.keys))

    @classmethod
    def load(cls, path):
        with onp.load(path) as f:
            return cls(*(np.asarray(f[k]) for k in ("G", "b", "yy", "n_good", "lognorm")),
                       tuple(str(k) for k in f["keys"]))


def _with_disc(fit, d):
    return eqx.tree_at(lambda f: f.d_override, fit, d, is_leaf=lambda x: x is None)


@eqx.filter_jit
def _response_group(model, group, batch_size):
    """(G, b, yy, n_good, lognorm) of every member of a batching.Group, stacked."""
    wid = group.template.fit.mean.shape[0]
    n = wid*wid

    def member(j):
        mdl, e = group.member(model, j)
        good = ~e.bad & np.isfinite(e.err) & np.isfinite(e.data)
        err = np.where(good, e.err, 1.)
        sw = np.where(good, 1/err, 0.)

        def image(d):  # whitened model image, zero at masked pixels
            return np.where(good, _with_disc(e.fit, d.reshape(wid, wid))(mdl, e)*sw, 0.).ravel()

        a, lin = jax.linearize(image, np.zeros(n))
        # one column of the whitened response per disc pixel; the one-hot tangent is built from the traced index
        # (constant tangents can be constant-folded through the jvp, cf. precond._map_tangents)
        Jw = jax.lax.map(lambda i: lin(jax.nn.one_hot(i, n, dtype=a.dtype)), np.arange(n), batch_size=batch_size)
        r0 = np.where(good, e.data*sw, 0.).ravel() - a
        G = Jw @ Jw.T
        lognorm = np.sum(np.where(good, np.log(err) + 0.5*np.log(2*np.pi), 0.))
        return 0.5*(G + G.T), Jw @ r0, r0 @ r0, np.sum(good).astype(a.dtype), lognorm

    return jax.lax.map(member, np.arange(len(group.gidx)))


def _check_response_inputs(model, exposures):
    for e in exposures:
        assert isinstance(e.fit, GPResolvedFit), f"{e.key}: LinearResponse needs a GPResolvedFit, got {type(e.fit)}"
        # the response and its check both inject the disc through d_override, so a subclass must not bypass it
        assert type(e.fit).get_distribution is GPResolvedFit.get_distribution, \
            f"{type(e.fit).__name__} overrides get_distribution; LinearResponse would not see the disc"
    keys = {e.fit.get_key(e, "resolved") for e in exposures}
    assert len(keys) == 1, f"LinearResponse handles one disc; exposures read {keys}"
    assert "quadrature" not in model.params, "a fitted error scale (quadrature) makes the likelihood non-quadratic"


def build_linear_response(model, exposures, batch_size=16):
    """LinearResponse of `exposures` at the parameters in `model` (optics, star and bias fixed there). One jvp per disc
    pixel per exposure (batch_size at a time), exposures lax.mapped as in batching.py (one compile)."""
    _check_response_inputs(model, exposures)
    out = [None]*len(exposures)
    for g in make_batch(exposures, model).groups:
        res = _response_group(model, g, batch_size)
        for m, gi in enumerate(onp.asarray(g.gidx)):
            out[gi] = [r[m] for r in res]
    G, b, yy, n_good, lognorm = (np.stack([o[i] for o in out]) for i in range(5))
    return LinearResponse(G, b, yy, n_good, lognorm, tuple(e.key for e in exposures))


@eqx.filter_jit
def _full_nll_group(model, group, ds):
    wid = group.template.fit.mean.shape[0]

    def member(j):
        mdl, e = group.member(model, j)
        return jax.lax.map(lambda d: ModelFit.loglike(_with_disc(e.fit, d.reshape(wid, wid)), mdl, e), ds)

    return jax.lax.map(member, np.arange(len(group.gidx)))


def verify_linear_response(resp, model, exposures, ds):
    """Full-model ModelFit.loglike vs resp.nll for each flat disc in ds (k, wid^2). Returns (full, surrogate), each
    (n_exposures, k). They agree to rounding if the image is linear in the disc at fixed optics."""
    _check_response_inputs(model, exposures)
    ds = np.atleast_2d(np.asarray(ds, dtype=float))
    full = [None]*len(exposures)
    for g in make_batch(exposures, model).groups:
        res = _full_nll_group(model, g, ds)
        for m, gi in enumerate(onp.asarray(g.gidx)):
            full[gi] = res[m]
    return np.stack(full), np.stack([resp.nll(d) for d in ds], axis=1)


def simulate_response(resp, d_true, key, rtol=1e-12):
    """The LinearResponse of simulated data y_e = A_e + R_e d_true + err * N(0, 1), from resp alone (no optics).

    With the whitened response J_e = U S V^T (G_e = V S^2 V^T), the noise enters only through J_e^T eps = V S u,
    u ~ N(0, I) on the r non-zero singular values, and |eps|^2 = |u|^2 + chi^2_{n_good - r}."""
    d = np.ravel(np.asarray(d_true, dtype=float))
    bs, yys = [], []
    for e, k in enumerate(jr.split(key, len(resp.keys))):
        lam, V = np.linalg.eigh(resp.G[e])
        keep = lam > rtol*lam.max()
        k1, k2 = jr.split(k)
        u = np.where(keep, jr.normal(k1, lam.shape), 0.)
        Jte = V @ (np.sqrt(np.where(keep, lam, 0.))*u)
        rest = 2*jr.gamma(k2, (resp.n_good[e] - keep.sum())/2)
        Gd = resp.G[e] @ d
        bs.append(Gd + Jte)
        yys.append(d @ Gd + 2*d @ Jte + u @ u + rest)
    return eqx.tree_at(lambda r: (r.b, r.yy), resp, (np.stack(bs), np.stack(yys)))


# ----------------------------------------------------------------------------- MAP, evidence, noise
def _dct2(x):
    """Orthonormal 2D DCT-II of a (w, w, ...) array over its first two axes (Psi^T, the inverse of gp_field's map)."""
    return jsp.fft.dctn(x, type=2, norm="ortho", axes=(0, 1))


def _congruence_dct(H4):
    """Psi^T H Psi for H reshaped (w, w, w, w): DCT over the row-pixel axes, then over the column-pixel axes."""
    return jsp.fft.dctn(_dct2(H4), type=2, norm="ortho", axes=(2, 3))


def _weighted(resp, beta):
    return np.einsum("e,eij->ij", beta, resp.G), beta @ resp.b, beta @ resp.yy


def _whitened_metric(G, d, prior):
    """J^T G J for theta = (z, u), J = dd/dtheta = ln10 diag(d) [Psi diag(amp) | level_sd 1], Psi the idctn matrix.
    Psi^T H Psi is two 2D DCTs of H reshaped (w, w, w, w), not a dense matmul."""
    w = prior.amp.shape[0]
    n = w*w
    D = LN10*d
    H = G*D[:, None]*D[None, :]
    B = _congruence_dct(H.reshape(w, w, w, w)).reshape(n, n)
    a = prior.amp.ravel()
    r = H.sum(1)
    zu = a*_dct2(r.reshape(w, w)).ravel()*prior.level_sd
    top = np.concatenate([B*a[:, None]*a[None, :], zu[:, None]], axis=1)
    bottom = np.concatenate([zu, (r.sum()*prior.level_sd**2)[None]])[None, :]
    return np.concatenate([top, bottom], axis=0)


def _grad(G, b, d, prior, theta):
    """Gradient of 0.5 d^T G d - b^T d + 0.5 |theta|^2 in theta."""
    w = prior.amp.shape[0]
    Dg = LN10*d*(G @ d - b)
    gz = prior.amp.ravel()*_dct2(Dg.reshape(w, w)).ravel()
    return np.concatenate([gz, (prior.level_sd*Dg.sum())[None]]) + theta


def _log10_step(prior, delta):
    """Change of log10 I (flat) for a step delta in theta."""
    w = prior.amp.shape[0]
    return gp_field(delta[:-1].reshape(w, w), prior.amp).ravel() + prior.level_sd*delta[-1]


@eqx.filter_jit
def _map_core(resp, beta, prior, theta0, max_iter, tol, max_dex):
    G, b, yy = _weighted(resp, beta)
    eye = np.eye(theta0.size)

    def obj(theta):
        d = prior.distribution(theta).ravel()
        f = 0.5*(d @ (G @ d)) - b @ d + 0.5*yy + 0.5*theta @ theta
        return np.where(np.isfinite(f), f, np.inf)

    # a start whose image overflows (e.g. a warm start far from these hyperparameters) falls back to the prior mean
    f0 = obj(theta0)
    theta0 = np.where(np.isfinite(f0), theta0, np.zeros_like(theta0))

    def cond(c):
        theta, lam, f, k, dec = c
        return (k < max_iter) & ~(dec < tol)

    def step(c):
        theta, lam, f, k, _ = c
        d = prior.distribution(theta).ravel()
        g = _grad(G, b, d, prior, theta)
        M = _whitened_metric(G, d, prior) + eye
        delta = -jsp.linalg.cho_solve(jsp.linalg.cho_factor(M + lam*eye, lower=True), g)
        # the Newton decrement g^T M^-1 g ~ 2 (f - f_min) at theta is at most (1 + lam) g^T (M + lam I)^-1 g, as M >= I:
        # converged when that bound is below tol
        dec = -(1 + lam)*(g @ delta)
        # trust region in the image: no pixel's log10 I moves by more than max_dex per step (no overflow, no GN
        # overshoot on a cold start)
        delta = delta*np.minimum(1., max_dex/np.max(np.abs(_log10_step(prior, delta))))
        new = theta + delta
        fn = obj(new)
        ok = fn < f
        theta, f = np.where(ok, new, theta), np.where(ok, fn, f)
        lam = np.where(ok, np.maximum(lam*0.3, 1e-12), lam*10.)
        return theta, lam, f, k + 1, dec

    theta, lam, f, k, dec = jax.lax.while_loop(
        cond, step, (theta0, np.asarray(1., theta0.dtype), obj(theta0), 0, np.asarray(np.inf, theta0.dtype)))
    return theta, dict(objective=f, steps=k, decrement=dec, damping=lam)


def _beta(resp, beta):
    return np.ones(len(resp.keys)) if beta is None else np.asarray(beta, dtype=float)


def gp_map(resp, prior, theta0=None, beta=None, n_iter=200, tol=1e-6, max_dex=1.):
    """MAP theta = (z, u) at fixed hyperparameters, by Levenberg-Marquardt with the Gauss-Newton metric I + J^T G J
    (exact Cholesky solves) and a trust region of max_dex per pixel per step, until the Newton decrement is below tol
    or n_iter steps. Returns (theta, info): objective (without the lognorm constants), steps, decrement, damping."""
    theta0 = np.zeros(prior.amp.size + 1) if theta0 is None else np.asarray(theta0, dtype=float)
    return _map_core(resp, _beta(resp, beta), prior, theta0, n_iter, tol, max_dex)


@eqx.filter_jit
def _evidence_core(resp, beta, prior, theta):
    G, b, _ = _weighted(resp, beta)
    d = prior.distribution(theta).ravel()
    A = _whitened_metric(G, d, prior)
    mu, V = np.linalg.eigh(A)
    mu = np.clip(mu, 0., None)
    chi2 = resp.chi2(d)
    nll = 0.5*beta*chi2 + resp.lognorm - 0.5*resp.n_good*np.log(beta)
    g = _grad(G, b, d, prior, theta)
    # tr(M^-1 A_e), M = I + A: each exposure's share of the well-measured directions, divided by beta_e
    tr = jax.lax.map(lambda Ge: np.sum(V*(_whitened_metric(Ge, d, prior) @ V), 0) @ (1/(1 + mu)), resp.G)
    return dict(logZ=-nll.sum() - 0.5*theta @ theta - 0.5*np.sum(np.log1p(mu)), nll=nll, chi2=chi2,
                prior=0.5*theta @ theta, occam=0.5*np.sum(np.log1p(mu)), gamma=np.sum(mu/(1 + mu)),
                gamma_e=beta*tr, tr_e=tr, newton_decrement=np.sum((V.T @ g)**2/(1 + mu)), grad_norm=np.sqrt(g @ g))


def gp_log_evidence(resp, prior, theta, beta=None):
    """Laplace log evidence at the MAP theta (exact Gaussian integral around it, Gauss-Newton curvature):

        log Z = -sum_e nll_e(d) - 0.5 |theta|^2 - 0.5 sum_i log(1 + mu_i),

    mu_i the eigenvalues of J^T G J in the whitened coordinates (the prior's normalisation cancels the Laplace
    (2 pi)^(n/2)). Also returns the terms, chi2 per exposure, gamma = sum mu / (1 + mu) (the number of well-measured
    directions), gamma_e (per exposure) and the Newton decrement g^T M^-1 g (~0 at a converged MAP)."""
    return _evidence_core(resp, _beta(resp, beta), prior, np.asarray(theta, dtype=float))


def gp_error_scale(resp, prior, theta0=None, beta0=None, n_outer=20, tol=1e-4, n_iter=200, verbose=False):
    """Per-exposure noise scales s_e (err -> s_e err) maximising the Laplace evidence (MacKay 1992): at the stationary
    point beta_e = n_good_e / (chi2_e + tr(M^-1 A_e)), i.e. beta_e chi2_e = n_good_e - gamma_e, iterated with the MAP.
    s >> 1 means the model fails to fit (fix the model rather than rescale). Returns (s, theta, evidence dict)."""
    beta = _beta(resp, beta0)
    theta = theta0
    change = np.inf
    for k in range(n_outer):
        theta, _ = gp_map(resp, prior, theta, beta, n_iter)
        ev = gp_log_evidence(resp, prior, theta, beta)
        new = resp.n_good/(ev["chi2"] + ev["tr_e"])
        change = float(np.max(np.abs(np.log(new/beta))))
        beta = new
        if verbose:
            print(f"error_scale iteration {k}: s = {onp.round(onp.asarray(1/np.sqrt(beta)), 4)}, change {change:.2e}")
        if change < tol:
            break
    else:
        warnings.warn(f"gp_error_scale: not converged after {n_outer} iterations (last change {change:.2e})")
    theta, _ = gp_map(resp, prior, theta, beta, n_iter)
    return 1/np.sqrt(beta), theta, gp_log_evidence(resp, prior, theta, beta)


# ----------------------------------------------------------------------------- hyperparameter grid
def _point_file(outdir, tag, sigma, length):
    return os.path.join(outdir, f"{tag}_s{sigma:.6g}_l{length:.6g}.npz")


def _point_meta(resp, prior, beta, n_iter):
    """Digest of everything a grid point depends on besides (sigma, length), stored with it and checked on resume."""
    import hashlib
    h = hashlib.sha256()
    for a in (resp.b, resp.yy, resp.n_good, resp.G.diagonal(axis1=1, axis2=2), beta, prior.order, prior.mean,
              prior.level_mu, prior.level_sd, n_iter):
        h.update(onp.ascontiguousarray(onp.asarray(a, dtype=float)).tobytes())
    return h.hexdigest()[:16]


def _ratio(vals, name):
    """The common ratio of a log-spaced axis (None for a single value)."""
    if len(vals) < 2:
        return None
    r = onp.asarray(vals[1:])/onp.asarray(vals[:-1])
    assert onp.allclose(r, r[0], rtol=1e-6) and r[0] > 1, f"{name} must be log-spaced with one ratio > 1, got {vals}"
    return float(r[0])


def gp_evidence_grid(resp, sigmas, lengths, order=2., mean=None, level_prior=(0., 1.), beta=None, outdir=None,
                     tag="grid", max_growth=3, bounds=((0.03, 8.), (0.1, 30.)), n_iter=200, theta0=None,
                     converged_tol=1e-3, edge_dlogz=5., verbose=True):
    """Laplace evidence over a log-spaced (sigma, length) grid (isotropic length), each point a warm-started MAP.

    Each axis must have one common ratio. While an edge row / column holds a point within edge_dlogz of the best log Z
    (posterior mass at the edge, not only the argmax), the grid is extended by one step past that edge, at most
    max_growth times and within `bounds` ((sigma_lo, hi), (length_lo, hi)). Warm starts keep the neighbour's log10
    image (z rescaled by its amplitude ratio). A non-finite log Z counts as -inf (and unconverged). With `outdir`, each
    point is saved as {outdir}/{tag}_s{sigma}_l{length}.npz with a digest of the response, beta and prior settings, and
    reloaded on a rerun only if that digest matches (else recomputed). Returns a dict with sigmas, lengths, logZ (n_s,
    n_l), gamma, chi2 (summed), newton_decrement, converged, steps, the MAP thetas (n_s, n_l, n + 1), and the inputs."""
    sigmas, lengths = sorted(float(s) for s in sigmas), sorted(float(l) for l in lengths)
    r_s, r_l = _ratio(sigmas, "sigmas"), _ratio(lengths, "lengths")
    # starting values outside the bounds are dropped (e.g. a local grid about an optimum near a bound)
    for name, vals, (lo, hi) in (("sigmas", sigmas, bounds[0]), ("lengths", lengths, bounds[1])):
        keep = [v for v in vals if lo*(1 - 1e-9) <= v <= hi*(1 + 1e-9)]
        if len(keep) < len(vals):
            warnings.warn(f"gp_evidence_grid {tag}: {name} outside {(lo, hi)} dropped: {sorted(set(vals) - set(keep))}")
        assert keep, f"gp_evidence_grid {tag}: no {name} within {(lo, hi)}"
        vals[:] = keep
    beta = _beta(resp, beta)
    w = resp.wid
    results = {}
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    def prior_at(s, l):
        return gp_prior(w, s, l, order, mean, level_prior)

    def run(points):
        for s, l in points:
            if (s, l) in results:
                continue
            prior = prior_at(s, l)
            meta = _point_meta(resp, prior, beta, n_iter)
            path = None if outdir is None else _point_file(outdir, tag, s, l)
            if path is not None and os.path.exists(path):
                with onp.load(path) as f:
                    if "meta" in f.files and str(f["meta"]) == meta:
                        results[(s, l)] = {k: f[k] for k in f.files}
                        continue
                if verbose:
                    print(f"[gp grid {tag}] {path} was made with other inputs; recomputing", flush=True)
            if results:   # warm start from the nearest evaluated point, keeping its log10 image
                near = min(results, key=lambda p: (onp.log(p[0]/s))**2 + (onp.log(p[1]/l))**2)
                th = onp.asarray(results[near]["theta"])
                a_old, a_new = onp.asarray(prior_at(*near).amp).ravel(), onp.asarray(prior.amp).ravel()
                z = onp.where(a_new > 0, th[:-1]*a_old/onp.where(a_new > 0, a_new, 1.), 0.)
                start = np.asarray(onp.concatenate([z, th[-1:]]))
            else:
                start = theta0
            theta, info = gp_map(resp, prior, start, beta, n_iter)
            ev = jax.tree.map(onp.asarray, gp_log_evidence(resp, prior, theta, beta))
            r = dict(sigma=s, length=l, order=float(order), theta=onp.asarray(theta), meta=meta,
                     steps=int(info["steps"]),
                     **{k: ev[k] for k in ("logZ", "nll", "chi2", "prior", "occam", "gamma", "gamma_e",
                                          "newton_decrement", "grad_norm")})
            finite = bool(onp.isfinite(r["logZ"]))
            r["logZ"] = r["logZ"] if finite else onp.asarray(-onp.inf)
            r["converged"] = finite and bool(r["newton_decrement"] < converged_tol)
            results[(s, l)] = r
            if path is not None:
                _save_npz(path, **r)
            if verbose:
                print(f"[gp grid {tag}] sigma {s:.4g} dex, length {l:.4g} px: logZ {float(r['logZ']):.3f}, "
                      f"gamma {float(r['gamma']):.1f}, decrement {float(r['newton_decrement']):.1e}, "
                      f"{r['steps']} steps", flush=True)

    def table():
        return onp.array([[float(results[(s, l)]["logZ"]) for l in lengths] for s in sigmas])

    for growth in range(max_growth + 1):
        # snake order, so each point starts from a neighbour
        run([(s, l) for i, s in enumerate(sigmas) for l in (lengths if i % 2 == 0 else lengths[::-1])])
        logZ = table()
        if growth == max_growth:
            break
        near_max = logZ > onp.max(logZ) - edge_dlogz
        grew = False
        for vals, ratio, (lo, hi), lo_edge, hi_edge in (
                (sigmas, r_s, bounds[0], near_max[0].any(), near_max[-1].any()),
                (lengths, r_l, bounds[1], near_max[:, 0].any(), near_max[:, -1].any())):
            if ratio is None:
                continue
            if lo_edge and vals[0]/ratio >= lo:
                vals.insert(0, vals[0]/ratio)
                grew = True
            if hi_edge and vals[-1]*ratio <= hi:
                vals.append(vals[-1]*ratio)
                grew = True
        if not grew:
            break
    i, j = onp.unravel_index(onp.argmax(logZ), logZ.shape)
    get = lambda k: onp.array([[onp.asarray(results[(s, l)][k]) for l in lengths] for s in sigmas])
    grid = dict(sigmas=onp.array(sigmas), lengths=onp.array(lengths), logZ=logZ, gamma=get("gamma"),
                chi2=get("chi2").sum(-1), newton_decrement=get("newton_decrement"), converged=get("converged"),
                steps=get("steps"), thetas=get("theta"), order=float(order), best=(int(i), int(j)), tag=tag)
    ew = gp_hyper_posterior(grid)["edge_weight"]
    if ew > 1e-2:
        warnings.warn(f"gp_evidence_grid {tag}: {ew:.2f} of the posterior weight is on the edge of the final grid "
                      "(growth limit or bounds reached): its intervals are truncated")
    return grid


def _quantiles(x, w, qs=(0.16, 0.5, 0.84)):
    """Quantiles of a distribution on log-spaced cells x (weights w), interpolating the CDF in log x at the cell
    centres; a cell is assumed to span half a step either side."""
    lx = onp.log(x)
    edges = onp.concatenate([[lx[0] - 0.5*(lx[1] - lx[0])] if len(x) > 1 else [lx[0] - 0.5],
                             0.5*(lx[1:] + lx[:-1]),
                             [lx[-1] + 0.5*(lx[-1] - lx[-2])] if len(x) > 1 else [lx[0] + 0.5]])
    cdf = onp.concatenate([[0.], onp.cumsum(w)/onp.sum(w)])
    return onp.exp(onp.interp(qs, cdf, edges))


def gp_hyper_posterior(grid, box=None):
    """p(sigma, length | data) on the grid under log-uniform hyperpriors over its extent (each axis log-spaced, so a
    uniform weight per cell): weights = softmax(logZ). Returns the weights, the marginal quantiles (16, 50, 84 %) of
    sigma and length, edge_weight (the posterior weight on the grid's border: large means truncated intervals), and
    log_evidence = log of the mean of Z over the cells. log_evidence depends on the grid's extent, so to compare model
    variants (order, template) whose grids grew differently, pass box = (sigmas, lengths), a common set of cells, and
    compare log_evidence_box: the evidence under the same log-uniform hyperprior on that box for every variant."""
    logZ = grid["logZ"]
    lse = onp.logaddexp.reduce(logZ.ravel())
    w = onp.exp(logZ - lse)
    edge = onp.ones_like(w, dtype=bool)
    if min(w.shape) > 2:
        edge[1:-1, 1:-1] = False
    out = dict(weights=w, sigma_q=_quantiles(grid["sigmas"], w.sum(1)), length_q=_quantiles(grid["lengths"], w.sum(0)),
               log_evidence=float(lse - onp.log(logZ.size)), edge_weight=float(w[edge].sum()))
    if box is not None:
        si = [int(onp.argmin(onp.abs(onp.log(grid["sigmas"]/s)))) for s in box[0]]
        li = [int(onp.argmin(onp.abs(onp.log(grid["lengths"]/l)))) for l in box[1]]
        assert onp.allclose(grid["sigmas"][si], box[0]) and onp.allclose(grid["lengths"][li], box[1]), \
            "box cells must be grid cells"
        sub = logZ[onp.ix_(si, li)]
        out["log_evidence_box"] = float(onp.logaddexp.reduce(sub.ravel()) - onp.log(sub.size))
    return out


# ----------------------------------------------------------------------------- Laplace posterior, diagnostics
@eqx.filter_jit
def _laplace_core(resp, beta, prior, theta):
    G, _, _ = _weighted(resp, beta)
    mu, V = np.linalg.eigh(_whitened_metric(G, prior.distribution(theta).ravel(), prior))
    return np.clip(mu, 0., None), V


def gp_laplace(resp, prior, theta, beta=None):
    """(mu, V): eigenvalues and eigenvectors of J^T G J at theta; the Laplace posterior is
    N(theta, V diag(1 / (1 + mu)) V^T)."""
    return _laplace_core(resp, _beta(resp, beta), prior, np.asarray(theta, dtype=float))


def gp_laplace_samples(theta, laplace, key, n=200):
    """n exact draws (n, len(theta)) from the Laplace posterior N(theta, (I + J^T G J)^-1)."""
    mu, V = laplace
    return theta + (jr.normal(key, (n, theta.size))/np.sqrt(1 + mu)) @ V.T


def gp_log10_std(prior, laplace):
    """Exact per-pixel posterior std of log10 I under the Laplace approximation (log10 I is linear in theta)."""
    mu, V = laplace
    w = prior.amp.shape[0]
    W = V/np.sqrt(1 + mu)
    LW = jsp.fft.idctn(prior.amp[:, :, None]*W[:-1].reshape(w, w, -1), type=2, norm="ortho", axes=(0, 1))
    LW = LW.reshape(w*w, -1) + prior.level_sd*W[-1][None, :]
    return np.sqrt(np.sum(LW**2, axis=1)).reshape(w, w)


def gp_spectral_diagnostic(prior, theta, laplace):
    """Posterior power of the whitened DCT modes, binned in radial wavenumber, against the prior's (1 per mode):
    <z_k^2> = z_map_k^2 + var_k, which averages to 1 when the prior's spectral shape is right. A systematic excess
    (deficit) over a range of wavenumbers means the Matern shape has too little (too much) power there. Also returns
    the mean posterior variance per bin (1: unconstrained by the data, 0: fully measured). freq is in cycles / px."""
    mu, V = laplace
    w = prior.amp.shape[0]
    var = np.sum(V[:-1]**2/(1 + mu), axis=1)
    power = onp.asarray(theta[:-1]**2 + var)
    var = onp.asarray(var)
    j, k = onp.meshgrid(onp.arange(w), onp.arange(w), indexing="ij")
    kappa = onp.hypot(j, k).ravel()
    active = onp.asarray(prior.amp).ravel() > 0
    bins = onp.floor(kappa).astype(int)
    out = dict(freq=[], power=[], power_err=[], var=[], count=[])
    for bi in onp.unique(bins[active]):
        sel = active & (bins == bi)
        out["freq"].append((bi + 0.5)/(2*w))
        out["power"].append(power[sel].mean())
        out["power_err"].append(onp.sqrt(2/sel.sum()))   # sampling scatter of a mean of chi^2_1 / 1 under the prior
        out["var"].append(var[sel].mean())
        out["count"].append(sel.sum())
    return {k: onp.array(v) for k, v in out.items()}
