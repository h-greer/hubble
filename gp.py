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

Polar (multipole) field, after virgil's MultipoleField (fields.py; "provisional" there): a log10 field on a polar
(r, theta) grid in the deprojected disc plane, interpolated bilinearly onto the disc pixels,

    log10 I = level + mean + P[ sigma * taper( L_r Z F_theta ) ],

Z (n_r, n_theta) whitened, L_r the Cholesky factor of a radial Matern (nu_r in {1/2, 3/2, 5/2, inf}, length_r in
disc-plane grid px), F_theta a circulant filter with the periodic Matern spectrum log S(m) = (-nu - 1/2) log1p((m
length_theta)^2 / 2 nu) normalised to unit pointwise variance (length_theta in radians; at 2 rad ~30% of the variance
is still in |m| >= 1, from ~4 rad the field is nearly axisymmetric), taper = tanh(r / spacing)^|m| on harmonic m
(single-valued at r = 0), P the bilinear map from the polar grid onto the disc pixels (PolarGrid), set by the disc's
geometry: cos_inc (cos of the inclination), pa (deg, the projected major axis from +y = rows towards +x = columns, in
the disc grid's own frame; a flat disc's prior is unchanged by pa + 180) and the centre offset (dx, dy) in grid px.
The geometry is a hyperparameter like sigma and the lengths: it sets the prior covariance, so it is chosen by the
evidence on the same linear response (gp_evidence_grid_nd with linear / periodic axes), never by a MAP jointly with z
or by a fit to a reference image. The radial grid is fixed in the disc plane (r_max, spacing, n_r independent of the
geometry), so priors at different geometries share their latents and length_r its meaning; pixels beyond r_max (only
at inclinations above PolarGrid's cover_inc) take the outermost ring's value.
Unlike the DCT field no mode is removed, so `level` and the m = 0 part of the field share the overall level (the level
prior separates them, so keep level_sd loose unless level_mu is trustworthy: a tight, mis-centred level prior is absorbed
by the field and the evidence then prefers a larger sigma). A flat, thin disc: no flaring, phase function or brightness
Jacobian; near edge on (inc >~ 80 deg) a thin disc's deprojection breaks down, so bound cos_inc away from 0. sigma is the rms at the polar nodes away from the centre (the taper lowers the m >= 1 variance within ~3
spacings of r = 0); bilinear interpolation lowers it between nodes, so the pixel rms is ~0.75-0.9 sigma for length_r of
1-4 spacings and ripples on the node lattice (as virgil's). length_r much below the node spacing is not identifiable
(the pixels cannot see it): bound it below at ~the spacing.

Both fields are linear in z, log10 I = level + mean + B z, which is all the machinery below uses (GPPrior / PolarPrior:
field, field_T, congruence = B^T H B, field_matrix); the DCT applies B by fast transforms, the polar field as a dense
matrix built once per hyperparameter point (geometry included).

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
  gp_evidence_grid_nd grid over any hyperparameters, each axis log-spaced, linear or periodic (gp_evidence_grid: the DCT's
                      (sigma, length)), warm-started, grown by a step while posterior mass sits on an edge, resumable;
  local_axes          a local grid about a point, for refining / the joint grid around an optimum;
  gp_hyper_posterior  the grid as quadrature for p(hyperparameters | data) under log-uniform hyperpriors (INLA-style);
  gp_laplace / gp_laplace_samples / gp_log10_std   the Laplace posterior N(theta_map, (I + J^T G J)^-1);
  gp_spectral_diagnostic   posterior power per DCT wavenumber vs the prior (1 per whitened mode);
  gp_polar_diagnostic      the same per azimuthal order m and per radial latent row, for the polar field;
  simulate_response   injection: the response of simulated data, without any optics.
  multi_gp_prior / MultiGPResolvedFit / two_kernel_prior_factory / capped_posterior   multi-kernel DCT prior (a sum of K
                      Matern kernels on the one DCT basis) and its fit, kernel factory and length-capped hyper-posterior;
  bpl_gp_prior / BPLGPResolvedFit / bpl_prior_factory   broken-power-law spectrum prior (one spectrum, four hyperparameters;
                      gamma1 = 0 is exactly the Matern prior of order gamma2 / 2);
  SumPrior / PolarDCTResolvedFit   polar + DCT prior (polar field plus a DCT residual field) and its fit.
"""
import itertools
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
HYPER_NAMES = ("sigma", "length", "order", "length_r", "length_theta", "cos_inc", "pa", "dx", "dy", "mean",
               "level_prior", "sigmas", "lengths", "orders", "sigma_p", "sigma_d", "length_d", "order_d",
               "length_b", "gamma1", "gamma2")   # any GP fit's
# (sigmas, lengths, orders: MultiGPResolvedFit; sigma_p, sigma_d, length_d, order_d: SumPrior / PolarDCTResolvedFit;
# length_b, gamma1, gamma2: BPLGPResolvedFit)


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


# ----------------------------------------------------------------------------- polar (multipole) field
_NUS = (0.5, 1.5, 2.5, float("inf"))


def _check_nu(nu):
    nu = float(nu)
    assert nu in _NUS, f"nu must be one of {_NUS} (closed-form Matern), got {nu}"
    return nu


def matern_correlation(d, nu):
    """Matern correlation at separation d (in units of the length), nu in {1/2, 3/2, 5/2, inf} (virgil models._matern)."""
    if nu == 0.5:
        return np.exp(-d)
    if nu == 1.5:
        s = np.sqrt(3.)*d
        return (1 + s)*np.exp(-s)
    if nu == 2.5:
        s = np.sqrt(5.)*d
        return (1 + s + s**2/3)*np.exp(-s)
    return np.exp(-0.5*d**2)


def radial_factor(nodes, length, nu):
    """Cholesky factor of the radial Matern correlation on `nodes` (virgil fields._radial_factor); the jitter scales
    with the dtype's eps so that long lengths stay positive definite."""
    corr = matern_correlation(np.abs(nodes[:, None] - nodes[None, :])/length, nu)
    jitter = max(1e-6, 10*nodes.size*float(np.finfo(corr.dtype).eps))
    return np.linalg.cholesky(corr + jitter*np.eye(nodes.size))


def log_azimuthal_spectrum(m, length, nu):
    """log S(m) of the periodic Matern over azimuthal order m (virgil models._log_azimuthal_spectrum); in log space so
    that high orders underflow without NaN gradients."""
    x = (m*length)**2
    return -0.5*x if nu == float("inf") else (-nu - 0.5)*np.log1p(x/(2*nu))


def _orders(n):
    return np.minimum(np.arange(n), n - np.arange(n))


def azimuthal_filter(latent, length, nu):
    """Unit-variance periodic GP along the last axis from white `latent`: a circulant square root by FFT, normalised so
    mean_m S(m) = 1 (virgil fields._azimuthal_filter)."""
    n = latent.shape[-1]
    lp = log_azimuthal_spectrum(_orders(n), length, nu)
    amp = np.exp(0.5*(lp - jax.scipy.special.logsumexp(lp) + np.log(n)))
    return np.real(np.fft.ifft(amp*np.fft.fft(latent, axis=-1), axis=-1))


def polar_values(latent, nodes, spacing, sigma, length_r, length_theta, nu_r, nu_theta):
    """The field on the polar grid, (n_r, n_theta): sigma * taper(L_r latent F_theta), taper = tanh(r / spacing)^|m| on
    harmonic m (nodes[0] = 0, so only m = 0 survives at the centre; virgil fields.MultipoleField.polar_values)."""
    field = sigma*(radial_factor(nodes, length_r, nu_r) @ azimuthal_filter(latent, length_theta, nu_theta))
    taper = np.tanh(nodes/spacing)[:, None]**_orders(latent.shape[-1])[None, :]
    return np.real(np.fft.ifft(taper*np.fft.fft(field, axis=1), axis=1))


class PolarGrid(eqx.Module):
    """A polar grid in the disc plane, r = 0 .. r_max at n_r uniform nodes and n_theta azimuths, for a wid x wid disc,
    and the bilinear map from it onto the disc pixels for a given geometry (virgil fields.polar_interpolate /
    undo_elliptical_transf_coord). The radial grid is static and the same for every geometry: r_max defaults to the
    box's half-diagonal deprojected at cover_inc (deg), so every pixel is inside the grid up to that inclination and
    the outermost ring's value extends beyond it. The geometry (cos_inc, pa, dx, dy) is an argument (traced: changing it
    never recompiles). Frame: pixel (row i, column j) is at x = j - (wid - 1) / 2 - dx, y = i - (wid - 1) / 2 - dy (the
    star at the centre for dx = dy = 0)."""
    nodes: jax.Array
    shape: tuple = eqx.field(static=True)
    wid: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    nu_r: float = eqx.field(static=True)
    nu_theta: float = eqx.field(static=True)

    @classmethod
    def build(cls, wid, n_r=44, n_theta=128, r_max=None, cover_inc=55., nu_r=1.5, nu_theta=None):
        assert n_r >= 2 and n_theta >= 2 and n_theta % 2 == 0, "n_theta must be even (pa + 180 symmetry)"
        if r_max is None:
            r_max = (wid - 1)/2*onp.sqrt(2)/onp.cos(onp.deg2rad(cover_inc))
        spacing = float(r_max)/(n_r - 1)
        nu_r = _check_nu(nu_r)
        nu_theta = nu_r if nu_theta is None else _check_nu(nu_theta)
        return cls(np.asarray(spacing*onp.arange(n_r)), (int(n_r), int(n_theta)), int(wid), spacing, nu_r, nu_theta)

    @property
    def n_latent(self):
        return self.shape[0]*self.shape[1]

    @property
    def r_max(self):
        return self.spacing*(self.shape[0] - 1)

    def coordinates(self, geometry):
        """Disc-plane (r in grid px, theta in rad from the major axis) of every disc pixel, (wid, wid) each.
        geometry = (cos_inc, pa_deg, dx, dy)."""
        cos_inc, pa, dx, dy = geometry
        c = (self.wid - 1)/2
        y, x = np.mgrid[:self.wid, :self.wid] - c
        x, y = x - dx, y - dy
        p = np.deg2rad(pa)
        xe = (x*np.cos(p) - y*np.sin(p))/np.maximum(cos_inc, 1e-3)
        ye = x*np.sin(p) + y*np.cos(p)
        return np.sqrt(xe**2 + ye**2), np.arctan2(xe, ye)

    def interpolation(self, geometry):
        """(idx, weights), each (wid^2, 4): the flat polar indices of each pixel's bilinear neighbours and their weights;
        pixels beyond r_max get the outermost ring."""
        n_r, n_t = self.shape
        r, theta = self.coordinates(geometry)
        f = np.minimum(r.ravel()/self.spacing, n_r - 1)
        i0 = np.minimum(np.floor(f), n_r - 2)
        wr = np.clip(f - i0, 0., 1.)
        t = np.mod(theta.ravel(), 2*np.pi)*n_t/(2*np.pi)
        j0 = np.floor(t)
        wt = t - j0
        i0, j0 = i0.astype(int), j0.astype(int) % n_t
        j1 = (j0 + 1) % n_t
        idx = np.stack([i0*n_t + j0, i0*n_t + j1, (i0 + 1)*n_t + j0, (i0 + 1)*n_t + j1], -1)
        return idx, np.stack([(1 - wr)*(1 - wt), (1 - wr)*wt, wr*(1 - wt), wr*wt], -1)

    def fraction_outside(self, geometry):
        """The fraction of disc pixels beyond r_max (given the outermost ring's value) at this geometry."""
        return np.mean(self.coordinates(geometry)[0] > self.r_max)

    def values(self, z, sigma, length_r, length_theta):
        return polar_values(z.reshape(self.shape), self.nodes, self.spacing, sigma, length_r, length_theta,
                            self.nu_r, self.nu_theta)

    def field(self, z, sigma, length_r, length_theta, geometry, interp=None):
        """log10 field on the disc pixels (wid, wid) from latents z (n_r, n_theta); interp: precomputed
        self.interpolation(geometry)."""
        idx, w = self.interpolation(geometry) if interp is None else interp
        v = self.values(z, sigma, length_r, length_theta).ravel()
        return np.sum(w*v[idx], axis=1).reshape(self.wid, self.wid)

    def digest(self):
        import hashlib
        h = hashlib.sha256(repr((self.shape, self.wid, self.spacing, self.nu_r, self.nu_theta)).encode())
        h.update(onp.ascontiguousarray(onp.asarray(self.nodes)).tobytes())
        return h.hexdigest()[:16]


def _geometry(cos_inc=1., pa=0., dx=0., dy=0.):
    return np.stack([np.asarray(x, dtype=float) for x in (cos_inc, pa, dx, dy)])


class _GPFitBase(PointResolvedFit):
    """Shared by the GP disc fits (always on an InterpolatedResolvedSource): log10 I = level + mean + log10_field(z).

    Parameters: "resolved" (the whitened latents z, start at zeros) and "resolved_level" (scalar, dex), both keyed by
    target. Hyperparameters (type(fit).HYPER) are fit leaves, not parameters: change them with set_hyper (no
    recompilation). The loss gains the negative log prior 0.5 |z|^2 + 0.5 ((level - mu) / sd)^2, divided by n_share,
    the number of exposures in the loss sharing one target-keyed disc (with_n_share), so it is counted once in total.
    d_override (internal): a disc used instead of the parameters (LinearResponse, verify_linear_response).
    Subclasses define log10_field(z) and surrogate_prior(), and must not override get_distribution."""
    PARAM_KEYS = PointResolvedFit.PARAM_KEYS | {"resolved_level": "target"}
    HYPER = ("mean", "level_prior")
    mean: jax.Array
    level_prior: jax.Array
    n_share: int
    d_override: jax.Array | None

    def _init_gp(self, wid, mean, level_prior, n_share):
        self.mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
        assert self.mean.shape == (wid, wid), f"mean template shape {self.mean.shape} != ({wid}, {wid})"
        self.level_prior = np.asarray(level_prior, dtype=float)
        self.n_share = int(n_share)
        self.regulariser = None  # the GP prior replaces the L1/TSV weights; precond.py handles it through n_share
        self.d_override = None

    def log10_field(self, z):
        raise NotImplementedError

    def surrogate_prior(self):
        raise NotImplementedError

    def get_distribution(self, model, exposure):
        if self.d_override is not None:
            return self.d_override
        z = model.get(self.map_param(exposure, "resolved"))
        level = model.get(self.map_param(exposure, "resolved_level"))
        return 10**(level + self.mean + self.log10_field(z))

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


class GPResolvedFit(_GPFitBase):
    """The DCT Matern prior: "resolved" holds the wid x wid whitened DCT coefficients. Hyperparameters: sigma (dex),
    length (grid px, (row, col)), order, mean (log10 template, wid x wid), level_prior ((mean, std) of resolved_level,
    the spatial mean of log10 I - mean)."""
    HYPER = ("sigma", "length", "order", "mean", "level_prior")
    sigma: jax.Array
    length: jax.Array
    order: jax.Array

    def __init__(self, spectrum_basis, filter, wid, sigma=1., length=1.6, order=2., mean=None, level_prior=(0., 1.),
                 n_share=1, resolved_kwargs=None):
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.sigma = np.asarray(sigma, dtype=float)
        self.length = np.broadcast_to(np.asarray(length, dtype=float), (2,))
        self.order = np.asarray(order, dtype=float)
        self._init_gp(wid, mean, level_prior, n_share)

    def amplitude(self):
        return matern_amplitude(self.mean.shape, self.length, self.sigma, self.order)

    def log10_field(self, z):
        return gp_field(z, self.amplitude())

    def surrogate_prior(self):
        return gp_prior(self.mean.shape[0], self.sigma, self.length, self.order, self.mean, self.level_prior)


class PolarGPResolvedFit(_GPFitBase):
    """The polar (multipole) prior on a PolarGrid: "resolved" holds the (n_r, n_theta) whitened latents.
    Hyperparameters: sigma (dex), length_r (disc-plane grid px), length_theta (rad), the geometry cos_inc, pa (deg),
    dx, dy (grid px), mean and level_prior (as GPResolvedFit's, but the level is no longer exactly the spatial mean: the
    polar field keeps its constant part). Change any of them with set_hyper (no recompilation)."""
    HYPER = ("sigma", "length_r", "length_theta", "cos_inc", "pa", "dx", "dy", "mean", "level_prior")
    grid: PolarGrid
    sigma: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    cos_inc: jax.Array
    pa: jax.Array
    dx: jax.Array
    dy: jax.Array

    def __init__(self, spectrum_basis, filter, wid, grid, sigma=0.5, length_r=4., length_theta=0.5, cos_inc=1.,
                 pa=0., dx=0., dy=0., mean=None, level_prior=(0., 1.), n_share=1, resolved_kwargs=None):
        assert grid.wid == wid, f"PolarGrid is for a {grid.wid} px disc, not {wid}"
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.grid = grid
        self.sigma, self.length_r, self.length_theta, self.cos_inc, self.pa, self.dx, self.dy = (
            np.asarray(x, dtype=float) for x in (sigma, length_r, length_theta, cos_inc, pa, dx, dy))
        self._init_gp(wid, mean, level_prior, n_share)

    def geometry(self):
        return _geometry(self.cos_inc, self.pa, self.dx, self.dy)

    def log10_field(self, z):
        return self.grid.field(z, self.sigma, self.length_r, self.length_theta, self.geometry())

    def surrogate_prior(self):
        return polar_prior(self.grid, self.sigma, self.length_r, self.length_theta, self.cos_inc, self.pa, self.dx,
                           self.dy, self.mean, self.level_prior)


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
    """`exposures` with GP hyperparameters replaced (type(fit).HYPER: GPResolvedFit sigma, length, order, mean,
    level_prior; PolarGPResolvedFit sigma, length_r, length_theta, cos_inc, pa, dx, dy, mean, level_prior; None leaves
    one alone)."""
    out = []
    for e in exposures:
        fit = e.fit
        unknown = set(hyper) - set(type(fit).HYPER)
        assert not unknown, f"{type(fit).__name__} has no hyperparameters {unknown}; it has {type(fit).HYPER}"
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
    return 10**(p["resolved_level"][key] + fit.mean + fit.log10_field(p["resolved"][key]))


# ----------------------------------------------------------------------------- surrogate priors (theta = (z, u))
class _SurrogatePrior(eqx.Module):
    """A GP prior in the surrogate's coordinates theta = (z.ravel(), u) ~ N(0, I), level = level_mu + level_sd u,
    log10 I = level + mean + field(z), with field linear in z (field(z) = B z). Subclasses give field / field_T (B^T) /
    congruence (B^T H B) / field_matrix (B W) and meta_arrays (what a grid point's digest covers). All leaves are
    arrays, so jitted functions taking a prior never recompile when its hyperparameters change."""

    @property
    def wid(self):
        return self.mean.shape[0]

    def log10_image(self, theta):
        return self.level_mu + self.level_sd*theta[-1] + self.mean + self.field(theta[:-1])

    def distribution(self, theta):
        return 10**self.log10_image(theta)

    def theta(self, z, level):
        return np.concatenate([np.ravel(z), np.atleast_1d((level - self.level_mu)/self.level_sd)])

    def params(self, theta):
        """(z, level) for the fit's "resolved" and "resolved_level"."""
        return theta[:-1].reshape(self.latent_shape), self.level_mu + self.level_sd*theta[-1]


class GPPrior(_SurrogatePrior):
    """The DCT Matern prior: B = Psi diag(amp), applied by fast DCTs."""
    amp: jax.Array
    mean: jax.Array
    level_mu: jax.Array
    level_sd: jax.Array
    sigma: jax.Array
    length: jax.Array
    order: jax.Array

    @property
    def latent_shape(self):
        return self.amp.shape

    @property
    def n_latent(self):
        return self.amp.size

    def field(self, z):
        return gp_field(np.reshape(z, self.amp.shape), self.amp)

    def field_T(self, v):
        w = self.amp.shape[0]
        return self.amp.ravel()*_dct2(v.reshape(w, w)).ravel()

    def congruence(self, H):
        w = self.amp.shape[0]
        a = self.amp.ravel()
        return _congruence_dct(H.reshape(w, w, w, w)).reshape(w*w, w*w)*a[:, None]*a[None, :]

    def field_matrix(self, Wz):
        w = self.amp.shape[0]
        return jsp.fft.idctn(self.amp[:, :, None]*Wz.reshape(w, w, -1), type=2, norm="ortho", axes=(0, 1)).reshape(w*w, -1)

    def fit_field(self, f):
        """The z whose field is closest to the log10 field f (exact on the modes with amp > 0)."""
        a = self.amp.ravel()
        return np.where(a > 0, _dct2(f.reshape(self.amp.shape)).ravel()/np.where(a > 0, a, 1.), 0.)

    def meta_arrays(self):
        return (self.sigma, self.length, self.order, self.mean, self.level_mu, self.level_sd)


def gp_prior(wid, sigma, length, order=2., mean=None, level_prior=(0., 1.)):
    length = np.broadcast_to(np.asarray(length, dtype=float), (2,))
    mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    sigma, order = np.asarray(sigma, dtype=float), np.asarray(order, dtype=float)
    return GPPrior(matern_amplitude((wid, wid), length, sigma, order), mean, mu, sd, sigma, length, order)


class PolarPrior(_SurrogatePrior):
    """The polar prior: B (wid^2, n_r n_theta) = d log10 I / d z at one geometry, a dense matrix built by polar_prior."""
    B: jax.Array
    mean: jax.Array
    level_mu: jax.Array
    level_sd: jax.Array
    sigma: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    geometry: jax.Array
    nodes: jax.Array
    latent_shape: tuple = eqx.field(static=True)
    grid_digest: str = eqx.field(static=True)

    @property
    def n_latent(self):
        return self.B.shape[1]

    def field(self, z):
        w = self.mean.shape[0]
        return (self.B @ np.ravel(z)).reshape(w, w)

    def field_T(self, v):
        return self.B.T @ v

    def congruence(self, H):
        return self.B.T @ (H @ self.B)

    def field_matrix(self, Wz):
        return self.B @ Wz

    def fit_field(self, f):
        """The minimum-norm z whose field is closest to the log10 field f (ridge-regularised B^T (B B^T)^-1 f). Call it
        eagerly or with the prior as a jit argument: closed over in a jit, B B^T is constant-folded (minutes)."""
        BBt = self.B @ self.B.T
        ridge = 1e-8*np.trace(BBt)/BBt.shape[0]
        return self.B.T @ np.linalg.solve(BBt + ridge*np.eye(BBt.shape[0]), f.ravel())

    def meta_arrays(self):
        return (self.sigma, self.length_r, self.length_theta, self.geometry, self.mean, self.level_mu, self.level_sd,
                onp.frombuffer(self.grid_digest.encode(), dtype=onp.uint8))


@eqx.filter_jit
def _polar_basis(grid, sigma, length_r, length_theta, geometry):
    """B = d(log10 field)/dz, column by column: the field is linear in z, so column i is the field of the i-th unit
    latent (built from the traced index, cf. precond._map_tangents; batched to bound memory). Everything is traced,
    geometry included, so this compiles once per grid."""
    n = grid.n_latent
    interp = grid.interpolation(geometry)
    cols = jax.lax.map(lambda i: grid.field(jax.nn.one_hot(i, n, dtype=float), sigma, length_r, length_theta, geometry,
                                            interp).ravel(), np.arange(n), batch_size=256)
    return cols.T


def polar_prior(grid, sigma, length_r, length_theta, cos_inc=1., pa=0., dx=0., dy=0., mean=None, level_prior=(0., 1.)):
    mean = np.zeros((grid.wid, grid.wid)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    sigma, length_r, length_theta = (np.asarray(x, dtype=float) for x in (sigma, length_r, length_theta))
    geometry = _geometry(cos_inc, pa, dx, dy)
    return PolarPrior(_polar_basis(grid, sigma, length_r, length_theta, geometry), mean, mu, sd, sigma, length_r,
                      length_theta, geometry, grid.nodes, grid.shape, grid.digest())


def gp_prior_from_fit(fit):
    """The surrogate prior of a GP fit's current hyperparameters."""
    return fit.surrogate_prior()


# ----------------------------------------------------------------------------- multi-kernel DCT prior
# A sum of K independent stationary GPs on the same DCT basis is one GP whose DCT spectrum is the sum of the kernels'
# spectra, so the field needs the SAME latents z (wid x wid), the same fast transforms and the whole existing machinery:
#
#     amp = sqrt(sum_k S_k),   S_k = matern_spectrum(shape, length_k, sigma_k, order_k)   (each S_k has mean sigma_k^2,
#                                                                                       its zero mode removed)
#     field = idctn(amp * z),  pixel variance = sum_k sigma_k^2,  covariance = sum_k Cov_k exactly.
#
# GPPrior just receives a different `amp` (its sigma / length / order leaves are the (K,) hyperparameter arrays, used only
# by meta_arrays, i.e. the grid-point digests). Nothing above changes behaviour; HYPER_NAMES only gains the names
# sigmas, lengths, orders (used by check_gp_setup).

def multi_matern_spectrum(shape, sigmas, lengths, orders):
    """Sum over kernels of matern_spectrum: sigmas, lengths, orders are (K,) (isotropic lengths in grid px). Traced-safe
    (changing the hyperparameters never recompiles; K is the only static quantity)."""
    sigmas, lengths, orders = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (sigmas, lengths, orders))
    return jax.vmap(lambda s, l, o: matern_spectrum(shape, l, s, o))(sigmas, lengths, orders).sum(axis=0)


def multi_matern_amplitude(shape, sigmas, lengths, orders):
    """sqrt(multi_matern_spectrum), with a finite gradient at the removed constant mode (S = 0 there for every kernel)."""
    S = multi_matern_spectrum(shape, sigmas, lengths, orders)
    pos = S > 0
    return np.where(pos, np.sqrt(np.where(pos, S, 1.)), 0.)


def multi_gp_prior(wid, sigmas, lengths, orders, mean=None, level_prior=(0., 1.)):
    """GPPrior of the K-kernel spectrum; prior.sigma / length / order hold the (K,) arrays (meta_arrays digests them)."""
    sigmas, lengths, orders = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (sigmas, lengths, orders))
    assert sigmas.shape == lengths.shape == orders.shape and sigmas.ndim == 1, "sigmas, lengths, orders: (K,) each"
    mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    return GPPrior(multi_matern_amplitude((wid, wid), sigmas, lengths, orders), mean, mu, sd, sigmas, lengths, orders)


# ----------------------------------------------------------------------------- fit
class MultiGPResolvedFit(_GPFitBase):
    """GPResolvedFit with a K-kernel spectrum. Hyperparameters (set_hyper names): sigmas, lengths, orders ((K,) each),
    mean, level_prior. GPResolvedFit itself is untouched (its sigma / length / order names stay as they were)."""
    HYPER = ("sigmas", "lengths", "orders", "mean", "level_prior")
    sigmas: jax.Array
    lengths: jax.Array
    orders: jax.Array

    def __init__(self, spectrum_basis, filter, wid, sigmas=(0.5, 0.5), lengths=(1.6, 12.8), orders=(2., 2.), mean=None,
                 level_prior=(0., 1.), n_share=1, resolved_kwargs=None):
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.sigmas, self.lengths, self.orders = (np.atleast_1d(np.asarray(x, dtype=float))
                                                  for x in (sigmas, lengths, orders))
        assert self.sigmas.shape == self.lengths.shape == self.orders.shape and self.sigmas.ndim == 1
        self._init_gp(wid, mean, level_prior, n_share)

    def amplitude(self):
        return multi_matern_amplitude(self.mean.shape, self.sigmas, self.lengths, self.orders)

    def log10_field(self, z):
        return gp_field(z, self.amplitude())

    def surrogate_prior(self):
        return multi_gp_prior(self.mean.shape[0], self.sigmas, self.lengths, self.orders, self.mean, self.level_prior)


# ----------------------------------------------------------------------------- two-kernel grid parametrisation
# Label-switching is removed by ordering, imposed through the grid axes (the grid has no masks): kernel 1 is the SHORT one,
# length2 = length1 * ratio with the "ratio" axis bounded below by > 1. The hyperprior is then uniform in
# (log length1, log ratio) on a box, i.e. on the triangle-like region length1 < length2 (the Jacobian of
# (log l1, log l2) -> (log l1, log ratio) is 1).
TWO_AXES = ("sigma1", "length1", "sigma2", "ratio")
TWO_ABBREV = {"sigma1": "s1", "length1": "l1", "sigma2": "s2", "ratio": "r"}


def two_kernel_prior_factory(wid, orders=(2., 2.), mean=None, level_prior=(0., 1.)):
    """make_prior for gp_evidence_grid_nd with axes sigma1, length1, sigma2, ratio (length2 = length1 * ratio)."""
    def make_prior(sigma1, length1, sigma2, ratio):
        return multi_gp_prior(wid, [sigma1, sigma2], [length1, length1*ratio], orders, mean, level_prior)
    return make_prior


def two_kernel_hypers(point):
    """{sigma1, length1, sigma2, ratio} -> (sigmas, lengths) for MultiGPResolvedFit / set_hyper."""
    return ([point["sigma1"], point["sigma2"]], [point["length1"], point["length1"]*point["ratio"]])


# ----------------------------------------------------------------------------- capped evidence (gp_hyper_posterior has no mask)
def capped_posterior(grid, max_length2=30., bounds=None):
    """Posterior weights and log evidence of a two-kernel grid (axes sigma1, length1, sigma2, ratio) under the hyperprior
    uniform (in logs) on the bounds BUT restricted to length2 = length1 * ratio <= max_length2 (cells whose centre exceeds
    it get zero prior mass; the rest is renormalised), so a plateau of lengths beyond the box (logZ identical there) does
    not inflate the prior volume, and the single kernel's length <= 30 bound is matched. Only converged points carry
    weight (as gp_hyper_posterior). Assumes the grid covers the allowed region (check edge_weight of the unmasked
    posterior). Returns weights (grid-shaped), log_evidence_bounds, allowed (mask)."""
    axs = _axes(grid)
    bounds = {**grid.get("bounds", {}), **(bounds or {})}
    names = [a.name for a in axs]
    assert names == list(TWO_AXES), names
    logZ = onp.asarray(grid["logZ"], dtype=float)
    ok = onp.asarray(grid["converged"]).astype(bool)
    shape = logZ.shape
    logmass = onp.zeros(shape)
    for i, a in enumerate(axs):
        sh = [1]*logZ.ndim
        sh[i] = -1
        logmass = logmass + a.log_masses(bounds[a.name]).reshape(sh)
    l1 = onp.asarray(axs[1].vals).reshape(1, -1, 1, 1)
    r = onp.asarray(axs[3].vals).reshape(1, 1, 1, -1)
    allowed = onp.broadcast_to(l1*r <= max_length2*(1 + 1e-9), shape)
    log_prior = onp.where(allowed, logmass, -onp.inf)
    log_prior = log_prior - onp.logaddexp.reduce(log_prior.ravel())          # renormalised on the allowed region
    logpost = onp.where(ok, logZ, -onp.inf) + log_prior
    lse = onp.logaddexp.reduce(logpost.ravel())
    return dict(weights=onp.exp(logpost - lse), log_evidence_bounds=float(lse), allowed=allowed)


def converged_max(grid):
    """max log Z over the converged points (gp_best's rule)."""
    logZ, ok = onp.asarray(grid["logZ"], dtype=float), onp.asarray(grid["converged"]).astype(bool)
    return float(onp.max(onp.where(ok, logZ, -onp.inf)) if ok.any() else onp.max(logZ))


def capped_best(grid, allowed):
    """(index tuple, {name: value}) of the best converged point among the `allowed` cells."""
    logZ = onp.where(onp.asarray(grid["converged"]).astype(bool) & allowed, onp.asarray(grid["logZ"], dtype=float), -onp.inf)
    idx = tuple(int(i) for i in onp.unravel_index(onp.argmax(logZ), logZ.shape))
    return idx, {k: float(grid["axes"][k][i]) for k, i in zip(grid["names"], idx)}


# ----------------------------------------------------------------------------- broken power-law spectrum
# One spectrum on the same DCT latents (GPPrior just receives another `amp`), with a low-k slope gamma1, a high-k slope
# gamma2 and a smooth break (fixed sharpness s = 2) at the wavenumber k_b = 1 / length_b:
#
#     S(k) ~ k^-gamma1 [1 + (k / k_b)^2]^(-(gamma2 - gamma1) / 2),   k^2 = lap_0 + lap_1 (the DCT / Neumann eigenvalues
#                                                                     of matern_spectrum, rad / px; f = k / 2 pi cycles / px)
#
# normalised to a mean pixel variance sigma^2 with the constant mode removed, exactly as matern_spectrum. For gamma1 = 0
# this is matern_spectrum(shape, length_b, sigma, order = gamma2 / 2) EXACTLY (1 + l^2 k^2 is its base), so the Matern
# prior is the gamma1 = 0 face of this one and a like-for-like evidence comparison is a nested one. The high-k slope is
# gamma2 (the Matern's is 2 order). For length_b >> 1 / k_min (the lowest mode, ~ wid / pi px) the spectrum is the pure
# power law k^-gamma2 and (length_b, gamma1) are not separately identified. Because the normalisation sums over modes, for
# gamma1 > 2 a handful of the lowest modes carry most of sigma^2.

def bpl_spectrum(shape, sigma, length_b, gamma1, gamma2):
    """Broken power-law power spectrum (variance per DCT-II mode) on `shape`, normalised to a mean pixel variance sigma^2,
    constant mode removed. length_b (px, isotropic) is 1 / k_b; all arguments may be traced (no recompiles)."""
    lap = [(2*np.sin(np.pi*np.arange(n)/(2*n)))**2 for n in shape]
    q2 = lap[0][:, None] + lap[1][None, :]
    pos = q2 > 0
    q2 = np.where(pos, q2, 1.)                      # safe at the removed mode (finite gradients)
    g1, g2 = (np.asarray(g, dtype=float) for g in (gamma1, gamma2))
    S = np.where(pos, q2**(-0.5*g1)*(1 + length_b**2*q2)**(-0.5*(g2 - g1)), 0.)
    return sigma**2 * S * S.size / S.sum()


def bpl_amplitude(shape, sigma, length_b, gamma1, gamma2):
    """sqrt(bpl_spectrum), with a finite gradient at the removed constant mode."""
    S = bpl_spectrum(shape, sigma, length_b, gamma1, gamma2)
    pos = S > 0
    return np.where(pos, np.sqrt(np.where(pos, S, 1.)), 0.)


def bpl_from_matern(length, order):
    """(length_b, gamma1, gamma2) whose bpl_spectrum is matern_spectrum(length, order) (isotropic length)."""
    return length, 0., 2.*order


def bpl_gp_prior(wid, sigma, length_b, gamma1, gamma2, mean=None, level_prior=(0., 1.)):
    """GPPrior of the broken power-law spectrum; prior.length = length_b and prior.order = (gamma1, gamma2) (they only
    enter meta_arrays, the grid-point digests)."""
    mean = np.zeros((wid, wid)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    sigma, length_b, gamma1, gamma2 = (np.asarray(x, dtype=float) for x in (sigma, length_b, gamma1, gamma2))
    return GPPrior(bpl_amplitude((wid, wid), sigma, length_b, gamma1, gamma2), mean, mu, sd, sigma, length_b,
                   np.stack([gamma1, gamma2]))


class BPLGPResolvedFit(_GPFitBase):
    """GPResolvedFit with the broken power-law spectrum. Hyperparameters (set_hyper names): sigma (dex), length_b (px),
    gamma1, gamma2, mean, level_prior."""
    HYPER = ("sigma", "length_b", "gamma1", "gamma2", "mean", "level_prior")
    sigma: jax.Array
    length_b: jax.Array
    gamma1: jax.Array
    gamma2: jax.Array

    def __init__(self, spectrum_basis, filter, wid, sigma=1., length_b=6., gamma1=1., gamma2=3., mean=None,
                 level_prior=(0., 1.), n_share=1, resolved_kwargs=None):
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.sigma, self.length_b, self.gamma1, self.gamma2 = (np.asarray(x, dtype=float)
                                                               for x in (sigma, length_b, gamma1, gamma2))
        self._init_gp(wid, mean, level_prior, n_share)

    def amplitude(self):
        return bpl_amplitude(self.mean.shape, self.sigma, self.length_b, self.gamma1, self.gamma2)

    def log10_field(self, z):
        return gp_field(z, self.amplitude())

    def surrogate_prior(self):
        return bpl_gp_prior(self.mean.shape[0], self.sigma, self.length_b, self.gamma1, self.gamma2, self.mean,
                            self.level_prior)


# Grid parametrisation: the high-k slope is gamma2 = gamma1 + dgamma with dgamma > 0 (a steepening break), so the
# hyperprior is a plain box in (log sigma, log length_b, gamma1, dgamma) (unit Jacobian from (gamma1, gamma2)) and every
# constraint is an axis bound: gp_hyper_posterior's quantiles, edge weights and log_evidence_bounds all see it (no mask
# as capped_posterior needs). The Matern face is gamma1 = 0, dgamma = 2 order.
BPL_AXES = ("sigma", "length_b", "gamma1", "dgamma")
BPL_ABBREV = {"sigma": "s", "length_b": "lb", "gamma1": "g1", "dgamma": "dg"}
BPL_SCALES = {"gamma1": "linear", "dgamma": "linear"}      # sigma and length_b are log axes (the default)


def bpl_prior_factory(wid, mean=None, level_prior=(0., 1.)):
    """make_prior for gp_evidence_grid_nd with axes BPL_AXES (gamma2 = gamma1 + dgamma)."""
    def make_prior(sigma, length_b, gamma1, dgamma):
        return bpl_gp_prior(wid, sigma, length_b, gamma1, gamma1 + dgamma, mean, level_prior)
    return make_prior


def bpl_hypers(point):
    """{sigma, length_b, gamma1, dgamma} -> {sigma, length_b, gamma1, gamma2} for BPLGPResolvedFit / set_hyper."""
    return dict(sigma=point["sigma"], length_b=point["length_b"], gamma1=point["gamma1"],
                gamma2=point["gamma1"] + point["dgamma"])



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
        assert isinstance(e.fit, _GPFitBase), f"{e.key}: LinearResponse needs a GP fit, got {type(e.fit)}"
        # the response and its check both inject the disc through d_override, so a subclass must not bypass it
        assert type(e.fit).get_distribution is _GPFitBase.get_distribution, \
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
    """J^T G J for theta = (z, u), J = dd/dtheta = ln10 diag(d) [B | level_sd 1], B = d(field)/dz (prior.congruence:
    two 2D DCTs for the DCT prior, B^T H B for the polar one)."""
    D = LN10*d
    H = G*D[:, None]*D[None, :]
    r = H.sum(1)
    zu = prior.field_T(r)*prior.level_sd
    top = np.concatenate([prior.congruence(H), zu[:, None]], axis=1)
    bottom = np.concatenate([zu, (r.sum()*prior.level_sd**2)[None]])[None, :]
    return np.concatenate([top, bottom], axis=0)


def _grad(G, b, d, prior, theta):
    """Gradient of 0.5 d^T G d - b^T d + 0.5 |theta|^2 in theta."""
    Dg = LN10*d*(G @ d - b)
    return np.concatenate([prior.field_T(Dg), (prior.level_sd*Dg.sum())[None]]) + theta


def _log10_step(prior, delta):
    """Change of log10 I (flat) for a step delta in theta."""
    return prior.field(delta[:-1]).ravel() + prior.level_sd*delta[-1]


@eqx.filter_jit
def _map_core(resp, beta, prior, theta0, max_iter, tol, max_dex):
    G, b, yy = _weighted(resp, beta)
    eye = np.eye(theta0.size)

    def obj(theta):
        d = prior.distribution(theta).ravel()
        f = 0.5*(d @ (G @ d)) - b @ d + 0.5*yy + 0.5*theta @ theta
        return np.where(np.isfinite(f), f, np.inf)

    # a start worse than the prior mean (e.g. a warm start far from these hyperparameters, or one whose image overflows)
    # is replaced by it
    theta0 = np.where(obj(theta0) <= obj(np.zeros_like(theta0)), theta0, np.zeros_like(theta0))

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
    theta0 = np.zeros(prior.n_latent + 1) if theta0 is None else np.asarray(theta0, dtype=float)
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
    # tr(M^-1 A_e), M = I + A, A_e = J^T G_e J: each exposure's share of the well-measured directions, divided by beta_e.
    # In pixel space: tr(M^-1 J^T G_e J) = sum(G_e * (J M^-1 J^T)), J M^-1 J^T = (JV / (1 + mu)) (JV)^T
    JV = LN10*d[:, None]*(prior.field_matrix(V[:-1]) + prior.level_sd*V[-1][None, :])
    K = (JV/(1 + mu)) @ JV.T
    tr = jax.lax.map(lambda Ge: np.sum(Ge*K), resp.G)
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
def _point_meta(resp, prior, beta, n_iter):
    """Digest of everything a grid point depends on, stored with it and checked on resume."""
    import hashlib
    h = hashlib.sha256(type(prior).__name__.encode())
    for a in (resp.b, resp.yy, resp.n_good, resp.G.diagonal(axis1=1, axis2=2), beta, n_iter, *prior.meta_arrays()):
        h.update(onp.ascontiguousarray(onp.asarray(a, dtype=float)).tobytes())
    return h.hexdigest()[:16]


def _ratio(vals, name):
    """The common ratio of a log-spaced axis (None for a single value)."""
    if len(vals) < 2:
        return None
    r = onp.asarray(vals[1:])/onp.asarray(vals[:-1])
    assert onp.allclose(r, r[0], rtol=1e-6) and r[0] > 1, f"{name} must be log-spaced with one ratio > 1, got {vals}"
    return float(r[0])


def _difference(vals, name):
    """The common step of a linearly spaced axis (None for a single value)."""
    if len(vals) < 2:
        return None
    d = onp.diff(vals)
    assert onp.allclose(d, d[0], rtol=1e-6, atol=1e-12) and d[0] > 0, \
        f"{name} must be evenly spaced with one step > 0, got {vals}"
    return float(d[0])


class _Axis:
    """One grid axis: scale "log" (a common ratio), "linear" (a common step) or "periodic" (a common step, period
    `period`, values covering the whole period: no edges, no growth). step is the ratio (log) or the difference."""

    def __init__(self, name, vals, scale="log", period=None):
        assert scale in ("log", "linear", "periodic"), f"{name}: scale {scale}"
        self.name, self.scale, self.period = name, scale, period
        self.vals = sorted(float(v) for v in vals)
        self.step = _ratio(self.vals, name) if scale == "log" else _difference(self.vals, name)
        if scale == "periodic":
            assert period is not None and self.step is not None and onp.isclose(self.step*len(self.vals), period), \
                f"{name}: a periodic axis needs evenly spaced values covering its period {period}, got {self.vals}"

    def distance(self, a, b):
        """|a - b| in steps (log: in log; periodic: wrapped)."""
        if self.step is None:
            return 0.
        if self.scale == "log":
            return abs(onp.log(a/b))/onp.log(self.step)
        d = abs(a - b)
        if self.scale == "periodic":
            d = min(d % self.period, -d % self.period)
        return d/self.step

    def next_out(self, lower):
        v = self.vals[0] if lower else self.vals[-1]
        if self.scale == "log":
            return v/self.step if lower else v*self.step
        return v - self.step if lower else v + self.step

    def can_grow(self, lower, bounds):
        if self.step is None or self.scale == "periodic":
            return False
        lo, hi = bounds
        v = self.next_out(lower)
        return v >= lo - 1e-9*abs(lo) - 1e-12 if lower else v <= hi + 1e-9*abs(hi) + 1e-12

    def measure(self, v):
        """The axis's uniform measure: log v for "log", v otherwise."""
        with onp.errstate(divide="ignore"):
            return onp.log(onp.asarray(v, dtype=float)) if self.scale == "log" else onp.asarray(v, dtype=float)

    def cell_edges(self, bounds=None):
        """(lower, upper) edges of each cell in the axis's measure, half a step either side of its value, clipped to
        finite bounds (so a cell at a bound does not count prior mass beyond it)."""
        t = self.measure(self.vals)
        h = 0. if self.step is None else 0.5*(onp.log(self.step) if self.scale == "log" else self.step)
        e0, e1 = t - h, t + h
        if bounds is not None and self.scale != "periodic":
            e0, e1 = onp.maximum(e0, self.measure(bounds[0])), onp.minimum(e1, self.measure(bounds[1]))
        return e0, e1

    def log_masses(self, bounds):
        """log prior mass of each cell under the axis's uniform hyperprior on `bounds` (cells clipped to them); None
        when the bounds are not finite in the axis's measure."""
        if self.step is None:
            return onp.zeros(len(self.vals))
        if self.scale == "periodic":
            return onp.full(len(self.vals), onp.log(self.step/self.period))
        if bounds is None:
            return None
        lo, hi = self.measure(bounds[0]), self.measure(bounds[1])
        if not (onp.isfinite(lo) and onp.isfinite(hi)):
            return None
        e0, e1 = self.cell_edges(bounds)
        with onp.errstate(divide="ignore"):
            return onp.log(onp.maximum(e1 - e0, 0.)) - onp.log(hi - lo)

    def at_bound(self, lower, bounds):
        """Whether the face's cell reaches the hyperprior's bound (posterior mass there is not truncated by the grid); a
        bound not finite in the axis's measure is never reached."""
        if bounds is None:
            return False
        b = self.measure(bounds[0] if lower else bounds[1])
        if not onp.isfinite(b):
            return False
        t = self.measure(self.vals[0] if lower else self.vals[-1])
        h = 0. if self.step is None else 0.5*(onp.log(self.step) if self.scale == "log" else self.step)
        return bool(t - h <= b + 1e-9*abs(b) + 1e-12) if lower else bool(t + h >= b - 1e-9*abs(b) - 1e-12)


def _axes(grid):
    """_Axis objects of a grid dict (a legacy 2D DCT grid: log sigma, length)."""
    names = grid.get("names", ["sigma", "length"])
    vals = grid.get("axes") or {"sigma": grid["sigmas"], "length": grid["lengths"]}
    scales, periods = grid.get("scales", {}), grid.get("periods", {})
    return [_Axis(k, vals[k], scales.get(k, "log"), periods.get(k)) for k in names]


def gp_best(grid, converged_only=True):
    """{name: value} at the grid's best log Z among its converged points (an unconverged MAP's log Z is not
    trustworthy); warns and uses every point if none converged."""
    logZ = onp.asarray(grid["logZ"], dtype=float)
    if converged_only:
        ok = onp.asarray(grid["converged"]).astype(bool)
        if ok.any():
            logZ = onp.where(ok, logZ, -onp.inf)
        else:
            warnings.warn(f"gp_best {grid.get('tag')}: no converged point; using the best unconverged one")
    idx = onp.unravel_index(onp.argmax(logZ), logZ.shape)
    return {a.name: float(a.vals[i]) for a, i in zip(_axes(grid), idx)}


def local_axes(center, steps, scales=None, n=1):
    """Axes {name: values} of a local grid about `center` ({name: value}): 2n + 1 values per axis, spaced by steps[name]
    (a ratio for a "log" axis, else a difference; a "periodic" axis becomes a linear one about the centre, with its
    period's uniform prior kept through bounds = centre -+ period / 2)."""
    scales = scales or {}
    out = {}
    for k, c in center.items():
        if scales.get(k, "log") == "log":
            out[k] = [c*steps[k]**i for i in range(-n, n + 1)]
        else:
            out[k] = [c + steps[k]*i for i in range(-n, n + 1)]
    return out


def gp_evidence_grid_nd(resp, axes, make_prior, beta=None, outdir=None, tag="grid", abbrev=None, max_growth=3,
                        bounds=None, scales=None, periods=None, n_iter=200, theta0=None, converged_tol=1e-3,
                        edge_dlogz=5., recheck_top=3, verbose=True):
    """Laplace evidence over a grid of hyperparameters, each point a warm-started MAP.

    axes: {name: values}; scales: {name: "log" (default; one common ratio) | "linear" (one common step) | "periodic"
    (one common step covering periods[name])}; make_prior(**{name: value}) -> surrogate prior. While a face of the grid
    holds a point within edge_dlogz of the best log Z (posterior mass at the edge, not only the argmax), the grid is
    extended by one step past that face, at most max_growth times and within bounds ({name: (lo, hi)}; periodic axes
    never grow). Warm starts keep the nearest evaluated point's log10 field (prior.fit_field), nearness in steps along
    each axis. A non-finite log Z counts as -inf (and unconverged). With `outdir`, each point is saved as
    {outdir}/{tag}_{abbrev[name]}{value}....npz with a digest of the response, beta and prior settings, and reloaded on a
    rerun only if that digest matches (else recomputed). The recheck_top best points are re-solved from the prior mean
    (a warm start can land in a worse MAP) and the better solution kept. Returns a dict: names, axes {name: values},
    scales, periods,
    bounds, logZ / gamma / chi2 (summed) / newton_decrement / converged / steps (grid-shaped), thetas (grid shape +
    (n + 1,)), best (index tuple of the best converged point, as gp_best), tag."""
    names = list(axes)
    abbrev = abbrev or {k: k for k in names}
    bounds = dict(bounds or {})
    scales = {k: (scales or {}).get(k, "log") for k in names}
    periods = {k: (periods or {}).get(k) for k in names}
    ax = {k: _Axis(k, axes[k], scales[k], periods[k]) for k in names}
    for k in names:   # starting values outside the bounds are dropped (e.g. a local grid about an optimum near a bound)
        if scales[k] == "periodic":
            continue
        lo, hi = bounds.setdefault(k, (0., onp.inf) if scales[k] == "log" else (-onp.inf, onp.inf))
        keep = [v for v in ax[k].vals if lo - 1e-9*abs(lo) - 1e-12 <= v <= hi + 1e-9*abs(hi) + 1e-12]
        if len(keep) < len(ax[k].vals):
            warnings.warn(f"gp_evidence_grid {tag}: {k} outside {(lo, hi)} dropped: {sorted(set(ax[k].vals) - set(keep))}")
        assert keep, f"gp_evidence_grid {tag}: no {k} within {(lo, hi)}"
        ax[k].vals = keep
    beta = _beta(resp, beta)
    results = {}
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    def path(pt):
        return os.path.join(outdir, tag + "".join(f"_{abbrev[k]}{v:.6g}" for k, v in zip(names, pt)) + ".npz")

    def run(points):
        for pt in points:
            if pt in results:
                continue
            prior = make_prior(**dict(zip(names, pt)))
            meta = _point_meta(resp, prior, beta, n_iter)
            fpath = None if outdir is None else path(pt)
            if fpath is not None and os.path.exists(fpath):
                with onp.load(fpath) as f:
                    if "meta" in f.files and str(f["meta"]) == meta:
                        results[pt] = {k: f[k] for k in f.files}
                        continue
                if verbose:
                    print(f"[gp grid {tag}] {fpath} was made with other inputs; recomputing", flush=True)
            if results:   # warm start from the nearest evaluated point (in steps along each axis), keeping its field
                near = min(results, key=lambda q: sum(ax[k].distance(a, b)**2 for k, a, b in zip(names, q, pt)))
                th = onp.asarray(results[near]["theta"])
                start = (np.concatenate([prior.fit_field(np.asarray(results[near]["field"])), np.asarray(th[-1:])])
                         if "field" in results[near] else np.asarray(th))
            else:
                start = theta0
            solve(pt, prior, meta, fpath, start, cold_checked=start is None)   # a cold first point needs no recheck

    def solve(pt, prior, meta, fpath, start, cold_checked=False):
        theta, info = gp_map(resp, prior, start, beta, n_iter)
        ev = jax.tree.map(onp.asarray, gp_log_evidence(resp, prior, theta, beta))
        r = dict(zip(names, pt), theta=onp.asarray(theta), field=onp.asarray(prior.field(theta[:-1])), meta=meta,
                 steps=int(info["steps"]), cold_checked=cold_checked,
                 **{k: ev[k] for k in ("logZ", "nll", "chi2", "prior", "occam", "gamma", "gamma_e",
                                      "newton_decrement", "grad_norm")})
        finite = bool(onp.isfinite(r["logZ"]))
        r["logZ"] = r["logZ"] if finite else onp.asarray(-onp.inf)
        r["converged"] = finite and bool(r["newton_decrement"] < converged_tol)
        if pt in results and ((bool(results[pt]["converged"]), float(results[pt]["logZ"]))
                              >= (r["converged"], float(r["logZ"]))):   # converged first, then the higher log Z
            results[pt]["cold_checked"] = onp.asarray(True)   # the warm solution stands
            if fpath is not None:
                _save_npz(fpath, **results[pt])
            return
        results[pt] = r
        if fpath is not None:
            _save_npz(fpath, **r)
        if verbose:
            print(f"[gp grid {tag}] " + ", ".join(f"{k} {v:.4g}" for k, v in zip(names, pt))
                  + f": logZ {float(r['logZ']):.3f}, gamma {float(r['gamma']):.1f}, decrement "
                  f"{float(r['newton_decrement']):.1e}, {r['steps']} steps", flush=True)

    def points():
        return list(itertools.product(*(ax[k].vals for k in names)))

    def table(key, reduce=None):
        a = onp.array([onp.asarray(results[pt][key]) for pt in points()])
        a = a if reduce is None else reduce(a)
        return a.reshape(tuple(len(ax[k].vals) for k in names) + a.shape[1:])

    for growth in range(max_growth + 1):
        run(points())
        logZ = table("logZ").astype(float)
        if growth == max_growth:
            break
        near_max = logZ > onp.max(logZ) - edge_dlogz
        grew = False
        for i, k in enumerate(names):
            for lower, face in ((True, 0), (False, -1)):
                if onp.take(near_max, face, axis=i).any() and ax[k].can_grow(lower, bounds.get(k)):
                    v = ax[k].next_out(lower)
                    ax[k].vals = [v] + ax[k].vals if lower else ax[k].vals + [v]
                    grew = True
        if not grew:
            break
    for flat in onp.argsort(logZ.ravel())[::-1][:recheck_top]:   # cold restarts of the best points
        pt = points()[flat]
        if bool(results[pt].get("cold_checked", False)):
            continue
        prior = make_prior(**dict(zip(names, pt)))
        old = float(results[pt]["logZ"])
        solve(pt, prior, str(results[pt]["meta"]), None if outdir is None else path(pt), None, cold_checked=True)
        if verbose and float(results[pt]["logZ"]) > old:
            print(f"[gp grid {tag}] cold start better by {float(results[pt]['logZ']) - old:.3g} at {pt}", flush=True)
    logZ, ok = table("logZ").astype(float), table("converged").astype(bool)
    if not ok.any():
        warnings.warn(f"gp_evidence_grid {tag}: no point converged; best is the best unconverged one")
    best = tuple(int(i) for i in onp.unravel_index(onp.argmax(onp.where(ok, logZ, -onp.inf) if ok.any() else logZ),
                                                   logZ.shape))   # the best converged point, as gp_best
    grid = dict(names=names, axes={k: onp.array(ax[k].vals) for k in names}, scales=scales, periods=periods,
                bounds=bounds, logZ=logZ, gamma=table("gamma"), chi2=table("chi2", lambda a: a.sum(-1)),
                newton_decrement=table("newton_decrement"), converged=table("converged"), steps=table("steps"),
                thetas=table("theta"), best=best, tag=tag)
    if not onp.isfinite(logZ).any():
        warnings.warn(f"gp_evidence_grid {tag}: no point has a finite log Z")
        return grid
    ew = gp_hyper_posterior(grid)["edge_weight"]
    if ew > 1e-2:
        warnings.warn(f"gp_evidence_grid {tag}: {ew:.2f} of the posterior weight is on faces short of the hyperprior's "
                      "bounds (growth limit, or a step past the bound): its intervals are truncated")
    return grid


def gp_evidence_grid(resp, sigmas, lengths, order=2., mean=None, level_prior=(0., 1.), beta=None, outdir=None,
                     tag="grid", max_growth=3, bounds=((0.03, 8.), (0.1, 30.)), n_iter=200, theta0=None,
                     converged_tol=1e-3, edge_dlogz=5., verbose=True):
    """gp_evidence_grid_nd for the DCT prior over (sigma, length) (isotropic length), with the 2D return keys sigmas,
    lengths, best (i, j), order, and point files {tag}_s{sigma}_l{length}.npz as before."""
    w = resp.wid
    grid = gp_evidence_grid_nd(
        resp, {"sigma": sigmas, "length": lengths},
        lambda sigma, length: gp_prior(w, sigma, length, order, mean, level_prior), beta=beta, outdir=outdir, tag=tag,
        abbrev={"sigma": "s", "length": "l"}, max_growth=max_growth,
        bounds={"sigma": bounds[0], "length": bounds[1]}, n_iter=n_iter, theta0=theta0, converged_tol=converged_tol,
        edge_dlogz=edge_dlogz, recheck_top=0, verbose=verbose)
    out = {k: v for k, v in grid.items() if k not in ("names", "axes", "scales", "periods", "bounds")}   # plain arrays
    return out | dict(sigmas=grid["axes"]["sigma"], lengths=grid["axes"]["length"], order=float(order))


def _quantiles(x, w, qs=(0.16, 0.5, 0.84), scale="log", period=None, bounds=None):
    """Quantiles of a distribution on evenly spaced cells x (weights w), interpolating the CDF at the cell edges (a
    cell spans half a step either side, clipped to finite bounds), in log x for "log", in x for "linear"; for
    "periodic", about the circular mean (x unwrapped to within half a period of it) and returned modulo the period."""
    x, w = onp.asarray(x, dtype=float), onp.asarray(w, dtype=float)
    if len(x) < 2:
        return onp.full(len(qs), x[0])
    if scale == "periodic":
        a = 2*onp.pi*x/period
        mean = onp.arctan2(onp.sum(w*onp.sin(a)), onp.sum(w*onp.cos(a)))*period/(2*onp.pi)
        u = mean + (x - mean + period/2) % period - period/2
        order = onp.argsort(u)
        q = _quantiles(u[order], w[order], qs, "linear") % period
        return onp.where(onp.isclose(q, period), 0., q)
    ax = _Axis("q", x, scale)
    e0, e1 = ax.cell_edges(bounds)   # an infinite bound clips nothing (maximum / minimum)
    edges = onp.concatenate([e0[:1], e1])
    cdf = onp.concatenate([[0.], onp.cumsum(w)/onp.sum(w)])
    q = onp.interp(qs, cdf, edges)
    return onp.exp(q) if scale == "log" else q


def gp_hyper_posterior(grid, box=None, bounds=None):
    """p(hyperparameters | data) on the grid under uniform hyperpriors on each axis (in log for "log" axes, else
    linear; each axis evenly spaced in that measure, so a uniform weight per cell): weights = softmax(logZ). Returns the
    weights, quantiles {name: (16, 50, 84 %)} of each marginal (circular for periodic axes), edge_weight (the posterior
    weight on faces whose cells stop short of the hyperprior's bounds: large means truncated intervals; a face whose
    cell reaches a bound, or on a periodic axis, is not an edge), and log_evidence = log of the mean of Z over the cells. log_evidence depends on the
    grid's extent; to compare grids of different extent pass box = {name: values} (a common set of cells; for a 2D DCT
    grid also a (sigmas, lengths) tuple) and compare log_evidence_box, or pass bounds = {name: (lo, hi)} (default: the
    grid's own) for log_evidence_bounds, the evidence under the uniform hyperprior on those fixed bounds (each cell's
    prior mass its width over the range; the posterior assumed ~0 outside the grid: check edge_weight), which compares
    models with different hyperparameters fairly given bounds chosen before seeing the data. Axes with a single value
    are fixed (no edge, quantiles = the value). A single-cell posterior gives intervals of one cell's width.
    Unconverged points (grid["converged"]) get zero weight throughout (an unconverged MAP's log Z is not trustworthy;
    every point is used, with a warning, if none converged), in log_evidence(_box, _bounds) too; unconverged_fraction
    is their share of the cells and unconverged_weight the largest posterior weight one of them would have had."""
    axs = _axes(grid)
    names = [a.name for a in axs]
    bounds = {**grid.get("bounds", {}), **(bounds or {})}
    logZ = onp.asarray(grid["logZ"], dtype=float)
    conv = onp.asarray(grid.get("converged", onp.ones(logZ.shape)), dtype=bool)
    ok = conv if conv.any() else onp.ones(logZ.shape, dtype=bool)   # the points that carry weight
    if not conv.any():
        warnings.warn(f"gp_hyper_posterior {grid.get('tag')}: no converged point; using every point")
    logZ_all, logZ = logZ, onp.where(ok, logZ, -onp.inf)
    # each cell's prior mass: uniform on each axis's bounds (cells at a bound clipped to it); axes without finite
    # bounds count every cell equally and leave log_evidence_bounds undefined
    masses = [a.log_masses(bounds.get(a.name)) for a in axs]
    bounded = all(m is not None for m in masses)
    logmass = onp.zeros(logZ.shape)
    for i, m in enumerate(masses):
        if m is not None:
            shape = [1]*logZ.ndim
            shape[i] = -1
            logmass = logmass + m.reshape(shape)
    logpost = logZ + logmass
    lse = onp.logaddexp.reduce(logpost.ravel())
    # the weight the unconverged points would have had, counting them
    lse_all = onp.logaddexp.reduce((logZ_all + logmass).ravel())
    w_un = onp.exp(logZ_all + logmass - lse_all)[~ok] if onp.isfinite(lse_all) else onp.zeros(0)
    w = onp.exp(logpost - lse) if onp.isfinite(lse) else onp.full(logZ.shape, onp.nan)
    edge = onp.zeros(w.shape, dtype=bool)
    for i, a in enumerate(axs):
        if len(a.vals) < 2 or a.scale == "periodic":
            continue
        for lower, face in ((True, 0), (False, len(a.vals) - 1)):
            if a.at_bound(lower, bounds.get(a.name)):   # the cell reaches the hyperprior's bound: not truncated
                continue
            sl = [slice(None)]*w.ndim
            sl[i] = face
            edge[tuple(sl)] = True
    q = {a.name: _quantiles(a.vals, w.sum(axis=tuple(j for j in range(w.ndim) if j != i)), scale=a.scale,
                            period=a.period, bounds=bounds.get(a.name)) for i, a in enumerate(axs)}
    out = dict(weights=w, quantiles=q, log_evidence=float(onp.logaddexp.reduce(logZ.ravel()) - onp.log(logZ.size)),
               edge_weight=float(w[edge].sum()), unconverged_fraction=float(onp.mean(~conv)),
               unconverged_weight=float(w_un.max()) if w_un.size else 0.)
    if bounded:
        out["log_evidence_bounds"] = float(lse)
    if "sigmas" in grid:   # the DCT grid's 2D keys
        out |= dict(sigma_q=q["sigma"], length_q=q["length"])
    if box is not None:
        if not isinstance(box, dict):
            box = dict(zip(names, box))
        idx = []
        for a in axs:
            vals = onp.asarray(a.vals)
            ii = [int(onp.argmin(onp.abs(vals - v))) for v in box[a.name]]
            assert onp.allclose(vals[ii], box[a.name]), "box cells must be grid cells"
            idx.append(ii)
        sub = logZ[onp.ix_(*idx)]
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
    w = prior.wid
    W = V/np.sqrt(1 + mu)
    LW = prior.field_matrix(W[:-1]) + prior.level_sd*W[-1][None, :]
    return np.sqrt(np.sum(LW**2, axis=1)).reshape(w, w)


def gp_spectral_diagnostic(prior, theta, laplace):
    """Posterior power of the whitened DCT modes, binned in radial wavenumber, against the prior's (1 per mode):
    <z_k^2> = z_map_k^2 + var_k, which averages to 1 when the prior's spectral shape is right. A systematic excess
    (deficit) over a range of wavenumbers means the Matern shape has too little (too much) power there. Also returns
    the mean posterior variance per bin (1: unconstrained by the data, 0: fully measured). freq is in cycles / px."""
    assert isinstance(prior, GPPrior), "gp_spectral_diagnostic is for the DCT prior; use gp_polar_diagnostic"
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


def gp_polar_diagnostic(prior, theta, laplace):
    """For the polar prior: posterior power of the whitened latents against the prior's (1 per latent), per azimuthal
    order |m| (orthonormal FFT of z along theta, which commutes with L_r, F_theta and the taper, so whitened mode m
    feeds only harmonic m) and per radial latent row k. An excess (deficit) at some |m| means the azimuthal spectrum
    has too little (too much) power there. The per-row numbers are only a rough radial guide: L_r is lower-triangular,
    so row k (the Cholesky innovation at node r_k) feeds every node at r >= r_k, and for length_r >> spacing the first
    rows set most of the field.
    Returns m, power_m, var_m (mean posterior variance: 1 unconstrained, 0 measured), count_m, power_m_err, and r
    (node radii, disc-plane px), power_r, var_r."""
    assert isinstance(prior, PolarPrior), "gp_polar_diagnostic is for the polar prior"
    mu, V = laplace
    n_r, n_t = prior.latent_shape
    W = (V/np.sqrt(1 + mu))[:-1].reshape(n_r, n_t, -1)
    Zf = np.fft.fft(theta[:-1].reshape(n_r, n_t), axis=1, norm="ortho")
    var = onp.asarray(np.sum(np.abs(np.fft.fft(W, axis=1, norm="ortho"))**2, axis=2))
    power = onp.abs(onp.asarray(Zf))**2 + var
    m = onp.asarray(_orders(n_t))
    ms = onp.unique(m)
    count = onp.array([n_r*onp.sum(m == k) for k in ms])
    zvar = onp.asarray(np.sum(W**2, axis=2))
    return dict(m=ms, power_m=onp.array([power[:, m == k].mean() for k in ms]),
                var_m=onp.array([var[:, m == k].mean() for k in ms]), count_m=count, power_m_err=onp.sqrt(2/count),
                r=onp.asarray(prior.nodes), power_r=onp.asarray(theta[:-1].reshape(n_r, n_t)**2 + zvar).mean(1),
                var_r=zvar.mean(1))


# =====================================================================================================================
# Polar + DCT (sum) prior
# =====================================================================================================================
# The two-component disc prior
#
#     log10 I = level + mean + field_polar(z_p) + field_dct(z_d)
#
# a smooth polar (multipole) field PLUS a stationary DCT Matern field. It is built on the gp.py machinery and plugs into the unchanged generic machinery (build_linear_response, verify_linear_response, gp_map,
# gp_log_evidence, gp_error_scale, gp_evidence_grid_nd, gp_laplace, gp_log10_std, ...) because that machinery only uses the
# prior through n_latent, field, field_T, congruence, field_matrix, fit_field, meta_arrays, distribution, params.
#
# Latents. theta = (z, u) with z = concat(z_p.ravel(), z_d.ravel()) FLAT, one fit parameter "resolved" of length
# n_p + wid^2 (so the L1/TSV-free GP plumbing, precond's C = 1 for "resolved", check_gp_setup, gp_image, optimise_new_resolved
# and `things_gp` all work unchanged; a second target-keyed parameter "resolved_dct" would need precond.py / fitting.py to
# learn it too: fitting.optimise_new_resolved hard-codes ones=("resolved",)). Split with split_latents.
#
# Polar part, optional azimuthal cutoff. With m_max the polar latents are the orthonormal real-Fourier coordinates of the
# theta axis, orders 0..m_max (2 m_max + 1 per ring: 1, cos m, sin m): Z = z_p Q^T with Q (n_theta, 2 m_max + 1)
# orthonormal, so z_p stays white N(0, 1) (a projection of white noise onto an orthonormal subspace). This is a HARD
# low-pass in m, but the cutoff is mainly a saving: it shrinks the latents (49 x 128 -> 49 x 9 for m_max = 4) and the build
# of B (one field evaluation per latent). What keeps the polar part smooth (no small-scale non-axisymmetric structure, i.e.
# no radial spokes) is the lower bound length_theta >= 1 rad together with sigma_p's prior; the cutoff makes that exact. sigma_p is the pointwise rms BEFORE the cutoff x 1 (the retained variance fraction of the
# azimuthal spectrum is divided out, so sigma_p is the rms of the retained field and decouples from length_theta).
# m_max = None keeps all n_theta latents (full polar field of gp.PolarGrid).


# ----------------------------------------------------------------------------- polar part with a cutoff
def azimuthal_basis(n_theta, m_max=None):
    """Q (n_theta, k): orthonormal real Fourier columns on the theta axis, orders 0..m_max as (1, cos m, sin m) (k = 2 m_max
    + 1; m_max < n_theta / 2, so no Nyquist), or the identity (k = n_theta) for m_max None."""
    if m_max is None:
        return np.eye(n_theta)
    assert 0 <= m_max and 2*m_max < n_theta, f"m_max {m_max} needs 2 m_max < n_theta = {n_theta}"
    th = 2*onp.pi*onp.arange(n_theta)/n_theta
    cols = [onp.full(n_theta, 1/onp.sqrt(n_theta))]
    for m in range(1, m_max + 1):
        cols += [onp.sqrt(2/n_theta)*onp.cos(m*th), onp.sqrt(2/n_theta)*onp.sin(m*th)]
    return np.asarray(onp.stack(cols, 1))


def retained_scale(length_theta, nu_theta, n_theta, m_max):
    """1/sqrt(f): f the fraction of the azimuthal spectrum's variance (azimuthal_filter's normalisation) at |m| <=
    m_max; 1 for m_max None."""
    if m_max is None:
        return np.asarray(1.)
    m = _orders(n_theta)
    lp = log_azimuthal_spectrum(m, length_theta, nu_theta)
    w = np.exp(lp - jsp.special.logsumexp(lp))
    return 1/np.sqrt(np.sum(np.where(m <= m_max, w, 0.)))


def polar_part(grid, Q, m_max, z_p, sigma, length_r, length_theta, geometry, interp=None):
    """log10 polar field on the disc pixels (wid, wid) from the (reduced) latents z_p, any shape with n_r x k entries."""
    Z = z_p.reshape(grid.shape[0], Q.shape[1]) @ Q.T
    s = sigma*retained_scale(length_theta, grid.nu_theta, grid.shape[1], m_max)
    return grid.field(Z, s, length_r, length_theta, geometry, interp)


@eqx.filter_jit
def _basis_unit(grid, Q, m_max, length_r, length_theta, geometry):
    """B_p at sigma = 1: column i = the polar field of the i-th unit (reduced) latent (field is linear in z_p). One field
    evaluation per column, built from the traced index (cf. _polar_basis), batched to bound memory."""
    n = grid.shape[0]*Q.shape[1]
    interp = grid.interpolation(geometry)
    cols = jax.lax.map(lambda i: polar_part(grid, Q, m_max, jax.nn.one_hot(i, n, dtype=float), 1., length_r,
                                            length_theta, geometry, interp).ravel(), np.arange(n), batch_size=256)
    return cols.T


_CACHE = {}
FIT_RIDGE = 0.01      # fit_field's ridge as a fraction of the mean eigenvalue of B B^T (unit ridge in z-units over-shrinks: B is in dex)
CACHE_MAX_ENTRIES = 16        # a grid over the polar axes (3^3 shape x 3^2 geometry ...) needs ~12-16 distinct bases
CACHE_MAX_BYTES = 2.5e9       # (3600 x 441 float64 = 13 MB each for m_max 4; 180 MB each with no cutoff)


def clear_basis_cache():
    """Call between grids (the cache is keyed on the polar hyperparameters, so a stale entry is only wasted memory)."""
    _CACHE.clear()


def _basis(grid, Q, m_max, length_r, length_theta, geometry):
    """B_p at sigma = 1, memoised (FIFO, by count and bytes) on (grid, m_max, length_r, length_theta, geometry): a grid
    that moves only sigma_p, sigma_d, length_d never rebuilds it."""
    key = (grid.digest(), m_max, float(length_r), float(length_theta), onp.asarray(geometry).tobytes())
    if key not in _CACHE:
        B = _basis_unit(grid, Q, m_max, np.asarray(length_r, dtype=float), np.asarray(length_theta, dtype=float), geometry)
        while _CACHE and (len(_CACHE) >= CACHE_MAX_ENTRIES or sum(v.nbytes for v in _CACHE.values()) + B.nbytes > CACHE_MAX_BYTES):
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = B
    return _CACHE[key]


# ----------------------------------------------------------------------------- surrogate prior
class SumPrior(_SurrogatePrior):
    """theta = (z_p, z_d, u), log10 I = level + mean + B_p z_p + Psi diag(amp) z_d. B_p (wid^2, n_p) dense (built once per
    polar hyperparameter point, sigma_p a plain scale of the memoised unit basis); the DCT part by fast transforms."""
    Bp: jax.Array
    amp: jax.Array
    mean: jax.Array
    level_mu: jax.Array
    level_sd: jax.Array
    sigma_p: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    geometry: jax.Array
    sigma_d: jax.Array
    length_d: jax.Array
    order_d: jax.Array
    nodes: jax.Array
    p_shape: tuple = eqx.field(static=True)
    m_max: int = eqx.field(static=True)          # -1: no cutoff
    grid_digest: str = eqx.field(static=True)

    @property
    def n_p(self):
        return self.Bp.shape[1]

    @property
    def n_latent(self):
        return self.n_p + self.amp.size

    @property
    def latent_shape(self):
        return (self.n_latent,)

    def split(self, z):
        return z[:self.n_p], z[self.n_p:]

    def field_p(self, z_p):
        w = self.mean.shape[0]
        return (self.Bp @ np.ravel(z_p)).reshape(w, w)

    def field_d(self, z_d):
        return gp_field(np.reshape(z_d, self.amp.shape), self.amp)

    def field(self, z):
        zp, zd = self.split(np.ravel(z))
        return self.field_p(zp) + self.field_d(zd)

    def field_T(self, v):
        w = self.amp.shape[0]
        return np.concatenate([self.Bp.T @ np.ravel(v), self.amp.ravel()*_dct2(np.reshape(v, (w, w))).ravel()])

    def congruence(self, H):
        """B^T H B, B = [B_p | X], X = Psi diag(amp): blocks B_p^T H B_p (dense), B_p^T H X (one DCT of H B_p) and
        X^T H X (two DCTs of H, as GPPrior.congruence)."""
        w = self.amp.shape[0]
        a = self.amp.ravel()
        C = H @ self.Bp                                              # (w^2, n_p)
        Spp = self.Bp.T @ C
        Xt_C = a[:, None]*_dct2(C.reshape(w, w, -1)).reshape(w*w, -1)   # X^T H B_p, (w^2, n_p)
        Sdd = _congruence_dct(H.reshape(w, w, w, w)).reshape(w*w, w*w)*a[:, None]*a[None, :]
        return np.block([[Spp, Xt_C.T], [Xt_C, Sdd]])

    def field_matrix(self, Wz):
        w = self.amp.shape[0]
        Wp, Wd = Wz[:self.n_p], Wz[self.n_p:]
        fd = jsp.fft.idctn(self.amp[:, :, None]*Wd.reshape(w, w, -1), type=2, norm="ortho", axes=(0, 1)).reshape(w*w, -1)
        return self.Bp @ Wp + fd

    def fit_field(self, f):
        """The prior-MAP latents of the log10 field f (a warm start): the minimum-norm z with B z ~ f under a ridge
        tau^2 = FIT_RIDGE mean(eig(B B^T)) (a unit ridge, tau^2 = 1, is far above B B^T's eigenvalues, which are in dex^2, and
        would return z ~ 0), z = B^T (B B^T + tau^2 I)^-1 f with B = [B_p | X], B B^T = B_p B_p^T + X X^T (X = Psi diag(amp)).
        The polar / DCT split is the one the prior prefers (smooth structure goes to whichever component explains it with
        the smaller latent norm), and |z| stays of the order of a prior draw's (not the huge near-exact solve of the DCT
        modes with tiny amp). Costs one (wid^2)^3 solve; call it eagerly."""
        f = np.ravel(f)
        w = self.amp.shape[0]
        X = jsp.fft.idctn(self.amp[:, :, None]*np.eye(w*w).reshape(w, w, w*w), type=2, norm="ortho",
                          axes=(0, 1)).reshape(w*w, w*w)
        K = self.Bp @ self.Bp.T + X @ X.T
        ridge = FIT_RIDGE*np.trace(K)/K.shape[0]
        y = np.linalg.solve(K + ridge*np.eye(K.shape[0]), f)
        return np.concatenate([self.Bp.T @ y, X.T @ y])

    def meta_arrays(self):
        return (self.sigma_p, self.length_r, self.length_theta, self.geometry, self.sigma_d, self.length_d, self.order_d,
                self.mean, self.level_mu, self.level_sd, onp.asarray([self.m_max]),
                onp.frombuffer(self.grid_digest.encode(), dtype=onp.uint8))


def sum_prior(grid, m_max, sigma_p, length_r, length_theta, cos_inc, pa, sigma_d, length_d, order_d=1., dx=0., dy=0.,
              mean=None, level_prior=(0., 1.)):
    """Surrogate prior of the polar (+ cutoff m_max) plus DCT fields at these hyperparameters."""
    w = grid.wid
    mean = np.zeros((w, w)) if mean is None else np.asarray(mean, dtype=float)
    mu, sd = (np.asarray(x, dtype=float) for x in level_prior)
    sigma_p, length_r, length_theta, sigma_d, length_d, order_d = (
        np.asarray(x, dtype=float) for x in (sigma_p, length_r, length_theta, sigma_d, length_d, order_d))
    geometry = _geometry(cos_inc, pa, dx, dy)
    Q = azimuthal_basis(grid.shape[1], m_max)
    Bp = sigma_p*_basis(grid, Q, m_max, length_r, length_theta, geometry)
    amp = matern_amplitude((w, w), np.broadcast_to(length_d, (2,)), sigma_d, order_d)
    return SumPrior(Bp, amp, mean, mu, sd, sigma_p, length_r, length_theta, geometry, sigma_d, length_d, order_d,
                    grid.nodes, (grid.shape[0], Q.shape[1]), -1 if m_max is None else int(m_max), grid.digest())


# ----------------------------------------------------------------------------- the fit
class PolarDCTResolvedFit(_GPFitBase):
    """Polar (+ optional m cutoff) plus DCT Matern prior on an InterpolatedResolvedSource. ONE parameter "resolved":
    the flat latents (z_p, z_d), length n_p + wid^2 (see the module docstring), and "resolved_level". Hyperparameters:
    sigma_p, length_r, length_theta, cos_inc, pa, dx, dy (as PolarGPResolvedFit) and sigma_d, length_d (grid px,
    isotropic), order_d (the DCT Matern order, fixed by the caller), mean, level_prior. get_distribution is the base's
    (it calls log10_field(z) with the flat z), so LinearResponse's checks pass."""
    HYPER = ("sigma_p", "length_r", "length_theta", "cos_inc", "pa", "dx", "dy", "sigma_d", "length_d", "order_d", "mean",
             "level_prior")
    grid: PolarGrid
    basis: jax.Array
    m_max: int = eqx.field(static=True)
    sigma_p: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    cos_inc: jax.Array
    pa: jax.Array
    dx: jax.Array
    dy: jax.Array
    sigma_d: jax.Array
    length_d: jax.Array
    order_d: jax.Array

    def __init__(self, spectrum_basis, filter, wid, grid, m_max=None, sigma_p=0.3, length_r=8., length_theta=2.,
                 cos_inc=1., pa=0., dx=0., dy=0., sigma_d=0.3, length_d=1.6, order_d=1., mean=None, level_prior=(0., 1.),
                 n_share=1, resolved_kwargs=None):
        assert grid.wid == wid, f"PolarGrid is for a {grid.wid} px disc, not {wid}"
        super().__init__(spectrum_basis, filter, wid, resolved_source="interp", resolved_kwargs=resolved_kwargs)
        self.grid = grid
        self.basis = azimuthal_basis(grid.shape[1], m_max)
        self.m_max = -1 if m_max is None else int(m_max)
        (self.sigma_p, self.length_r, self.length_theta, self.cos_inc, self.pa, self.dx, self.dy, self.sigma_d,
         self.length_d, self.order_d) = (np.asarray(x, dtype=float) for x in (
             sigma_p, length_r, length_theta, cos_inc, pa, dx, dy, sigma_d, length_d, order_d))
        self._init_gp(wid, mean, level_prior, n_share)

    @property
    def n_p(self):
        return self.grid.shape[0]*self.basis.shape[1]

    def geometry(self):
        return _geometry(self.cos_inc, self.pa, self.dx, self.dy)

    def log10_field(self, z):
        z = np.ravel(z)
        w = self.mean.shape[0]
        amp = matern_amplitude((w, w), np.broadcast_to(self.length_d, (2,)), self.sigma_d, self.order_d)
        return (polar_part(self.grid, self.basis, None if self.m_max < 0 else self.m_max, z[:self.n_p], self.sigma_p,
                           self.length_r, self.length_theta, self.geometry())
                + gp_field(z[self.n_p:].reshape(w, w), amp))

    def surrogate_prior(self):
        return sum_prior(self.grid, None if self.m_max < 0 else self.m_max, self.sigma_p, self.length_r,
                         self.length_theta, self.cos_inc, self.pa, self.sigma_d, self.length_d, self.order_d, self.dx,
                         self.dy, self.mean, self.level_prior)


def split_latents(z, grid, m_max, wid):
    """(z_p (n_r, k), z_d (wid, wid)) from the flat latents z."""
    k = grid.shape[1] if m_max is None else 2*m_max + 1
    n_p = grid.shape[0]*k
    return z[:n_p].reshape(grid.shape[0], k), z[n_p:].reshape(wid, wid)


# ----------------------------------------------------------------------------- diagnostics
def dct_view(prior, theta, laplace):
    """(GPPrior, theta, laplace) of the DCT component alone, for gp_spectral_diagnostic."""
    mu, V = laplace
    n_p = prior.n_p
    sub = GPPrior(prior.amp, prior.mean, prior.level_mu, prior.level_sd, prior.sigma_d,
                  np.broadcast_to(prior.length_d, (2,)), prior.order_d)
    return sub, np.concatenate([theta[n_p:-1], theta[-1:]]), (mu, np.concatenate([V[n_p:-1], V[-1:]]))


def sum_components(prior, theta, laplace):
    """Per-pixel split of the log10 field at the MAP and of its Laplace posterior: f_p, f_d (MAP components), std_p, std_d,
    std_total (posterior std of each and of the sum), and corr (posterior correlation of the two components per pixel:
    near -1 = they trade off, the identifiability warning), plus the rms of each and var_share = var_p / (var_p + var_d)
    of the MAP field."""
    mu, V = laplace
    w = prior.wid
    n_p = prior.n_p
    W = V/np.sqrt(1 + mu)
    Wp = prior.Bp @ W[:n_p]
    Wd = prior.field_matrix(W[:-1]) - Wp
    var_p, var_d = np.sum(Wp**2, 1), np.sum(Wd**2, 1)
    cov = np.sum(Wp*Wd, 1)
    z = theta[:-1]
    f_p, f_d = prior.field_p(z[:n_p]), prior.field_d(z[n_p:])
    return dict(f_p=f_p, f_d=f_d, std_p=np.sqrt(var_p).reshape(w, w), std_d=np.sqrt(var_d).reshape(w, w),
                std_total=np.sqrt(var_p + var_d + 2*cov).reshape(w, w),
                corr=(cov/np.sqrt(var_p*var_d + 1e-300)).reshape(w, w),
                rms_p=np.sqrt(np.mean(f_p**2)), rms_d=np.sqrt(np.mean(f_d**2)),
                var_share=np.mean(f_p**2)/(np.mean(f_p**2) + np.mean(f_d**2)))


def polar_orders_diagnostic(prior, theta, laplace):
    """Posterior power of the polar latents per azimuthal order m (reduced Fourier latents: the two columns of each m
    >= 1), <z^2> = z_map^2 + posterior variance (prior 1), and the mean posterior variance. Only for m_max not None."""
    assert prior.m_max >= 0, "reduced polar latents only (m_max not None)"
    mu, V = laplace
    n_r, k = prior.p_shape
    n_p = prior.n_p
    var = np.sum(V[:n_p]**2/(1 + mu), 1).reshape(n_r, k)
    power = theta[:n_p].reshape(n_r, k)**2 + var
    cols = [[0]] + [[2*m - 1, 2*m] for m in range(1, prior.m_max + 1)]
    return dict(m=onp.arange(prior.m_max + 1),
                power_m=onp.array([float(power[:, c].mean()) for c in cols]),
                var_m=onp.array([float(var[:, c].mean()) for c in cols]),
                count_m=onp.array([n_r*len(c) for c in cols]))
