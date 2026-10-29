"""Occulter-glint components for the NICMOS coronagraph models (PROPOSED NEW repo file: occulters.py, repo root).

Consolidates the v2 drafts (soft_edge, glint_common, climb_glint, image_occulter) into one module, without the
fallback / shim layers. Contents:
  soft_coverage, soft_disc            continuous area-fraction edge (replaces dlu.soft_binarise where selected)
  rim helpers                         rim_basis, rim_transmission, rim_terms_linear, rim_mask, rim_mask_terms,
                                      harmonics, centring_m1, combine_terms
  CLIMBGlintOcculter                  star-shaped edge (m >= 3 Fourier edge, area-centroid centred via the computed m = 1
                                      term) + one-sided rim (NOT centred: its first moment is absorbed by primary_tilt)
  ImageOcculter                       hole outline from the NICMOS handbook image (SDF in data/occulter/hole_sdf.npz)
  CLIMB_PARAMS, IMAGE_PARAMS          the OPTICS_PARAMS entries (also listed in the proposed models.py)
  glint_log_prior, edge_profile       priors (no existing prior hook in models.py: see integration NOTES.md)

Conventions: obstruction mask (1 on the hole side, d < 0), d > 0 on the mirror; rim terms live on the mirror side and
are gated by a smooth sigmoid; azimuthal harmonics m in {0, 2, 3, ...} (m = 1 is computed by centring_m1, never free).
float64 assumed (the repo enables jax_enable_x64).

This module Rim amplitude guard: rim_amp is a LOG-amplitude; the total log-amplitude field A(d, phi) is passed through
`logamp_guard` (identity for A <= 0 up to 1e-4, smooth saturation at ~0 for A > 0) so |T| <= 1 + 5e-5 and Re(mask) >= -5e-5
(no gain from a passive rim); glint_log_prior additionally carries a hinge penalty on A > 0 so the optimiser is not
left on the dead (zero-gradient) side. Zero coefficients give exactly T = 1.

Fixed (non-trainable) leaves: FIXED_LEAVES = rim_scales, wavelength0, sdf (and ghost_offset/width, spot_width). The sdf table
is stop_gradient'ed inside the occulter (exactly zero gradient), but optimisers must still select parameters by the
OPTICS_PARAMS paths (CLIMB_PARAMS / IMAGE_PARAMS), never by iterating over all float leaves (it has 66k entries).

This module must NOT import apertures.py (apertures.py imports it). ChromaticFresnelObstruction therefore lives in
the proposed apertures.py, next to FresnelFastObstruction which it subclasses.
"""
import dataclasses
import os

import dLux as dl
import dLux.utils as dlu
import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import Array

TAU_DEFAULT = 0.5e-6      # rim gate smoothing length (m), ~ pixel/3 at the 1.5 um occulter-plane pixel
LAMBDA_F_REF = 1.6e-6 * 24.          # lambda F# at 1.6 um, F# = 24 (m)
HOLE_SDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "occulter", "hole_sdf.npz")


# ====================================================================================================================
# Small helpers
# ====================================================================================================================
def safe_sqrt(x):
    """sqrt(x) with value 0 and gradient 0 (not inf/NaN) where x <= 0 (e.g. rho at the origin). Double-where."""
    x = np.asarray(x)
    pos = x > 0
    return np.where(pos, np.sqrt(np.where(pos, x, 1.)), 0.)


def safe_hypot(x, y):
    """sqrt(x^2 + y^2) with zero (finite) gradient at the origin."""
    return safe_sqrt(x**2 + y**2)


def pool(x, oversample=3):
    """Mean-pool an (n*os, n*os) fine-grid array to (n, n)."""
    n = x.shape[-1]//oversample
    return x.reshape(n, oversample, n, oversample).mean((1, 3))


# ====================================================================================================================
# Soft coverage edge (from soft_edge.py)
# ====================================================================================================================
_G2_MIN = 1e-20     # |g|^2 below this: treat the gradient as zero (d is then used only through its sign)
_FAR = 1e3          # "far from the edge" distance in pixels used when |g| ~ 0


def soft_coverage(d, gx, gy, pixel_scale, eps_axis=1e-3):
    """Area fraction of {d < 0} in each square pixel, using d/|g| as signed distance and g/|g| as edge normal.

    d : signed function sampled at the pixel centres (negative inside the region); gx, gy its spatial gradient on the
        same grid (same units as d per unit of pixel_scale); pixel_scale: pixel side. Any array shape (elementwise).
    The pixel is cut by a straight line at signed distance s = d/|g| from its centre with normal n = g/|g|.
    The covered fraction is F(-s/ps), F the CDF of the pixel's projection on n: a trapezoid (sum of uniforms of widths
    a = max(|nx|,|ny|), b = min(...)) with outer half-width h1 = (a+b)/2, inner h2 = (a-b)/2, piecewise quadratic, C1:
        t = -s/ps ;  F = (t+h1)^2/(2ab)      for -h1 < t < -h2
                     = 1/2 + t/a              for |t| <= h2
                     = 1 - (h1-t)^2/(2ab)     for  h2 < t < h1 ;  0 / 1 outside.
    Axis-aligned normals (b -> 0) are regularised smoothly: b -> hypot(b, eps_axis) (edge blur ~ eps_axis px at worst,
    ~1e-3 px, error O(eps_axis) in coverage only for pixels within eps of the edge).
    |g| ~ 0 (below 1e-10): coverage is the step on sign(d) (0.5 at d == 0) with zero gradient.
    Gradients are finite everywhere (double-where on the safe norm, clip on the far field)."""
    d, gx, gy = np.asarray(d), np.asarray(gx), np.asarray(gy)
    gn2 = gx * gx + gy * gy
    ok = gn2 > _G2_MIN
    gxs, gys = np.where(ok, gx, 1.0), np.where(ok, gy, 0.0)            # safe gradient (double-where)
    gn = np.sqrt(gxs * gxs + gys * gys)
    a0, b0 = np.abs(gxs) / gn, np.abs(gys) / gn
    a = np.maximum(a0, b0)
    b = np.hypot(np.minimum(a0, b0), eps_axis)                         # at exactly 45 deg b = hypot(0.7071, eps) > a, so h2 < 0
    # (~ -3.5e-7): the branch order tc < -h2 / tc > h2 still returns a continuous value (measured 45 deg error 6e-8).
    t = np.where(ok, -d / (gn * pixel_scale), -np.sign(d) * _FAR)      # inside distance in pixels
    h1, h2 = 0.5 * (a + b), 0.5 * (a - b)
    tc = np.clip(t, -h1, h1)                                            # far field: zero gradient
    lo = (tc + h1) ** 2 / (2 * a * b)
    mid = 0.5 + tc / a
    hi = 1 - (h1 - tc) ** 2 / (2 * a * b)
    return np.where(tc < -h2, lo, np.where(tc > h2, hi, mid))


def soft_disc(d_fn, npixels, pixel_scale, oversample=3):
    """Mean-pooled coverage of {d_fn(x, y) < 0} on an npixels x npixels grid (pixel_scale), evaluated at
    `oversample` x resolution. d_fn(xx, yy) -> d must be ELEMENTWISE (d[i,j] depends only on xx[i,j], yy[i,j]; true for
    any analytic function); the gradient is then obtained exactly with two forward-mode jvps (primal computed once), so
    no hand-derived gradient is needed. Non-elementwise d (e.g. interpolated SDF tables) should call soft_coverage
    directly with its own gradient. Grid convention = dlu.pixel_coords (centred, x along axis 1)."""
    n, ps = npixels * oversample, pixel_scale / oversample
    c = (np.arange(n) - (n - 1) / 2) * ps
    xx, yy = np.meshgrid(c, c)
    d = d_fn(xx, yy)
    z, o = np.zeros_like(xx), np.ones_like(xx)
    g = jax.vmap(lambda tx, ty: jax.jvp(d_fn, (xx, yy), (tx, ty))[1])(np.stack([o, z]), np.stack([z, o]))
    cov = soft_coverage(d, g[0], g[1], ps)
    return cov.reshape(npixels, oversample, npixels, oversample).mean((1, 3))


# ====================================================================================================================
# Harmonics, smooth one-sided radial basis, rim masks, centring
# ====================================================================================================================
def harmonic_orders(m_max, skip=(1,)):
    """Orders m in 1..m_max with those in `skip` removed (m = 0 is the constant, always present)."""
    return [m for m in range(1, m_max + 1) if m not in skip]


def m_max_from_n(n_harm, skip=(1,)):
    """Inverse of len(harmonics) = 1 + 2 len(orders): the m_max that gives n_harm harmonics for the given `skip`
    (the n_orders-th integer >= 1 not in skip). 1 harmonic -> 0."""
    if n_harm < 1 or (n_harm - 1) % 2:
        raise ValueError(f"n_harm must be odd and >= 1, got {n_harm}")
    n_orders, m, count = (n_harm - 1)//2, 0, 0
    while count < n_orders:
        m += 1
        count += m not in skip
    return m


def harmonics(phi, m_max, skip=(1,)):
    """Real stack [1, cos 2phi, sin 2phi, cos 3phi, sin 3phi, ...] of shape (1 + 2 len(ms), *phi.shape); m in `skip`
    (default m = 1) is excluded."""
    phi = np.asarray(phi)
    out = [np.ones_like(phi)]
    for m in harmonic_orders(m_max, skip):
        out += [np.cos(m*phi), np.sin(m*phi)]
    return np.stack(out)


def harmonic_m_list(n_harm, skip=(1,)):
    """Azimuthal order m of each of the n_harm harmonic rows: [0, 2, 2, 3, 3, ...]."""
    ms = [0]
    for m in harmonic_orders(m_max_from_n(n_harm, skip), skip):
        ms += [m, m]
    return ms


def rim_basis(d, phi, scales, m_max, tau=TAU_DEFAULT):
    """Smooth one-sided radial x azimuthal basis, shape (K, H, *d.shape): B_k(d) H_h(phi) with the smooth gate
        B_k = sigmoid(d/tau) exp(-softplus(d/tau) tau / s_k)
    (-> exp(-d/s_k) for d >> tau, -> 0 like exp(d/tau) for d << -tau). Differentiable everywhere with finite gradients
    (this fixes the review's B1: the old hard step made AD gradients w.r.t. edge-moving parameters wrong). B is NOT
    exactly 0 for d <= 0 (1e-3 at d = -7 tau), so T != 1 just inside the edge: the product with (1 - disc) in
    `rim_mask` removes it on the hole side. tau = 0 is not allowed."""
    d = np.asarray(d)
    s = np.asarray(scales).reshape((-1,) + (1,)*d.ndim)
    u = d/tau
    B = jax.nn.sigmoid(u)[None]*np.exp(-jax.nn.softplus(u)[None]*tau/s)
    return B[:, None]*harmonics(phi, m_max)[None]


_GUARD_EPS = 1e-4


def logamp_guard(A, eps=_GUARD_EPS):
    """Smooth one-sided guard on a log-amplitude field: g(A) = (A - sqrt(A^2 + eps^2) + eps)/2. g(0) = 0 exactly (zero
    coefficients stay T = 1), g(A) = A + O(eps) for A < -eps (slope 1 away from 0, 1/2 at 0), and 0 <= g <= eps/2
    for A > 0, so exp(g) <= 1 + eps/2: a passive rim cannot amplify. C-infinity. Positive A is therefore dead (zero
    gradient): glint_log_prior penalises it with a hinge instead."""
    return 0.5*(A - np.sqrt(A**2 + eps**2) + eps)


def _contract(coeffs, basis):
    """sum_{k,h} coeffs[k, h] basis[k, h, ...] (tensordot, not a broadcasted sum: see jax 0.10.2 float32 miscompile)."""
    return np.tensordot(np.asarray(coeffs), basis, axes=([0, 1], [0, 1]))


def rim_transmission(d, phi, rim_amp, rim_opd, scales, wavelength, tau=TAU_DEFAULT):
    """Complex rim transmission T = exp(sum_{k,h} (amp_kh + 2 pi i opd_kh / lambda) B_k(d) H_h(phi)); amp is a
    dimensionless log-amplitude (achromatic), opd in metres (phase 2 pi opd / lambda, chromatic). T = 1 when all
    coefficients are 0; with the smooth gate T != 1 slightly inside the edge (d < 0). The log-amplitude field goes through
    `logamp_guard`, so |T| <= 1 + 5e-5 for any coefficients (positive log-amplitude is not representable)."""
    rim_amp = np.asarray(rim_amp)
    basis = rim_basis(d, phi, scales, m_max_from_n(rim_amp.shape[1]), tau)
    return np.exp(logamp_guard(_contract(rim_amp, basis)) + 2j*np.pi*_contract(rim_opd, basis)/wavelength)


def rim_terms_linear(d, phi, rim_amp, rim_opd, scales, wavelength0, tau=TAU_DEFAULT):
    """(T0, T1) with T(lambda) ~= T0 + (lambda0/lambda - 1) T1: T0 = T(lambda0), T1 = i phi0 T0 with
    phi0 = 2 pi OPD(d, phi) / lambda0 (exact first-order term; amplitude is achromatic). Truncation error ~ (phi0 s)^2 / 2 per
    unit rim transmission; measured mask error (review): ~5.6e-3 at 1.2 um and 3.8e-3 at 2.2 um for amp 0.03 + 30 nm
    summed over all 3 scales on m = 0 (effective phase 0.35 rad at 1.6 um); ~1e-16 at lambda0. Use mask(wavelength) (exact)
    when that accuracy matters."""
    rim_amp = np.asarray(rim_amp)
    basis = rim_basis(d, phi, scales, m_max_from_n(rim_amp.shape[1]), tau)
    phi0 = 2*np.pi*_contract(rim_opd, basis)/wavelength0
    T0 = np.exp(logamp_guard(_contract(rim_amp, basis)) + 1j*phi0)
    return T0, 1j*phi0*T0


def rim_mask(disc3, T3, oversample=3):
    """Obstruction mask with the rim, built on the FINE grid before pooling (fixes the v1 pooling bug that halved the
    rim on edge pixels): mask = pool(disc3) + pool((1 - disc3)(1 - T3)).
    disc3 : (n*os, n*os) coverage of the hole side (1 inside, 0 on the mirror), on the fine grid;
    T3    : complex rim transmission on the same fine grid."""
    return pool(disc3, oversample) + pool((1 - disc3)*(1 - T3), oversample)


def rim_mask_terms(disc3, T0_3, T1_3, oversample=3):
    """(M0, M1) with mask(lambda) = M0 + (lambda0/lambda - 1) M1 for T = T0 + s T1: M0 = rim_mask(disc3, T0_3),
    M1 = -pool((1 - disc3) T1_3). Both on the pooled grid; build once per propagation."""
    return rim_mask(disc3, T0_3, oversample), -pool((1 - disc3)*T1_3, oversample)


def centring_m1(r_cos, r_sin, ms, n_phi=512, n_iter=8):
    """Computed m = 1 coefficients (a1, b1) such that the area centroid of the region
        r(phi) = 1 + a1 cos(phi) + b1 sin(phi) + sum_{m in ms} (a_m cos(m phi) + b_m sin(m phi))
    is exactly at the origin. The centroid is proportional to int r^3 (cos phi, sin phi) dphi, which is a trig
    polynomial of finite degree 3 max(1, m_max), so a uniform phi grid with n_phi > 3 m_max + 1 points integrates it
    exactly. (a1, b1) solves F(a1, b1) = 0 by `n_iter` fixed (unrolled, hence differentiable) Newton steps from 0
    with the analytic 2x2 Jacobian. Fixed iteration count: no data-dependent control flow, jit/grad safe.

    r_cos, r_sin : (len(ms),) cos / sin coefficients of the orders `ms` (m = 1 is rejected: it is the unknown). The
    condition is scale invariant (multiply by the caller's r0 afterwards). Returns (a1, b1) scalars.
    A linear map of a region with zero area centroid has zero area centroid, so this is valid in the elliptical
    frame where the harmonics are defined."""
    ms_np = onp.atleast_1d(onp.asarray(ms))
    if onp.any(ms_np == 1):
        raise ValueError("centring_m1: ms must not contain m = 1 (it is the computed term)")
    r_cos, r_sin = np.atleast_1d(np.asarray(r_cos, float)), np.atleast_1d(np.asarray(r_sin, float))
    ms = np.asarray(ms_np)
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    c, s = np.cos(phi), np.sin(phi)
    base = 1 + np.tensordot(r_cos, np.cos(ms[:, None]*phi[None]), 1) + np.tensordot(r_sin, np.sin(ms[:, None]*phi[None]), 1)

    def resid_jac(x):
        r = base + x[0]*c + x[1]*s
        F = np.stack([np.mean(r**3*c), np.mean(r**3*s)])
        J = 3*np.stack([np.stack([np.mean(r**2*c*c), np.mean(r**2*c*s)]),
                        np.stack([np.mean(r**2*c*s), np.mean(r**2*s*s)])])
        return F, J

    x = np.zeros(2, dtype=base.dtype)
    for _ in range(n_iter):
        F, J = resid_jac(x)
        x = x - np.linalg.solve(J, F)
    return x[0], x[1]


def combine_terms(terms, wavelength):
    """M0 + (lam0/lam - 1) M1 from an occulter's `mask_terms()` output (M0, M1, lam0); wavelength None -> M0."""
    M0, M1, lam0 = terms
    return M0 if wavelength is None else M0 + (lam0/wavelength - 1)*M1


# ====================================================================================================================
# CLIMBGlintOcculter
# ====================================================================================================================
EDGE_M0 = 3                          # first free edge order; m = 2 is the ellipticity e, m = 1 is computed
F_MIN = 0.05                         # floor on f(phi) inside the latent / distance (see class docstring)
_SQ2 = np.sqrt(2.)


def _m_max_from_H(H):
    """Harmonic stack [1, cos 2, sin 2, ..., cos M, sin M] has H = 1 + 2 (M - 1) rows."""
    return (H - 1)//2 + 1


def _ghost_gate(dn):
    """Smooth hole-side gate sigmoid(-dn/TAU_DEFAULT) of the ghost ring (replaces the hard step dn < 0)."""
    return jax.nn.sigmoid(-dn/TAU_DEFAULT)


def _pool(a, npixels):
    return a.reshape(npixels, 3, npixels, 3).mean((1, 3))


class CLIMBGlintOcculter(dl.OpticalLayer):
    """
    r, e            : as EllipticalOcculter (edge at rho = sqrt(2) r, area-preserving ellipticity).
    edge_coeffs     : (M-2, 2), (cos, sin) coefficients of R(phi)/(sqrt(2) r) for m = 3..M (odd and even m). m = 1 is
                      computed (centring); m = 2 is NOT a parameter (it is e). m_edge = M, default 5.
    edge            : static, "soft_binarise" (DEFAULT, same edge as EllipticalOcculter) or "coverage" (option).
                      "coverage" (option): the fine (3x) grid holds soft_coverage(d, gx, gy, pixel_scale/3) with the signed function
                      d = rho - R(phi) (circularised frame, metres, < 0 in the hole) and its ANALYTIC gradient in the
                      occulter frame, g = Q (e_r - (R'/rho) e_phi); the mask pools that 3x3. C1 in the parameters, exact
                      area fraction for a straight edge.
                      "soft_binarise" (default): dlu.soft_binarise of the latent 0.5 (1 - (rho/R)^2); exactly equal to
                      EllipticalOcculter at zero glint (to 1e-12) but with the plane-fit sub-pixel noise in its gradients.
                      The rim product in this mode still uses a fine-grid coverage (the same soft_coverage) for (1 - disc3).
    rim_amp, rim_opd: (K, H) log-amplitude and OPD (m) of the one-sided rim on the mirror side (d > 0); H = 1 + 2 (M_rim - 1)
                      harmonics (m = 0, 2, 3, ..., M_rim; the m = 1 rows do not exist). rim_scales (K,) m, fixed.
                      The rim uses dn = d/|grad d| (first-order true normal distance), and is formed on the fine grid
                      BEFORE pooling (glint_common.rim_mask).
    ghost           : (Hg, 2) real/imag complex amplitude of a ghost ring just inside the edge (d < 0), harmonics
                      m = 0, 2, ..., M_ghost; Gaussian radial profile centred at dn = -ghost_offset, width ghost_width (fixed), times the smooth gate
                      sigmoid(-dn/TAU_DEFAULT) (hole side; not a hard step, so gradients w.r.t. r, e are exact).
    spot_pos, spot_amp : (N, 2) positions (m, occulter x,y) and real/imag complex amplitudes of N Gaussian dust/defect spots
                      (width spot_width, fixed). mask = ... - sum amp exp(-|x - pos|^2 / 2 w^2).
    legacy_glint, legacy_width_raw : optional EllipticalOcculter-style ring (off by default; has no rows). legacy_glint
                      (Hl, 2) raw real/imag for m = 2..M_l (cos, sin pairs: Hl must be EVEN, no m = 0/1), g = 0.3 tanh(raw);
                      width w = 1.5 um + 1 um softplus(legacy_width_raw), profile exp(-d^2/2w^2) in the raw d. Azimuth is the
                      CIRCULARISED phi (same as the edge/rim; equals EllipticalOcculter's theta only for e = 0). DEGENERATE
                      with edge_coeffs / r / e (a real ring on the edge is an edge displacement): only for continuity.
    wavelength0     : reference wavelength (m); mask(wavelength=None) uses it.
    f floor         : f(phi) is floored at F_MIN = 0.05 inside the latent / d (a numerical guard so rho/R and R' stay finite).
                      It silently hides non-star-shaped edges (min f < 0.05), for which the shape is meaningless; use
                      `edge_validity()` (min f) and `edge_log_prior()` (a smooth penalty when min f < fmin) in the fit so
                      the optimiser is pushed away from that region instead of the guard hiding it.
    """
    r: Array
    e: Array
    edge_coeffs: Array
    rim_amp: Array
    rim_opd: Array
    rim_scales: Array
    ghost: Array
    ghost_offset: Array
    ghost_width: Array
    spot_pos: Array
    spot_amp: Array
    spot_width: Array
    legacy_glint: Array
    legacy_width_raw: Array
    wavelength0: Array

    normalise: bool
    invert: bool
    edge: str = eqx.field(static=True)

    def __init__(self, r, e=np.zeros(2), edge_coeffs=None, m_edge=5, m_rim=3, rim_scales=None, rim_amp=None, rim_opd=None,
                 m_ghost=2, ghost=None, ghost_offset=None, ghost_width=None, n_spots=0, spot_pos=None, spot_amp=None,
                 spot_width=3e-6, legacy_glint=None, legacy_width_raw=0., wavelength0=1.6e-6, normalise=False,
                 invert=False, edge="soft_binarise"):
        if edge not in ("coverage", "soft_binarise"):
            raise ValueError(f"edge must be 'coverage' or 'soft_binarise', got {edge!r}")
        f = lambda x: np.asarray(x, float)
        self.r = f(r)
        self.e = f(e)
        self.edge_coeffs = np.zeros((max(m_edge - EDGE_M0 + 1, 0), 2)) if edge_coeffs is None else f(edge_coeffs)
        self.rim_scales = f(np.array([0.25, 0.5, 1.]) * LAMBDA_F_REF if rim_scales is None else rim_scales)
        H = 1 + 2*(m_rim - 1)
        K = self.rim_scales.shape[0]
        self.rim_amp = np.zeros((K, H)) if rim_amp is None else f(rim_amp)
        self.rim_opd = np.zeros((K, H)) if rim_opd is None else f(rim_opd)
        self.ghost = np.zeros((1 + 2*(m_ghost - 1), 2)) if ghost is None else f(ghost)
        self.ghost_offset = f(0.5*LAMBDA_F_REF if ghost_offset is None else ghost_offset)
        self.ghost_width = f(0.25*LAMBDA_F_REF if ghost_width is None else ghost_width)
        self.spot_pos = np.zeros((n_spots, 2)) if spot_pos is None else f(spot_pos)
        self.spot_amp = np.zeros((n_spots, 2)) if spot_amp is None else f(spot_amp)
        self.spot_width = f(spot_width)
        self.legacy_glint = np.zeros((0, 2)) if legacy_glint is None else f(legacy_glint)
        assert self.legacy_glint.shape[0] % 2 == 0, "legacy_glint needs an even number of rows (cos, sin pairs, m = 2..M)"
        self.legacy_width_raw = f(legacy_width_raw)
        self.wavelength0 = f(wavelength0)
        self.normalise = normalise
        self.invert = invert
        self.edge = edge

    # ------------------------------------------------------------------ geometry
    def _rho2(self, xx, yy):
        """Squared elliptical radius, identical expression to EllipticalOcculter._rho2."""
        e1, e2 = self.e
        return ((1 - e1)*xx**2 - 2*e2*xx*yy + (1 + e1)*yy**2) / np.sqrt(1 - e1**2 - e2**2)

    def _Q(self):
        """Q = (A + I)/sqrt(tr A + 2) (entries q11, q12, q22): X = Q x, X^2 + Y^2 = rho^2."""
        e1, e2 = self.e
        s = 1/np.sqrt(1 - e1**2 - e2**2)
        a11, a12, a22 = (1 - e1)*s, -e2*s, (1 + e1)*s
        n = 1/np.sqrt(a11 + a22 + 2)
        return (a11 + 1)*n, a12*n, (a22 + 1)*n

    def _circularise(self, xx, yy):
        q11, q12, q22 = self._Q()
        return q11*xx + q12*yy, q12*xx + q22*yy

    def _azimuth(self, xx, yy):
        """Circularised azimuth, with the origin made safe for gradients (value there is arbitrary)."""
        X, Y = self._circularise(xx, yy)
        ok = (X**2 + Y**2) > 0
        return np.arctan2(np.where(ok, Y, 0.), np.where(ok, X, 1.))

    @property
    def edge_ms(self):
        return onp.arange(EDGE_M0, EDGE_M0 + self.edge_coeffs.shape[0])   # static (numpy): must not be traced under jit

    def edge_m1(self):
        """(a1, b1) that put the area centroid of the hole at the origin."""
        return centring_m1(self.edge_coeffs[:, 0], self.edge_coeffs[:, 1], self.edge_ms)

    def _profile(self, phi, a1b1=None):
        """(f, f') with f = R/(sqrt(2) r) incl. the computed m = 1 term, and f' = df/dphi."""
        a1, b1 = self.edge_m1() if a1b1 is None else a1b1
        f = 1. + a1*np.cos(phi) + b1*np.sin(phi)
        fp = -a1*np.sin(phi) + b1*np.cos(phi)
        for i, m in enumerate(range(EDGE_M0, EDGE_M0 + self.edge_coeffs.shape[0])):
            c, s = self.edge_coeffs[i, 0], self.edge_coeffs[i, 1]
            f = f + c*np.cos(m*phi) + s*np.sin(m*phi)
            fp = fp + m*(-c*np.sin(m*phi) + s*np.cos(m*phi))
        return f, fp

    def edge_profile(self, phi):
        """f(phi) = R(phi) / (sqrt(2) r), including the computed m = 1 term (no floor)."""
        return self._profile(phi)[0]

    def edge_validity(self, n_phi=256):
        """min over phi of f(phi): the edge is star-shaped about the origin iff this is > 0 (the latent floors f at F_MIN)."""
        return np.min(self.edge_profile(np.linspace(0, 2*np.pi, n_phi, endpoint=False)))

    def edge_log_prior(self, fmin=0.3, sigma=0.05, n_phi=256):
        """Soft warning for use in a log-prior: -0.5 (relu(fmin - min f)/sigma)^2 (exactly 0 for any sane edge, C1)."""
        f = self.edge_profile(np.linspace(0, 2*np.pi, n_phi, endpoint=False))
        return -0.5*(np.maximum(fmin - np.min(f), 0.)/sigma)**2

    def _geometry(self, xx, yy):
        """dict: phi, rho (safe sqrt), d = rho - R(phi) (m), analytic physical gradient (gx, gy) of d, dn = d/|grad d|."""
        rho2 = self._rho2(xx, yy)
        rho = np.sqrt(np.where(rho2 > 1e-24, rho2, 1e-24))          # safe sqrt (origin / odd grids): finite gradients
        phi = self._azimuth(xx, yy)
        f, fp = self._profile(phi)
        ok = f > F_MIN
        f, fp = np.where(ok, f, F_MIN), np.where(ok, fp, 0.)
        R = _SQ2*self.r*f
        d = rho - R
        # grad_X d = e_r - (R'/rho) e_phi in the circularised frame; physical gradient = Q grad_X d (Q symmetric).
        # R'/rho is only clamped within 2% of R of the origin, where d is very large and the value is irrelevant.
        rho_g = np.where(rho > 0.02*R, rho, 0.02*R)
        k = _SQ2*self.r*fp/rho_g
        c, s = np.cos(phi), np.sin(phi)
        gX, gY = c + k*s, s - k*c
        q11, q12, q22 = self._Q()
        gx, gy = q11*gX + q12*gY, q12*gX + q22*gY
        dn = d/np.sqrt(gx**2 + gy**2)
        return dict(phi=phi, rho=rho, f=f, d=d, gx=gx, gy=gy, dn=dn)

    def signed_distance(self, xx, yy):
        """(dn, phi): dn = d/|grad d| (m, > 0 on the mirror; true normal distance to first order), phi the circularised
        azimuth used by the harmonics. Zero contour and sign identical to d = rho - R(phi)."""
        g = self._geometry(xx, yy)
        return g['dn'], g['phi']

    # ------------------------------------------------------------------ mask pieces
    def _grid(self, npixels, pixel_scale):
        return dlu.pixel_coords(npixels*3, pixel_scale=pixel_scale/3)

    def _disc3(self, geo, pixel_scale):
        """Fine-grid coverage of the hole."""
        return soft_coverage(geo['d'], geo['gx'], geo['gy'], pixel_scale/3)

    def _disc_sb(self, xx, yy, geo):
        """v1 CLIMB disc (pooled): latent 0.5 (1 - (rho/R)^2), zero contour exactly rho = R(phi); equal to
        EllipticalOcculter's latent for f = 1."""
        rho2 = self._rho2(xx, yy)
        f = geo['f']                                         # already floored at F_MIN
        return dlu.soft_binarise(-rho2/(2*self.r*f)**2 + 0.5, oversample=3)

    def _others(self, xx, yy, geo, npixels):
        """Everything subtracted from the obstruction (ghost, spots, legacy ring), pooled to the base grid."""
        out = 0.
        phi = geo['phi']
        if self.ghost.shape[0] > 0:
            H = harmonics(phi, _m_max_from_H(self.ghost.shape[0]))
            g = np.tensordot(self.ghost[:, 0] + 1j*self.ghost[:, 1], H, 1)
            dn = geo['dn']
            gate = _ghost_gate(dn)                            # smooth hole-side gate (like the rim's): AD gradient w.r.t. r, e is correct
            prof = gate*np.exp(-(dn + self.ghost_offset)**2/(2*self.ghost_width**2))
            out = out + _pool(g*prof, npixels)
        if self.spot_pos.shape[0] > 0:
            r2 = (xx[None] - self.spot_pos[:, 0, None, None])**2 + (yy[None] - self.spot_pos[:, 1, None, None])**2
            blobs = np.exp(-r2/(2*self.spot_width**2))
            out = out + _pool(np.tensordot(self.spot_amp[:, 0] + 1j*self.spot_amp[:, 1], blobs, 1), npixels)
        if self.legacy_glint.shape[0] > 0:
            raw = self.legacy_glint
            H = harmonics(phi, 1 + raw.shape[0]//2)[1:]       # no m = 0, no m = 1: [cos 2, sin 2, ...]
            g = np.tensordot(0.3*np.tanh(raw[:, 0]) + 0.3j*np.tanh(raw[:, 1]), H, 1)
            w = 1.5e-6 + 1e-6*jax.nn.softplus(self.legacy_width_raw)
            out = out + _pool(g*np.exp(-geo['d']**2/(2*w**2)), npixels)
        return out

    def _setup(self, npixels, pixel_scale):
        xx, yy = self._grid(npixels, pixel_scale)
        geo = self._geometry(xx, yy)
        disc3 = self._disc3(geo, pixel_scale)
        disc = _pool(disc3, npixels) if self.edge == "coverage" else self._disc_sb(xx, yy, geo)
        return xx, yy, geo, disc3, disc

    def _finish(self, m):
        return 1 - m if self.invert else m

    def mask(self, npixels, pixel_scale, wavelength=None):
        """Complex obstruction mask (1 on the disc) at `wavelength` (default wavelength0)."""
        wavelength = self.wavelength0 if wavelength is None else wavelength
        xx, yy, geo, disc3, disc = self._setup(npixels, pixel_scale)
        T3 = rim_transmission(geo['dn'], geo['phi'], self.rim_amp, self.rim_opd, self.rim_scales, wavelength)
        rim = rim_mask(disc3, T3, 3) - _pool(disc3, npixels)       # pool((1 - disc3)(1 - T3)); product on the fine grid
        return self._finish(disc + rim - self._others(xx, yy, geo, npixels))

    def mask_terms(self, npixels, pixel_scale):
        """(M0, M1, wavelength0) with mask(lambda) ~= M0 + (lambda0/lambda - 1) M1 (rim linearised; rest exact)."""
        xx, yy, geo, disc3, disc = self._setup(npixels, pixel_scale)
        T0, T1 = rim_terms_linear(geo['dn'], geo['phi'], self.rim_amp, self.rim_opd, self.rim_scales, self.wavelength0)
        M0, M1 = rim_mask_terms(disc3, T0, T1)
        M0 = M0 - _pool(disc3, npixels) + disc - self._others(xx, yy, geo, npixels)
        if self.invert:
            M0, M1 = 1 - M0, -M1
        return M0, M1, self.wavelength0

    def __call__(self, wavefront):
        wf = wavefront * self.mask(wavefront.npixels, wavefront.pixel_scale, wavefront.wavelength)
        if self.normalise:
            return wf.normalise()
        return wf


# ====================================================================================================================
# ImageOcculter
# ====================================================================================================================
def load_hole_sdf(path=HOLE_SDF_PATH):
    z = onp.load(path, allow_pickle=True)
    return z["sdf"], float(z["extent"][0])


def _bilinear(sdf, ux, uy, half, grad="nodal"):
    """Bilinear lookup of a (N, N) field [iy, ix] on [-half, half]^2 with a gradient (d/dux, d/duy).
    grad="cell": exact gradient of the bilinear interpolant (piecewise constant, jumps across cells by ~1-2 deg in
    direction); grad="nodal": central-difference node gradients, bilinearly interpolated (continuous; agrees with the
    cell gradient to O(h^2)).  Outside the grid: border value + Euclidean excess (all 'mirror')."""
    N = sdf.shape[0]; k = (N - 1)/(2*half)
    cx, cy = np.clip(ux, -half, half), np.clip(uy, -half, half)
    fx, fy = (cx + half)*k, (cy + half)*k
    i = np.clip(np.floor(fx), 0, N - 2).astype(int); j = np.clip(np.floor(fy), 0, N - 2).astype(int)
    tx, ty = fx - i, fy - j
    s00, s10, s01, s11 = sdf[j, i], sdf[j, i + 1], sdf[j + 1, i], sdf[j + 1, i + 1]
    s = (1 - tx)*(1 - ty)*s00 + tx*(1 - ty)*s10 + (1 - tx)*ty*s01 + tx*ty*s11
    if grad == "cell":
        gx = k*((1 - ty)*(s10 - s00) + ty*(s11 - s01))
        gy = k*((1 - tx)*(s01 - s00) + tx*(s11 - s10))
    else:
        gyn, gxn = np.gradient(sdf, 1./k)                       # node gradients (rows = iy, cols = ix)
        lerp = lambda a: ((1 - tx)*(1 - ty)*a[j, i] + tx*(1 - ty)*a[j, i + 1] + (1 - tx)*ty*a[j + 1, i] + tx*ty*a[j + 1, i + 1])
        gx, gy = lerp(gxn), lerp(gyn)
    ex, ey = ux - cx, uy - cy
    excess = np.sqrt(ex**2 + ey**2 + 1e-30) - 1e-15
    gx = np.where(np.abs(ux) > half, 0., gx) + ex/(excess + 1e-15)
    gy = np.where(np.abs(uy) > half, 0., gy) + ey/(excess + 1e-15)
    return s + excess, gx, gy


class ImageOcculter(dl.OpticalLayer):
    r: Array              # m; hole equivalent-area radius is sqrt(2) r (EllipticalOcculter convention)
    e: Array              # (2,) area-preserving ellipticity (axes fixed in the occulter frame), as EllipticalOcculter
    rotation: Array       # () deg, clockwise rotation of the image outline (full range); see module docstring
    rim_amp: Array        # (K, H)
    rim_opd: Array        # (K, H) m
    rim_scales: Array     # (K,) m, fixed
    wavelength0: Array
    sdf: Array            # (N, N) fixed SDF in units of R, d > 0 on mirror, index [iy, ix]
    extent: float = eqx.field(static=True)   # half-width of the SDF grid in units of R
    m_max: int = eqx.field(static=True)
    mirror: bool = eqx.field(static=True)    # reflect the outline x -> -x in the SDF frame (static; see with_orientation)
    edge: str = eqx.field(static=True)       # "soft_binarise" (default) | "coverage"
    grad: str = eqx.field(static=True)       # SDF gradient used for the edge normal / rim distance: "nodal" | "cell"
    normalise: bool
    invert: bool

    def __init__(self, r, e=onp.zeros(2), rotation=0., mirror: bool = False, sdf=None, extent=None, m_max=3,
                 rim_scales=None, rim_amp=None, rim_opd=None, wavelength0=1.6e-6, normalise: bool = False,
                 invert: bool = False, edge: str = "soft_binarise", grad: str = "nodal"):
        if edge not in ("coverage", "soft_binarise"):
            raise ValueError(f"edge must be 'coverage' or 'soft_binarise', got {edge!r}")
        if grad not in ("nodal", "cell"):
            raise ValueError(f"grad must be 'nodal' or 'cell', got {grad!r}")
        self.grad = grad
        if sdf is None:
            sdf, extent = load_hole_sdf()
        self.sdf = np.asarray(sdf, float)
        self.extent = float(extent)
        self.m_max = m_max
        self.mirror = bool(mirror)
        self.edge = edge
        self.r = np.asarray(r, float)
        self.e = np.asarray(e, float)
        self.rotation = np.asarray(rotation, float)
        if rim_scales is None:
            rim_scales = onp.array([0.25, 0.5, 1.0])*wavelength0*24.    # (0.25, 0.5, 1) lambda F#, F# = 24
        self.rim_scales = np.asarray(rim_scales, float)
        H = 1 + 2*(m_max - 1)
        K = len(self.rim_scales)
        self.rim_amp = np.zeros((K, H)) if rim_amp is None else np.asarray(rim_amp, float)
        self.rim_opd = np.zeros((K, H)) if rim_opd is None else np.asarray(rim_opd, float)
        self.wavelength0 = np.asarray(wavelength0, float)
        self.normalise = normalise
        self.invert = invert

    def with_orientation(self, rotation, mirror):
        """Copy with the given rotation (deg) and static mirror flag (static fields cannot go through eqx.tree_at)."""
        new = object.__new__(type(self))
        for f in dataclasses.fields(self):                       # ALL fields: new fields cannot be silently dropped
            object.__setattr__(new, f.name, getattr(self, f.name))
        object.__setattr__(new, "rotation", np.asarray(rotation, float))
        object.__setattr__(new, "mirror", bool(mirror))
        return new

    # ---- geometry -------------------------------------------------------------------------------------------
    def linear_map(self):
        """L (2x2, det 1): occulter-plane -> SDF frame (before dividing by R); L = Rot(rotation) @ S(e)."""
        e1, e2 = self.e
        A = np.array([[1 - e1, -e2], [-e2, 1 + e1]])/np.sqrt(1 - e1**2 - e2**2)
        S = (A + np.eye(2))/np.sqrt(np.trace(A) + 2.)          # sqrt of a det-1 symmetric positive matrix
        t = np.deg2rad(self.rotation)
        Rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return Rot @ S

    def _sdf_frame(self, xx, yy):
        R = np.sqrt(2.)*self.r
        L = self.linear_map()
        return (L[0, 0]*xx + L[0, 1]*yy)/R, (L[1, 0]*xx + L[1, 1]*yy)/R

    def _geometry(self, xx, yy):
        """(d_edge, gx, gy, d_rim, phi): d_edge = R sdf (m) and its occulter-plane gradient (L^T grad_u sdf);
        d_rim = d_edge/|L^T n| (n the unit normal; first-order true distance); phi azimuth in the SDF-frame coordinates, reflected
        (x -> -x) when mirror=True so the rim harmonics follow the outline."""
        R = np.sqrt(2.)*self.r
        L = self.linear_map()
        ux, uy = self._sdf_frame(xx, yy)
        sgn = -1. if self.mirror else 1.
        s, gux, guy = _bilinear(jax.lax.stop_gradient(self.sdf), sgn*ux, uy, self.extent, self.grad)
        gux = sgn*gux
        gx = L[0, 0]*gux + L[1, 0]*guy
        gy = L[0, 1]*gux + L[1, 1]*guy
        d = s*R
        # stretch of the normal direction by the map: |L^T n|, n = grad_u sdf / |grad_u sdf| (the stored sdf is a distance up
        # to ~8% bilinear/roughness noise in |grad|; using the unit normal keeps that noise out of the correction)
        d_rim = d*np.sqrt(gux**2 + guy**2 + 1e-12)/np.sqrt(gx**2 + gy**2 + 1e-12)
        return d, gx, gy, d_rim, np.arctan2(uy, sgn*ux)          # azimuth in the same (reflected) frame as the outline

    def signed_distance(self, xx, yy):
        """(d, phi): d (m) > 0 on the mirror (rim distance, stretch-corrected), phi azimuth in the SDF frame."""
        _, _, _, d, phi = self._geometry(xx, yy)
        return d, phi

    # ---- masks ----------------------------------------------------------------------------------------------
    def _fine(self, npixels, pixel_scale):
        xx, yy = dlu.pixel_coords(npixels*3, pixel_scale=pixel_scale/3)
        d, gx, gy, d_rim, phi = self._geometry(xx, yy)
        if self.edge == "coverage":
            disc3 = soft_coverage(d, gx, gy, pixel_scale/3)
            extra = None
        else:                                                    # v1: dlu.soft_binarise disc, 1-sub-pixel ramp for the rim gate
            disc3 = np.clip(0.5 - d/(pixel_scale/3), 0., 1.)
            extra = dlu.soft_binarise(-d, oversample=3) - self._pool(disc3, npixels)
        return d_rim, phi, disc3, extra

    def _pool(self, a, n):
        return a.reshape(n, 3, n, 3).mean((1, 3))

    def mask(self, npixels, pixel_scale, wavelength=None):
        wl = self.wavelength0 if wavelength is None else wavelength
        d, phi, disc3, extra = self._fine(npixels, pixel_scale)
        T3 = rim_transmission(d, phi, self.rim_amp, self.rim_opd, self.rim_scales, wl)
        m = rim_mask(disc3, T3, 3)
        if extra is not None:
            m = m + extra
        return 1 - m if self.invert else m

    def mask_terms(self, npixels, pixel_scale):
        """(M0, M1, wavelength0): mask(lam) ~= M0 + (lam0/lam - 1) M1."""
        d, phi, disc3, extra = self._fine(npixels, pixel_scale)
        T0, T1 = rim_terms_linear(d, phi, self.rim_amp, self.rim_opd, self.rim_scales, self.wavelength0)
        M0, M1 = rim_mask_terms(disc3, T0, T1)
        if extra is not None:
            M0 = M0 + extra
        if self.invert:
            M0, M1 = 1 - M0, -M1
        return M0, M1, self.wavelength0

    def __call__(self, wavefront):
        wf = wavefront * self.mask(wavefront.npixels, wavefront.pixel_scale, wavefront.wavelength)
        return wf.normalise() if self.normalise else wf


def best_orientation(loss_fn, occulter, rotations=(0., 90., 180., 270.), mirrors=(False, True)):
    """Evaluate loss_fn(occulter') for rotation in {0, 90, 180, 270} deg x mirror in {False, True} (all else unchanged) and
    return (best occulter, table) with table a list of dicts {rotation, mirror, loss}, in evaluation order. Non-finite losses
    are skipped (kept in the table); ValueError if all are non-finite. No fitting."""
    table, best, bl = [], None, onp.inf
    for mirror in mirrors:
        for rot in rotations:
            cand = occulter.with_orientation(rot, mirror)
            l = float(loss_fn(cand))
            table.append(dict(rotation=float(rot), mirror=bool(mirror), loss=l))
            if onp.isfinite(l) and l < bl:                       # non-finite losses are recorded but never selected
                best, bl = cand, l
    if best is None:
        raise ValueError("best_orientation: all candidate losses are non-finite: " + repr(table))
    return best, table


# ====================================================================================================================
# Fixed leaves and OPTICS_PARAMS entries (mirrored in the proposed models.py; test_integration asserts they agree)
# ====================================================================================================================
FIXED_LEAVES = ("rim_scales", "wavelength0", "sdf", "ghost_offset", "ghost_width", "spot_width")
_P = "occulter.layers.occulter."
CLIMB_PARAMS = [
    ("occulter_edge", [_P + "edge_coeffs"], lambda x: x),            # (M-2, 2) m = 3..M relative radius, unscaled
    ("occulter_rim_amp", [_P + "rim_amp"], lambda x: x),             # (K, H) log-amplitude, unscaled
    ("occulter_rim_opd", [_P + "rim_opd"], lambda x: x*1e-9),        # (K, H) nm -> m
]
IMAGE_PARAMS = [
    ("occulter_rotation", [_P + "rotation"], lambda x: x),           # degrees (the leaf is in degrees)
    ("occulter_rim_amp", [_P + "rim_amp"], lambda x: x),
    ("occulter_rim_opd", [_P + "rim_opd"], lambda x: x*1e-9),
]


# ====================================================================================================================
# Priors
# ====================================================================================================================
DEFAULT_D_RING = onp.array([0.5, 1., 2., 5., 10., 20., 40.])*1e-6   # distances from the edge (m) at which OPD is penalised


def edge_profile(coeffs, scales, d_ring=DEFAULT_D_RING, n_phi=None, tau=TAU_DEFAULT):
    """Rim field sum_{kh} coeffs[k, h] B_k(d) H_h(phi) evaluated on a ring: array (len(d_ring), n_phi) over
    d_ring (m) x phi (n_phi uniform points, default max(16, 4 m_max + 4)). The coefficients' units carry through
    (nm for rim_opd, log-amplitude for rim_amp)."""
    coeffs = np.asarray(coeffs)
    m_max = m_max_from_n(coeffs.shape[1])
    n_phi = n_phi or max(16, 4*m_max + 4)
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    D = np.asarray(d_ring)[:, None]*np.ones((1, n_phi))
    PHI = np.ones((len(d_ring), 1))*phi[None]
    return _contract(coeffs, rim_basis(D, PHI, scales, m_max, tau))


def glint_log_prior(params, scales, sigma_opd=30., sigma_amp=0.03, sigma_edge=0.01, d_ring=DEFAULT_D_RING, n_phi=None,
                    sigma_ridge_opd=None, sigma_ridge_amp=None, tau=TAU_DEFAULT, sigma_pos_amp=3e-3, occulter=None):
    """Gaussian log-prior (up to a constant) on a params dict (as model.params; rim_opd in nm).
    The rim priors are on the RESULTING edge OPD / log-amplitude, not per coefficient (the exponential radial bases
    are correlated at 0.94, so per-coefficient priors let large +/- pairs cancel): -1/2 mean_{d,phi} (OPD(d, phi) /
    sigma_opd)^2 over the ring `d_ring` x phi, and likewise for the log-amplitude with sigma_amp. So a uniform OPD of
    sigma_opd nm over the ring costs 1/2. Optional per-coefficient ridge (sigma_ridge_*) to remove the ring's null
    space. occulter_edge (relative radius coefficients) keeps a per-coefficient sigma_edge. Missing keys skipped.
    Positive log-amplitude (gain; blocked by logamp_guard, so it would be a dead direction) is penalised by the hinge
    -1/2 mean (relu(amp profile)/sigma_pos_amp)^2. If `occulter` (a CLIMBGlintOcculter) is given its edge_log_prior
    (star-shapedness) is added."""
    lp = 0.
    for key, sig, rsig in (("occulter_rim_opd", sigma_opd, sigma_ridge_opd), ("occulter_rim_amp", sigma_amp, sigma_ridge_amp)):
        if key in params:
            prof = edge_profile(params[key], scales, d_ring, n_phi, tau)
            lp = lp - 0.5*np.mean((prof/sig)**2)
            if key == "occulter_rim_amp":
                lp = lp - 0.5*np.mean((np.maximum(prof, 0.)/sigma_pos_amp)**2)
            if rsig is not None:
                lp = lp - 0.5*np.sum((np.asarray(params[key])/rsig)**2)
    if "occulter_edge" in params:
        lp = lp - 0.5*np.sum((np.asarray(params["occulter_edge"])/sigma_edge)**2)
    if occulter is not None and hasattr(occulter, "edge_log_prior"):
        lp = lp + occulter.edge_log_prior()
    return lp
