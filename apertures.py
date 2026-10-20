from dLux.optical_systems import OpticalSystem
import jax.numpy as np
from jax import Array, vmap

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx
import jax
import numpy as onp

from abcdLux.lct import *
from abcdLux.abcd import *

from spectra import hermite_propagate

# ---- dLux soft-aperture fixes (monkeypatches of dLux.utils.geometry) ----------------------------
# 1. soften: stock dLux rescales the clipped distances by the array's own min/max. When a feature is
#    narrower than 2*clip_dist (the HST primary spider at 512 px / softening 5) the max is never
#    reached, so the transmission is rescaled by a data-dependent peak and its gradient flows through a
#    tied max/min: wrong, jit-dependent gradients. Here the scale is fixed, t = (clip(d,-c,c)+c)/(2c).
#    Identical to stock for wide features; narrow ones now stay partly transmitting at their centre
#    when softening > 2*half_width/pixel_scale (use primary softening <= 2 for a fully opaque spider).
# 2. soft_spider: the jvp of the stock `-spider_distances.min(axis=0)` is wrong under jit (corrupts
#    Hessians / jvp Gauss-Newton for primary_rot/shear/spider). Picking the arm with a stop-gradient
#    argmin and gathering it gives the same values and correct tangents.
import dLux.utils.geometry as dlg


def soften_fixed_scale(distances, clip_dist, invert=False):
    if invert:
        distances = -distances
    d = np.clip(distances, -clip_dist, clip_dist)
    constant_support = (d > 0).astype(d.dtype)
    return np.where(d.max() == d.min(), constant_support, (d + clip_dist) / (2. * clip_dist))


def soft_spider_gather(coords, width, angles, clip_dist=0.1, invert=False):
    dists = vmap(lambda a: dlg.spider_distance(coords, width, a))(np.asarray(angles, float))
    idx = np.argmin(jax.lax.stop_gradient(dists), axis=0)[None]
    spiders = -np.take_along_axis(dists, idx, axis=0)[0]
    return dlg.soften(spiders, clip_dist, invert)


dlg.soften = soften_fixed_scale             # soft_circle/spider/... look `soften` up in this module
dlg.soft_spider = dlu.soft_spider = soft_spider_gather   # dl.Spider.transmission calls dlu.soft_spider


class HSTMainAperture(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "mirror" : dl.CircularAperture(
                radius = 1.2,
                softening=self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.0256,#0.022*1.2,#0.038*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                # Secondary-plane features (spider, secondary) sit ~4.9 m in front of the primary, so an off-axis field
                # shears their shadow relative to the rim and pads (~1 cm for NIC2, 7.4' off-axis): fitted via secondary_shift
                transformation=dl.CoordTransform(translation=np.zeros(2)),
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330*1.2,
                occulting = True,
                transformation=dl.CoordTransform(translation=np.zeros(2)),
                softening = self.softening
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921*1.2, 0),
                ),
                softening = self.softening
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7606*1.2),
                ),
                softening=self.softening
            )
        }



class DistortedCoordTransform(dl.CoordTransform):
    """CoordTransform followed by a polynomial distortion (dl.DistortedCoords) evaluated in units of `radius`, in a
    frame rotated by `frame` (rotated back afterwards). An x^k or y^k coefficient is the coordinate shift on that axis
    at the edge as a fraction of the radius (the aperture itself moves the opposite way). The primary's transformation
    already rotates into the spider-aligned frame (frame=0); the low-order Zernike disc is unrotated, so it uses
    frame=pi/4 to see the same distortion without rotating its Zernikes (primary_rot's ~1 deg is neglected). As the
    transformation of a CompoundAperture it distorts every sub-aperture (rim, secondary, spiders, pads) together."""
    distortion: dl.DistortedCoords
    radius: float = eqx.field(static=True)
    frame: float = eqx.field(static=True)

    def __init__(self, order=3, radius=1.2, frame=0., **kwargs):
        super().__init__(**kwargs)
        # built here (gen_powers cannot be traced under jit); integer powers so x**p has no log(x) term, whose
        # gradient is nan where a coordinate is exactly zero (e.g. on the diagonals after the frame rotation)
        self.distortion = dl.DistortedCoords(order)
        self.distortion = self.distortion.set("powers", self.distortion.powers.astype(int))
        self.radius = radius
        self.frame = frame

    def __call__(self, coords):
        coords = dlu.rotate_coords(super().__call__(coords), self.frame)
        return dlu.rotate_coords(self.radius*self.distortion(coords/self.radius), -self.frame)


class NICMOSColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.955,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.077*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372*1.2,
                occulting = True,
                softening = self.softening
            ),
        }

class NIC2ColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25, normalise=False):
        self.normalise = normalise
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.9768,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.071*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.365*1.2,
                occulting = True,
                softening = self.softening
            ),

            "pad_1" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.9021*1.2, 0),
                    rotation=np.deg2rad(0)
                ),
                softening = self.softening
            ),
            "pad_2" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7655*1.2),
                    rotation=np.deg2rad(-121.52)
                ),
                softening = self.softening
            ),
            "pad_3" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7706*1.2),
                    rotation=np.deg2rad(121.15)
                ),
                softening = self.softening
            )
        }

def fourier_circle(x0, y0, r0, cc, ss, xx, yy):
    '''
    Create a circle with a radius that varies as a function of angle, defined by a Fourier series with coefficients cc and ss.
    The circle is centered at (x0, y0) and the radius is defined as r0 + cc * cos(theta) + ss * sin(theta), 
    where theta is the angle from the center of the circle to each point in the grid defined by xx and yy.

    Parameters
    ----------
    x0 : float
        x-coordinate of the center of the circle
    y0 : float
        y-coordinate of the center of the circle
    r0 : float
        Base radius of the circle
    cc : float
        Coefficient for the cosine term in the Fourier series
    ss : float
        Coefficient for the sine term in the Fourier series
    xx : jnp.ndarray
        2D array of x-coordinates for the grid
    yy : jnp.ndarray
        2D array of y-coordinates for the grid

    Returns
    -------
    latent : jnp.ndarray
        2D array representing the circle with varying radius, where each point is defined 
        by the distance from the center and the angle to that point.
    '''

    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    theta = np.arctan2(yy - y0, xx - x0)

    radius = r0 + np.sum(vmap(lambda c, s, n: c * np.cos(n*theta) + s * np.sin(n*theta))(cc, ss, np.arange(len(cc))+1), axis=0)#**2

    # radius = r0 + (cc * np.cos(theta) + ss * np.sin(theta))**2

    latent = -(rr ** 2)/(radius*2)**2 + 0.5

    return latent

def climb_circle(x0, y0, r0, cc, ss, xx, yy):
    '''
    Apply CLIMB to a Fourier-defined aperture.

    Assumes the latent is oversampled by a factor of 3.
    '''

    latent = fourier_circle(x0, y0, r0, cc, ss, xx, yy)

    latent = dlu.soft_binarise(latent,oversample=3)

    return latent

class CLIMBOcculter(dl.OpticalLayer):
    r: Array
    cc: Array
    ss: Array

    normalise: bool
    invert: bool

    def __init__(self, r, cc, ss, normalise: bool = False, invert: bool=False):
        assert cc.shape == ss.shape
        self.r = r
        self.cc = cc
        self.ss = ss
        self.normalise = normalise
        self.invert = invert
    
    def mask(self, npixels, pixel_scale):
        """Wavelength-independent transmission (so it can be hoisted out of the wavelength loop)."""
        xx, yy = dlu.pixel_coords(npixels*3, pixel_scale=pixel_scale/3)
        circ = climb_circle(0., 0., self.r, self.cc, self.ss, xx, yy)
        return 1-circ if self.invert else circ

    def __call__(self, wavefront):
        wf = wavefront * self.mask(wavefront.npixels, wavefront.pixel_scale)

        if self.normalise:
            return wf.normalise()
        return wf

class EllipticalOcculter(dl.OpticalLayer):
    """CLIMB occulter with an elliptical edge, parameterised by the ellipticity components e = (e1, e2): the edge is
    x^T A x = r^2 with A = [[1 - e1, -e2], [-e2, 1 + e1]] / sqrt(1 - e1^2 - e2^2). det A = 1, so r is the radius of
    the circle of equal area; |e| = (a^2 - b^2) / (a^2 + b^2) sets the axis ratio b/a = sqrt((1 - |e|) / (1 + |e|))
    and the major axis lies at atan2(e2, e1) / 2 from the x axis. Smooth through e = 0 (the circle of CLIMBOcculter
    with no Fourier terms), unlike an (ellipticity, angle) pair.
    glint : (2 n_max + 1, 2) real / imaginary parts of the Fourier coefficients (g0, g1 cos, g1 sin, ...) in the
            occulter-plane angle theta (from its x axis, after the model's flip) of a complex ring on the edge,
            subtracted from the obstruction: mask = disc - g(theta) exp(-(rho - rho_edge)^2 / 2 w^2), i.e. light the
            hole's edge reflects or scatters back into the beam (positive real g0 adds in-phase light). Zero = no glint.
    glint_width : w (m) in the occulter plane."""
    r: Array
    e: Array
    glint: Array
    glint_width: Array

    normalise: bool
    invert: bool

    def __init__(self, r, e=np.zeros(2), glint=np.zeros((1, 2)), glint_width=3e-6, normalise: bool = False, invert: bool = False):
        self.r = np.asarray(r, float)
        self.e = np.asarray(e, float)
        self.glint = np.asarray(glint, float)
        self.glint_width = np.asarray(glint_width, float)
        self.normalise = normalise
        self.invert = invert

    def _rho2(self, xx, yy):
        """Squared elliptical radius (area-preserving, see class docstring)."""
        e1, e2 = self.e
        return ((1 - e1)*xx**2 - 2*e2*xx*yy + (1 + e1)*yy**2) / np.sqrt(1 - e1**2 - e2**2)

    def mask(self, npixels, pixel_scale):
        """Wavelength-independent transmission (so it can be hoisted out of the wavelength loop); complex (glint)."""
        xx, yy = dlu.pixel_coords(npixels*3, pixel_scale=pixel_scale/3)
        rho2 = self._rho2(xx, yy)
        circ = dlu.soft_binarise(-rho2/(2*self.r)**2 + 0.5, oversample=3)   # as fourier_circle: edge at rho = sqrt(2) r

        # glint ring on the same 3x oversampled grid, mean-pooled to the mask's pixels (w is only ~2 pixels)
        theta = np.arctan2(yy, xx)
        harmonics = np.stack([np.ones_like(theta)] + [f(k*theta) for k in range(1, (len(self.glint) - 1)//2 + 1)
                                                      for f in (np.cos, np.sin)])
        g = np.tensordot(self.glint[:, 0] + 1j*self.glint[:, 1], harmonics, 1)
        ring = g * np.exp(-(np.sqrt(rho2) - np.sqrt(2)*self.r)**2 / (2*self.glint_width**2))
        ring = ring.reshape(npixels, 3, npixels, 3).mean((1, 3))
        circ = circ - ring
        return 1-circ if self.invert else circ

    def __call__(self, wavefront):
        wf = wavefront * self.mask(wavefront.npixels, wavefront.pixel_scale)

        if self.normalise:
            return wf.normalise()
        return wf

class SoummerFastObstruction(dl.OpticalLayer):
    layers: dict
    normalise: bool
    def __init__(self, layers, normalise: bool = False):
        self.layers = dlu.list2dictionary(layers, True)
        self.normalise = normalise
    
    def __call__(self, wavefront):
        wf_prop = wavefront
        for prop in self.layers.values():
            wf_prop = prop.apply(wf_prop)
        
        wf = wavefront.flip(axis=(0,1)) - wf_prop
        if self.normalise:
            return wf.normalise()
        return wf

# Unit of primary_amp (log-amplitude): the field perturbation of 1 nm OPD at 1.87 um, so primary_amp
# and primary_opd steps of the same size (e.g. the same adam learning rate) are comparable.
AMP_UNIT = 2*np.pi*1e-9/1.87e-6


class TransformedFourierBasis(dl.OpticalLayer):
    """OPD from a mean plus a linear combination of modes, all Fourier coefficients (e.g. from a PCA) in units of
    `scale` (1e-9: nm)."""
    mean: Array
    modes: Array
    coefficients: Array
    kernels: tuple[Array, Array]
    n_modes: Array
    scale: float = eqx.field(static=True)

    def __init__(self, npix, n_modes, basisfile, scale=1e-9):
        basis = np.load(basisfile)
        self.mean = basis["mean"]
        self.modes = basis["modes"]
        self.n_modes = n_modes
        self.scale = scale
        self.kernels = dlu.fourier_kernels((n_modes,n_modes), (npix,npix))
        self.coefficients = np.zeros(self.modes.shape[0])
    
    def eval_basis(self):
        fourier_coeffs = self.scale*(self.mean + np.dot(self.coefficients, self.modes)).reshape((self.n_modes,self.n_modes))
        return dlu.eval_fourier_basis(fourier_coeffs, *self.kernels)

    def __call__(self, wavefront):
        return wavefront.add_opd(self.eval_basis())


class TransformedFourierAmplitude(TransformedFourierBasis):
    """Log-amplitude counterpart of TransformedFourierBasis (mean and modes in units of AMP_UNIT)."""

    def __init__(self, npix, n_modes, basisfile):
        super().__init__(npix, n_modes, basisfile, scale=AMP_UNIT)

    def __call__(self, wavefront):
        return wavefront * np.exp(self.eval_basis())


class ScanAberratedAperture(dl.AberratedAperture):
    """dl.AberratedAperture (circular apertures only) whose OPD sum_j c_j Z_j(T x) is a checkpointed
    lax.scan over modes. The stock stack keeps ~1.5 GB of per-mode intermediates for reverse mode
    (N=512, 46 modes); here only one mode is live at a time."""
    def eval_basis(self, coords):
        assert self.aperture.nsides == 0
        if self.aperture.transformation is not None:
            coords = self.aperture.transformation(coords)
        coords = coords / self.aperture.extent
        zs = self.basis.basis
        # Zernike c, k tables padded to equal length with zeros; n, m are static python ints
        K = max(len(z._c) for z in zs)
        C = np.stack([np.pad(np.asarray(z._c), (0, K - len(z._c))) for z in zs])
        Kp = np.stack([np.pad(np.asarray(z._k), (0, K - len(z._k))) for z in zs])
        n = np.asarray([abs(z.n) for z in zs], float)
        am = np.asarray([abs(z.m) for z in zs], float)
        sn = np.asarray([z.m < 0 for z in zs], float)                       # sin vs cos
        nm = np.asarray([onp.sqrt(z.n + 1) * (onp.sqrt(2.) if z.m != 0 else 1.) for z in zs])

        @jax.checkpoint
        def add_mode(acc, xy, c_row, k_row, n_, am_, sn_, nm_, coef):
            rho, theta = dlu.cart2polar(xy)
            rads = jax.lax.pow(rho[:, :, None], (n_ - 2 * k_row)[None, None, :])
            radial = (c_row * rads).sum(-1)
            return acc + coef * nm_ * (rho <= 1.) * radial * np.cos(am_ * theta - sn_ * (np.pi / 2))

        body = lambda acc, xs: (add_mode(acc, coords, *xs), None)
        out, _ = jax.lax.scan(body, np.zeros(coords.shape[1:]), (C, Kp, n, am, sn, nm, self.coefficients))
        return out


class FourierAmplitude(dl.FourierBasis):
    """Real Fourier-series log-amplitude screen: field *= exp(sum of coefficients x Fourier modes)."""

    def __call__(self, wavefront):
        return wavefront * np.exp(self.eval_basis())


class _AmpOpd:
    """Stand-in for dl.Wavefront that accumulates (amplitude, OPD) instead of a phasor, so the
    wavelength-independent layers can be evaluated once (phase = 2 pi OPD / wavelength is applied later).
    Supports only what those layers call; anything else (e.g. `.wavelength`) raises AttributeError."""
    _from_ref = ("npixels", "pixel_scale", "diameter", "coordinates")

    def __init__(self, ref, amp, opd):
        self.ref, self.amp, self.opd = ref, amp, opd

    def __getattr__(self, name):
        if name in self._from_ref:
            return getattr(self.ref, name)
        raise AttributeError(f"{name} is not available on a wavelength-independent layer")

    def __mul__(self, other): return _AmpOpd(self.ref, self.amp * other, self.opd)
    def add_opd(self, opd): return _AmpOpd(self.ref, self.amp, self.opd + opd)
    def flip(self, axis): return _AmpOpd(self.ref, np.flip(self.amp, axis), np.flip(self.opd, axis))
    def tilt(self, angles, unit="rad"):
        coords = self.ref.coordinates(scale=dlu.unit_factor_to_rad(unit))
        return self.add_opd(np.sum(np.asarray(angles, float)[:, None, None] * coords, axis=0))
    def normalise(self):
        return _AmpOpd(self.ref, self.amp * np.sqrt(1. / np.sum(self.amp ** 2)), self.opd)


class NICMOSCoronagraph(dl.LayeredOpticalSystem):
    """wl_batch : None -> vmap all wavelengths at once (dLux default).  int k -> lax.map over wavelengths in chunks of k
                  (sequential; with remat=True the reverse pass only holds k wavelengths of residuals).
    remat    : jax.checkpoint the per-wavelength propagation (only useful together with wl_batch).
    amplitude : add a Fourier log-amplitude screen "primary_amp" after primary_opd.
    turboklip : file of OPD modes (mean, modes) for a "primary_klip" layer.
    amp_klip  : file of log-amplitude modes (mean, modes) for a "primary_amp_klip" layer.
    n_nodes   : None -> propagate every wavelength. int k -> Hermite-interpolate the PSF between k knot wavelengths
                (spectra.hermite_propagate; ~9 knots on-axis for ~2e-4 accuracy, independent of the fine grid).
    pad       : model the PSF `pad` detector pixels wider on each side than psf_npixels, for a
                NICMOSDetector(oversample, psf_npixels, pad=pad), which crops it back after resampling.
    """
    wl_batch: int | None = eqx.field(static=True)
    remat: bool = eqx.field(static=True)
    n_nodes: int | None = eqx.field(static=True)

    def __init__(self, wf_npixels, psf_npixels, oversample, n_modes=12, n_zernikes=1., turboklip=None, amp_klip=None, wl_batch=None, remat=False, amplitude=False, n_nodes=None, pad=0):
        diameter = 3.
        layers = [
            ("primary",HSTMainAperture(transformation=DistortedCoordTransform(rotation=np.pi/4), softening=5)),

            ("primary_tilt", dl.Tilt(angles=(0.,0.))),

            ("primary_opd", dl.FourierBasis(wf_npixels, n_modes=n_modes)),
        ]

        if amplitude:
            layers += [("primary_amp", FourierAmplitude(wf_npixels, n_modes=n_modes))]

        if turboklip:
            layers += [
                ("primary_klip", TransformedFourierBasis(wf_npixels, n_modes, turboklip)),
            ]

        if amp_klip:
            layers += [("primary_amp_klip", TransformedFourierAmplitude(wf_npixels, n_modes, amp_klip))]
        
        layers += [
            ("primary_low", ScanAberratedAperture(
                    dl.layers.CircularAperture(1.2,transformation=DistortedCoordTransform(frame=np.pi/4, translation=np.zeros(2)),),
                    noll_inds=np.arange(4,4+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                    
                )),

            ("flip", dl.Flip(axes=(0,1))),

            # ("prop1", dl.MFT(128, pixel_scale=dlu.arcsec2rad(0.01))),
            # ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3), np.zeros(1), np.zeros(1))),
            # ("occulter", dl.CircularAperture(dlu.arcsec2rad(0.3)*24*2.4)),

            # ("prop1", dl.MFT(128, focal_length=24*2.4, pixel_scale=3e-6)),
            # ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3)*24*2.4, np.zeros(1), np.zeros(1))),



            ("occulter", SoummerFastObstruction([
                ("prop1", dl.MFT(256, focal_length=24*2.4, pixel_scale=1.5e-6)),
                # ("occulter", dl.CircularAperture(dlu.arcsec2rad(0.3)*24*2.4)),
                ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3)*24*2.4, np.zeros(1), np.zeros(1))),
                ("prop1", dl.MFT(wf_npixels, focal_length=24*2.4, pixel_scale=diameter/wf_npixels)),
            ])),


            ("cold_mask",   NIC2ColdMask(transformation=DistortedCoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=5, normalise=False)),

            ("cold_mask_opd", ScanAberratedAperture(
                    # rotation as the mask's, so cold_mask_distortion acts in the same frame even if cold_mask_rot is unset
                    dl.layers.CircularAperture(1.2, transformation=DistortedCoordTransform(translation=np.asarray((-0.05, -0.05)), rotation=np.pi/4)),
                    noll_inds=np.arange(4,20),
                    coefficients = np.zeros(1),
                )),

            ("cold_mask_tilt", dl.Tilt(angles=(0.,0.))),
            
            ("prop1", dl.MFT((psf_npixels + 2*pad)*oversample, focal_length=45.7*2.4, pixel_scale=40e-6/oversample)),
        ]

        super().__init__(wf_npixels, diameter, layers)
        self.wl_batch = wl_batch
        self.remat = bool(remat)
        self.n_nodes = n_nodes

    # Layout is fixed: [wavelength-independent layers] occulter [wavelength-independent layers] final MFT.
    # The wavelength-independent layers (Zernikes, soft apertures, CLIMB mask) are evaluated once per call
    # instead of once per wavelength, which is what makes the wavelength loop cheap to remat / chunk.
    def _split_layers(self):
        layers, k = list(self.layers.values()), list(self.layers).index("occulter")
        return layers[:k], layers[k], layers[k+1:-1], layers[-1]

    def _static_stage(self, offset):
        """((amp, opd) before the occulter, occulter mask, (amp, opd) after it). Wrapped in a checkpoint
        by propagate so that only these few (N, N) arrays are kept for the backward pass."""
        ref = dl.Wavefront(1e-6, self.wf_npixels, self.diameter)     # only its coordinates are used
        before, occulter, after, _ = self._split_layers()
        zeros, ones = np.zeros((self.wf_npixels,) * 2), np.ones((self.wf_npixels,) * 2)
        # the source-offset tilt comes first (before the flip); a Wavefront starts at amplitude 1/npix^2
        rec = _AmpOpd(ref, ref.amplitude, zeros).tilt(offset)
        for layer in before:
            rec = layer(rec)
        rec2 = _AmpOpd(ref, ones, zeros)
        for layer in after:
            rec2 = layer(rec2)
        mft1, climb, _ = occulter.layers.values()
        w1 = mft1(ref)                                               # only pixel_scale/npixels are used
        return (rec.amp, rec.opd), climb.mask(w1.npixels, w1.pixel_scale), (rec2.amp, rec2.opd)

    def _one_wavelength(self, static, wavelength, weight):
        (amp1, opd1), mask, (amp2, opd2) = static
        _, occulter, _, final_mft = self._split_layers()
        wf = dl.Wavefront(wavelength, self.wf_npixels, self.diameter)
        wf = wf.set(phasor=np.ones((self.wf_npixels,) * 2, complex))   # the 1/npix^2 amplitude is in amp1
        wf = (wf * amp1).add_opd(opd1)
        mft1, _, mft2 = occulter.layers.values()
        wf = wf.flip(axis=(0, 1)) - mft2(mft1(wf) * mask)              # Babinet, as SoummerFastObstruction
        wf = (wf * amp2).add_opd(opd2)
        wf = final_mft(wf).multiply("phasor", weight ** 0.5)
        return wf.psf, wf.pixel_scale

    def propagate(self, wavelengths, offset=None, weights=None, return_wf=False, return_psf=False, n_nodes=None):
        """n_nodes : overrides self.n_nodes for this call (e.g. fewer knots for faint off-axis sources)."""
        n_nodes = self.n_nodes if n_nodes is None else n_nodes
        if return_wf:                                   # rarely needed: keep the stock dLux path
            return super().propagate(wavelengths, offset, weights, return_wf, return_psf)
        wavelengths = np.atleast_1d(wavelengths)
        weights = np.ones_like(wavelengths) / len(wavelengths) if weights is None else np.atleast_1d(weights)
        offset = np.zeros(2) if offset is None else np.asarray(offset)

        static = eqx.filter_checkpoint(NICMOSCoronagraph._static_stage)(self, offset)
        one = lambda s, st, wl, w: s._one_wavelength(st, wl, w)
        if self.remat:
            one = eqx.filter_checkpoint(one)
        if n_nodes and wavelengths.size > n_nodes:     # with <= n_nodes wavelengths the direct sum is exact and cheaper
            psf, ps = hermite_propagate(lambda wl: one(self, static, wl, 1.), wavelengths, weights, n_nodes, self.wl_batch)
            return dl.PSF(psf, ps) if return_psf else psf
        if self.wl_batch is None:
            psfs, ps = eqx.filter_vmap(lambda wl, w: one(self, static, wl, w))(wavelengths, weights)
        else:
            psfs, ps = jax.lax.map(lambda x: one(self, static, *x), (wavelengths, weights), batch_size=self.wl_batch)
        psf = psfs.sum(0)
        return dl.PSF(psf, ps.mean()) if return_psf else psf


class FresnelFastObstruction(dl.OpticalLayer):
    """SoummerFastObstruction with Fresnel propagation, so the occulter can be out of focus (Soummer et al. 2007,
    Babinet). The relay is lens (focal_length) -> defocus1 -> occulter -> defocus2 -> lens, i.e.
    M1 = D(defocus1) F(f) to the occulter and M2 = F(f) D(defocus2) back, and

        u_lyot = -flip(u) * exp(-i pi c r^2 / lambda) - LCT_M2(mask * LCT_M1(u)),

    where M2 M1 = [[-1, 0], [c, -1]], c = (defocus1 + defocus2) / f^2: the direct (unobstructed) term is the
    closed form of that b = 0 LCT, an inversion with curvature (-flip(u) in focus). The occulter's mask(n, d) is the obstruction itself (1 on the
    disc, not inverted), on an npixels grid of pixel_scale (m) in the occulter plane; the field returns to the
    input grid. The occulter is held as layers["occulter"] so its parameter paths match SoummerFastObstruction's."""
    layers: dict
    focal_length: Array
    defocus1: Array
    defocus2: Array
    npixels: int = eqx.field(static=True)
    pixel_scale: float = eqx.field(static=True)
    normalise: bool = eqx.field(static=True)

    def __init__(self, occulter, focal_length, npixels, pixel_scale, defocus1=0., defocus2=0., normalise=False):
        self.layers = {"occulter": occulter}
        self.focal_length = np.asarray(focal_length, float)
        self.defocus1 = np.asarray(defocus1, float)
        self.defocus2 = np.asarray(defocus2, float)
        self.npixels = npixels
        self.pixel_scale = pixel_scale
        self.normalise = normalise

    def mask(self):
        """Wavelength-independent occulter mask (so it can be hoisted out of the wavelength loop)."""
        return self.layers["occulter"].mask(self.npixels, self.pixel_scale)

    def apply_mask(self, wavefront, mask):
        u, xs, lam, f = wavefront.phasor, wavefront.xs, wavefront.wavelength, self.focal_length
        xo = (np.arange(self.npixels) - (self.npixels - 1) / 2) * self.pixel_scale
        M1 = compose_abcd([abcd_fraunhofer(f), abcd_free_space(self.defocus1)])
        M2 = compose_abcd([abcd_free_space(self.defocus2), abcd_fraunhofer(f)])
        u_res = lct_kernel_prop(mask * lct_kernel_prop(u, *lct_kernels(xs, xo, lam, M1)), *lct_kernels(xo, xs, lam, M2))
        c = (self.defocus1 + self.defocus2) / f**2
        u_dir = -np.flip(u, (0, 1)) * np.exp(-1j*np.pi*c*(xs[None, :]**2 + xs[:, None]**2)/lam)
        wf = wavefront.set(phasor=u_dir - u_res)
        if self.normalise:
            return wf.normalise()
        return wf

    def __call__(self, wavefront):
        return self.apply_mask(wavefront, self.mask())


class NICMOSFresnelCoronagraph(dl.LayeredOpticalSystem):
    """NICMOSCoronagraph with Fresnel (ABCD) propagation through the occulter, so the occulter (defocus1/2)
    and detector (defocus3) can be out of focus. Same layer names and parameter paths as NICMOSCoronagraph
    (incl. "prop1" for the final propagation, so fnumber sets its focal length). The occulter is a
    FresnelFastObstruction (defocus1/2 are its "occulter.defocus1/2") around an EllipticalOcculter
    (occulter_ellipticity = (e1, e2), not occulter_coeffs).
    amplitude, turboklip, amp_klip, n_nodes, pad : as NICMOSCoronagraph.
    """
    n_nodes: int | None = eqx.field(static=True)

    def __init__(self, wf_npixels, psf_npixels, oversample, n_modes=12, n_zernikes=1., turboklip=None, amp_klip=None, amplitude=False, n_nodes=None, pad=0):
        diameter = 3.
        layers = [
            ("primary",HSTMainAperture(transformation=DistortedCoordTransform(rotation=np.pi/4), softening=2)),

            ("primary_tilt", dl.Tilt(angles=(0.,0.))),

            ("primary_opd", dl.FourierBasis(wf_npixels, n_modes=n_modes)),
        ]

        if amplitude:
            layers += [("primary_amp", FourierAmplitude(wf_npixels, n_modes=n_modes))]

        if turboklip:
            layers += [
                ("primary_klip", TransformedFourierBasis(wf_npixels, n_modes, turboklip)),
            ]

        if amp_klip:
            layers += [("primary_amp_klip", TransformedFourierAmplitude(wf_npixels, n_modes, amp_klip))]

        layers += [
            ("primary_low", ScanAberratedAperture(
                    dl.layers.CircularAperture(1.2,transformation=DistortedCoordTransform(frame=np.pi/4, translation=np.zeros(2)),),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )),

            ("flip", dl.Flip(axes=(0,1))),

            ("occulter", FresnelFastObstruction(EllipticalOcculter(dlu.arcsec2rad(0.3)*24*2.4), 24*2.4, 256, 1.5e-6)),

            ("cold_mask",   NIC2ColdMask(transformation=DistortedCoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2, normalise=False)),

            ("cold_mask_opd", ScanAberratedAperture(
                    # rotation as the mask's, so cold_mask_distortion acts in the same frame even if cold_mask_rot is unset
                    dl.layers.CircularAperture(1.2, transformation=DistortedCoordTransform(translation=np.asarray((-0.05, -0.05)), rotation=np.pi/4)),
                    noll_inds=np.arange(4,20),
                    coefficients = np.zeros(1),
                )),

            ("cold_mask_tilt", dl.Tilt(angles=(0.,0.))),

            ("prop1", dl.MFTPropagator(
                [
                    ("ThinLens", dl.ABCDConjugatePlane(45.7*2.4)),
                    ("FreeSpace", dl.ABCDFreeSpace(0.)),
                ],
                dl.CoordSpec(n=(psf_npixels + 2*pad)*oversample, d=40e-6/oversample)
            )),
        ]

        super().__init__(wf_npixels, diameter, layers)
        self.n_nodes = n_nodes

    def propagate(self, wavelengths, offset=None, weights=None, return_wf=False, return_psf=False, n_nodes=None):
        """n_nodes : overrides self.n_nodes for this call (e.g. fewer knots for faint off-axis sources)."""
        n_nodes = self.n_nodes if n_nodes is None else n_nodes
        # with <= n_nodes wavelengths (e.g. a single-wavelength resolved source) the direct sum is exact and cheaper,
        # and Hermite needs >= 2 distinct knots
        if return_wf or not n_nodes or np.size(wavelengths) <= n_nodes:
            return super().propagate(wavelengths, offset, weights, return_wf, return_psf)
        wavelengths = np.atleast_1d(wavelengths)
        weights = np.ones_like(wavelengths) / len(wavelengths) if weights is None else np.atleast_1d(weights)
        offset = np.zeros(2) if offset is None else np.asarray(offset)
        mono = lambda wl: (lambda wf: (wf.psf, wf.pixel_scale))(self.propagate_mono(wl, offset, return_wf=True))
        psf, ps = hermite_propagate(mono, wavelengths, weights, n_nodes)
        return dl.PSF(psf, ps) if return_psf else psf


class NICMOSOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, psf_oversample=1, n_zernikes = 26):
        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,4+n_zernikes),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.zeros(n_zernikes)#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431/psf_oversample,
            oversample
        )


class NICMOSFresnelOptics(dl.AngularOpticalSystem):
    defocus: np.ndarray
    fnumber: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, fnumber, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.fnumber = fnumber

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf *= layer

        u_in = wf.phasor

        fl = self.fnumber*self.diameter
        abcd = compose_abcd([abcd_lens(fl), abcd_free_space(fl + self.defocus)])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop_basic(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(N_out, N_out*dx_out, wavelength).set(
            ["amplitude", "phase"], [np.abs(u_out), np.angle(u_out)]
        )

        if return_wf:
            return wf
        return wf.psf

def abcd_magnification(m):
    return np.array([[m, 0.], [0., 1/m]])

class NICMOSSecondaryFresnelOptics(dl.AngularOpticalSystem):
    defocus: np.ndarray
    despace: np.ndarray
    mag: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, despace, mag, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.despace = despace
        self.mag = mag

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf *= layer

        u_in = wf.phasor

        abcd = compose_abcd([
            abcd_lens(5.52085),
            abcd_free_space(4.907028205 + self.despace),
            abcd_lens(-0.6790325),
            abcd_free_space(6.3919974 + self.despace + self.defocus),
            abcd_magnification(self.mag),
        ])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop_basic(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(N_out, N_out*dx_out, wavelength).set(
            ["amplitude", "phase"], [np.abs(u_out), np.angle(u_out)]
        )

        if return_wf:
            return wf
        return wf.psf


class NICMOSDistortedOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, distortion_orders=5, n_zernikes = 26):

        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=DistortedCoords(order=distortion_orders),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=DistortedCoords(order=distortion_orders), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=np.pi/4)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,4+n_zernikes),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.zeros(n_zernikes),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431,
            oversample
        )
    #def apply(self, wavefront):


