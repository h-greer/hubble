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
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330*1.2,
                occulting = True,
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
                    translation = (-0.4615*1.2, -0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, 0.7606*1.2),
                ),
                softening=self.softening
            )
        }



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

class TransformedFourierBasis(dl.OpticalLayer):
    mean: Array
    modes: Array
    coefficients: Array
    kernels: tuple[Array, Array]
    n_modes: Array

    def __init__(self, npix, n_modes, basisfile):
        basis = np.load(basisfile)
        self.mean = basis["mean"]
        self.modes = basis["modes"]
        self.n_modes = n_modes
        self.kernels = dlu.fourier_kernels((n_modes,n_modes), (npix,npix))
        self.coefficients = np.zeros(self.modes.shape[0])
    
    def eval_basis(self):
        fourier_coeffs = 1e-9* (self.mean + np.dot(self.coefficients, self.modes)).reshape((self.n_modes,self.n_modes))
        return dlu.eval_fourier_basis(fourier_coeffs, *self.kernels)

    def __call__(self, wavefront):
        return wavefront.add_opd(self.eval_basis())



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
    """
    wl_batch: int | None = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, wf_npixels, psf_npixels, oversample, n_modes=12, n_zernikes=1., turboklip=None, wl_batch=None, remat=False, amplitude=False):
        diameter = 3.
        layers = [
            ("primary",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=5)),

            ("primary_tilt", dl.Tilt(angles=(0.,0.))),

            ("primary_opd", dl.FourierBasis(wf_npixels, n_modes=n_modes)),
        ]

        if amplitude:
            layers += [("primary_amp", FourierAmplitude(wf_npixels, n_modes=n_modes))]

        if turboklip:
            layers += [
                ("primary_klip", TransformedFourierBasis(wf_npixels, n_modes, turboklip)),
            ]
        
        layers += [
            ("primary_low", ScanAberratedAperture(
                    dl.layers.CircularAperture(1.2,transformation=dl.CoordTransform(translation=np.zeros(2)),),
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


            ("cold_mask",   NIC2ColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=5, normalise=False)),

            ("cold_mask_opd", ScanAberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform(translation=np.asarray((-0.05, -0.05)))),
                    noll_inds=np.arange(4,20),
                    coefficients = np.zeros(1),
                )),

            ("cold_mask_tilt", dl.Tilt(angles=(0.,0.))),
            
            ("prop1", dl.MFT(psf_npixels*oversample, focal_length=45.7*2.4, pixel_scale=40e-6/oversample)),
        ]

        super().__init__(wf_npixels, diameter, layers)
        self.wl_batch = wl_batch
        self.remat = bool(remat)

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

    def propagate(self, wavelengths, offset=None, weights=None, return_wf=False, return_psf=False):
        if return_wf:                                   # rarely needed: keep the stock dLux path
            return super().propagate(wavelengths, offset, weights, return_wf, return_psf)
        wavelengths = np.atleast_1d(wavelengths)
        weights = np.ones_like(wavelengths) / len(wavelengths) if weights is None else np.atleast_1d(weights)
        offset = np.zeros(2) if offset is None else np.asarray(offset)

        static = eqx.filter_checkpoint(NICMOSCoronagraph._static_stage)(self, offset)
        one = lambda s, st, wl, w: s._one_wavelength(st, wl, w)
        if self.remat:
            one = eqx.filter_checkpoint(one)
        if self.wl_batch is None:
            psfs, ps = eqx.filter_vmap(lambda wl, w: one(self, static, wl, w))(wavelengths, weights)
        else:
            psfs, ps = jax.lax.map(lambda x: one(self, static, *x), (wavelengths, weights), batch_size=self.wl_batch)
        psf = psfs.sum(0)
        return dl.PSF(psf, ps.mean()) if return_psf else psf


class NICMOSFresnelCoronagraph(dl.LayeredOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, n_modes=12, n_zernikes=1., turboklip=None):
        diameter = 3.
        layers = [
            ("primary",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=2)),

            ("primary_tilt", dl.Tilt(angles=(0.,0.))),

            ("primary_opd", dl.FourierBasis(wf_npixels, n_modes=n_modes)),
        ]

        if turboklip:
            layers += [
                ("primary_klip", TransformedFourierBasis(wf_npixels, n_modes, turboklip)),
            ]
        
        layers += [
            ("primary_low", dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                    # transformation = dl.CoordTransform(rotation=0.),
                )),

            ("flip", dl.Flip(axes=(0,1))),


            ("prop1", dl.FFTPropagator(
                [
                    ("ThinLens", dl.ABCDConjugatePlane(24*2.4)),
                    ("FreeSpace", dl.ABCDFreeSpace(0.)),
                ],
                dl.PadSpec(pad=2, crop=1)
            )),

            ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3)*24*2.4, np.zeros(1), np.zeros(1), invert=True)),

            ("prop2", dl.FFTPropagator(
                [
                    ("FreeSpace", dl.ABCDFreeSpace(0.)),
                    ("ThinLens", dl.ABCDConjugatePlane(24*2.4)),
                ],
                dl.PadSpec(pad=1, crop=1)#d=40e-6*psf_npixels/wf_npixels)
            )),

            ("resize", dl.Resize(wf_npixels)),

            ("cold_mask",   NIC2ColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),

            ("cold_mask_opd", dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform(translation=np.asarray((-0.05, -0.05)))),
                    noll_inds=np.arange(5,20),
                    coefficients = np.zeros(15),
                )),

            ("cold_mask_tilt", dl.Tilt(angles=(0.,0.))),

            ("prop3", dl.MFTPropagator(
                [
                    ("ThinLens", dl.ABCDConjugatePlane(45.7*2.4)),
                    ("FreeSpace", dl.ABCDFreeSpace(0.)),
                ],
                dl.CoordSpec(n=psf_npixels*oversample, d=40e-6/oversample)
            )),
        ]

        super().__init__(wf_npixels, diameter, layers)



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


