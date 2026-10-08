import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
import jax
import numpy as onp

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from abc import abstractmethod

from apertures import *
from detectors import *
from spectra import *
from filters import *
from stats import gauss_log_likelihood
#from vis_models import LogVisModel


class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    mjd: str = eqx.field(static=True)
    exptime: str = eqx.field(static=True)
    wcs: object = eqx.field(static=True)
    hdr: object = eqx.field(static=True)
    pam: Array#object = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array
    orient: Array


    fit: object# = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, err, bad, fit, mjd, exptime, wcs, pam, orient, hdr):
        """
        Initialise exposure
        """
        self.filename = filename
        self.target = name
        self.filter = filter
        self.data = data
        self.err = err
        self.bad = bad

        self.mjd = mjd

        self.fit = fit
        self.exptime = exptime
        self.wcs = wcs
        self.pam = pam
        self.orient = orient
        self.hdr = hdr
    
    def get_key(self, param):
        return self.fit.get_key(self, param)

    def map_param(self, param):
        return self.fit.map_param(self, param)
    
    @property
    def key(self):
        return self.filename

class BlankExposure(Exposure):
    def __init__(self, name, filter, fit):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        self.data = 0.
        self.err = 0.
        self.bad = 0.
        self.exptime = 0.
        self.pam = 0.
        self.orient=0.
        self.hdr = None


class InjectedExposure(Exposure):
    def __init__(self, name, filter, fit, model, t_exp, n_exp, read_noise=10.):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        gain = 3

        generated_data = self.fit(model, self) * t_exp * gain

        err = np.sqrt(generated_data/(gain*t_exp) + read_noise**2)/np.sqrt(n_exp)

        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data

        #err = np.sqrt(data/(gain*exptime) + err**2)

        self.data = data/t_exp/gain
        self.err = err/t_exp/gain
        self.bad = np.zeros(self.data.shape)

        self.exptime = t_exp
        self.pam = 0.
        self.orient = 0.
        self.hdr = None

def exposure_from_file(fname, fit, extra_bad=None, crop=None, flatcorr=None):

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)

    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    detector_mask = np.full((256, 256), False, dtype=bool).at[127:130, :].set(True)#.at[:, 127:130].set(True)

    if flatcorr:
        flatfile = flatcorr + hdr["FLATFILE"][5:] 
        holeflat = fits.getdata(flatfile, 1)[200:230, 60:90]

        thresh = 0.008
        hole_map = (np.abs((holeflat[:,1:]-holeflat[:,:-1]))[:-1,:]<thresh) & (np.abs((holeflat[1:,:]-holeflat[:-1,:]))[:,:-1]<thresh)

        detector_mask = detector_mask.at[200:229, 60:89].set(hole_map)

    bad = np.asarray((err==0.0) | (info&0b1111111111 != 0) | detector_mask)
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    wcs = WCS(image_hdr)


    pam = hdr['NPFOCUSP']

    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']

    exptime = float(hdr['EXPTIME'])
    orient = float(hdr["ORIENTAT"])

    mjd = hdr['EXPSTART']

    if crop:
        w = WCS(image_hdr)
        centre = SkyCoord(w.pixel_to_world(256-181,256-44), unit='deg')
        data = Cutout2D(data, centre, crop, wcs=w).data
        err = Cutout2D(err, centre, crop, wcs=w).data
        info = Cutout2D(info, centre, crop, wcs=w).data

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    # bad = np.asarray((err==0.0) | (info>0.))

    if extra_bad is not None:
        bad = bad | extra_bad

    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    err_with_poisson = err#np.sqrt(data/(gain*exptime) + err**2)

    bad_with_poisson = np.isnan(err_with_poisson)

    return Exposure(filename, name, filter, data, err_with_poisson, bad_with_poisson, fit, mjd, exptime, wcs, pam, orient, hdr)

def _nm(x):
    return x*1e-9


def _jitter_grid(n=3):
    """Gauss-Hermite offsets (units of sigma) and weights for a unit 2D Gaussian."""
    x, w = onp.polynomial.hermite_e.hermegauss(n)
    nodes = onp.stack(onp.meshgrid(x, x), -1).reshape(-1, 2)
    weights = onp.outer(w, w).ravel()
    return np.array(nodes), np.array(weights/weights.sum())


def _occulter(x):
    """Occulter size in units of 0.3 arcsec (at 24 * 2.4 focal ratio) to metres."""
    return x*dlu.arcsec2rad(0.3)*24*2.4


# Unit of primary_amp (log-amplitude): the field perturbation of 1 nm OPD at 1.87 um, so primary_amp
# and primary_opd steps of the same size (e.g. the same adam learning rate) are comparable.
AMP_UNIT = 2*np.pi*1e-9/1.87e-6


def _no_piston(x):
    """Zero the [0, 0] Fourier coefficient: a piston is a no-op (OPD) or degenerate with flux (amplitude)."""
    return x.at[0, 0].set(0.)


# Parameters that set the same transformed value at one or more optics paths:
# (param, optics paths, transform of the param value). The rest of update_optics is special-cased.
OPTICS_PARAMS = [
    ("primary_opd", ["primary_opd.coefficients"], lambda x: _no_piston(_nm(x))),
    ("primary_amp", ["primary_amp.coefficients"], lambda x: _no_piston(x*AMP_UNIT)),
    ("primary_klip", ["primary_klip.coefficients"], lambda x: x),
    ("primary_low", ["primary_low.coefficients"], _nm),
    ("primary_tilt", ["primary_tilt.angles"], dlu.arcsec2rad),
    ("cold_mask_tilt", ["cold_mask_tilt.angles"], dlu.arcsec2rad),
    ("cold_mask_opd", ["cold_mask_opd.coefficients"], _nm),
    ("cold_mask_shift", ["cold_mask.transformation.translation",
                         "cold_mask_opd.aperture.transformation.translation"], lambda x: x*1e-2),
    ("outer_radius", ["cold_mask.outer.radius", "cold_mask_opd.aperture.radius"], lambda x: x),
    ("secondary_radius", ["cold_mask.secondary.radius"], lambda x: x),
    ("spider_width", ["cold_mask.spider.width"], lambda x: x),
    ("primary_spider", ["primary.spider.width"], lambda x: x),
    ("primary_secondary", ["primary.secondary.radius"], lambda x: x),
    ("primary_outer", ["primary.mirror.radius", "primary_low.aperture.radius"], lambda x: x),
    ("primary_pad", ["primary.pad_1.radius", "primary.pad_2.radius", "primary.pad_3.radius"], lambda x: x),
    ("cold_mask_shear", ["cold_mask.transformation.shear"], lambda x: x),
    ("cold_mask_scale", ["cold_mask.transformation.compression",
                         "cold_mask_opd.aperture.transformation.compression"], lambda x: x),
    ("cold_mask_rot", ["cold_mask.transformation.rotation",
                       "cold_mask_opd.aperture.transformation.rotation"], lambda x: dlu.deg2rad(x)+np.pi/4),
    ("primary_rot", ["primary.transformation.rotation"], lambda x: dlu.deg2rad(x)+np.pi/4),
    ("primary_shear", ["primary.transformation.shear", "primary_low.aperture.transformation.shear"], lambda x: x),
    ("occulter_radius", ["occulter.layers.occulter.r"], _occulter),
    ("fnumber", ["prop1.focal_length"], lambda x: x*2.4),
    ("defocus1", ["prop1.FreeSpace.distance"], lambda x: x),
    ("defocus2", ["prop2.FreeSpace.distance"], lambda x: x),
    ("defocus3", ["prop3.FreeSpace.distance"], lambda x: x),
]


class ModelFit(zdx.Base):
    source: dl.Telescope

    # Parameter -> how its leaves are keyed in the model params: "exposure" (one per exposure),
    # "target" (one per target and filter) or "global" (shared by all exposures). Subclasses extend it.
    PARAM_KEYS = {
        "primary_low": "exposure", "primary_tilt": "exposure", "primary_klip": "exposure", "bias": "exposure",
        "primary_opd": "exposure", "primary_amp": "global", "primary_rot": "global",
        "primary_shear": "global",
        "cold_mask_opd": "global", "cold_mask_tilt": "global", "cold_mask_shift": "global",
        "cold_mask_rot": "global", "cold_mask_shear": "global", "cold_mask_scale": "global",
        "jitter": "exposure",
    }

    @abstractmethod
    def update_source(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        if param not in self.PARAM_KEYS:
            raise ValueError(f"Parameter {param} has no key")
        match self.PARAM_KEYS[param]:
            case "exposure": return exposure.key
            case "target": return f"{exposure.target}_{exposure.filter}"
            case "global": return "global"

    def map_param(self, exposure, param):
        if param in self.PARAM_KEYS:
            return f"{param}.{exposure.get_key(param)}"
        return param

    def update_optics(self, model, exposure):
        optics = model.optics
        if "occulter_coeffs" in model.params.keys():
            coeffs = model.get(self.map_param(exposure, "occulter_coeffs"))*dlu.arcsec2rad(0.3)*24*2.4
            optics = optics.set("occulter.layers.occulter.cc", coeffs[::2])
            optics = optics.set("occulter.layers.occulter.ss", coeffs[1::2])

        for name, paths, transform in OPTICS_PARAMS:
            if name in model.params.keys():
                value = transform(model.get(self.map_param(exposure, name)))
                for path in paths:
                    optics = optics.set(path, value)

        return optics

    def update_detector(self, model, exposure):
        detector = model.detector

        if "bias" in model.params.keys():
            bias = model.get(self.map_param(exposure, "bias"))
            detector = detector.set("bias.value", bias)
        
        if "anisotropy" in model.params.keys():
            anisotropy = model.get(self.map_param(exposure, "anisotropy"))
            detector = detector.set("resample.anisotropy", anisotropy)
        return detector

    def __call__(self, model, exposure):
        source = self.update_source(model, exposure)
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        def model_psf(offset):
            # Pointing offset as extra tilt at the primary, i.e. upstream of the occulter
            tilted = optics.set("primary_tilt.angles", optics.primary_tilt.angles + offset)
            psfs = tilted.model(source, return_psf=True)
            return psfs.data.sum(tuple(range(psfs.ndim))), psfs.pixel_scale.mean()

        if "jitter" in model.params.keys():
            # jitter is in mas (cf. primary_tilt, which is in arcsec)
            sigma = dlu.arcsec2rad(1e-3*np.abs(model.get(self.map_param(exposure, "jitter"))))
            nodes, weights = _jitter_grid()
            psfs, pixel_scales = jax.lax.map(model_psf, sigma*nodes)
            psf, pixel_scale = np.tensordot(weights, psfs, 1), pixel_scales[0]
        else:
            psf, pixel_scale = model_psf(np.zeros(2))

        return detector.model(dl.PSF(psf, pixel_scale), return_psf=False)
    
    def loglike(self, model, exposure, per_pix=False, return_im=False):
        psf = self(model, exposure)

        data = exposure.data
        err = exposure.err
        bad = exposure.bad
        err = np.where(bad, 1., err)

        # add excess noise in quadrature
        if "quadrature" in model.params.keys():
            quad_error = 10**model.get(self.map_param(exposure, "quadrature"))
            err = err*quad_error#np.sqrt(err**2 + quad_error**2 + 1e-10)        

        posterior_im = gauss_log_likelihood(psf, (data, err, bad))
        if return_im:
            return posterior_im
        
        if per_pix:
            return np.nanmean(posterior_im)
        return np.nansum(posterior_im)
        
        

class SinglePointFit(ModelFit):
    PARAM_KEYS = ModelFit.PARAM_KEYS | {"positions": "exposure", "spectrum": "target"}
    #nwavels: int = eqx.field(static=True)
    #spectrum: CombinedSpectrum
    time_series: bool = eqx.field(static=True)

    def __init__(self, spectrum_basis, filter, time_series=False):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.source = dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))
        self.time_series=time_series
    
    def get_key(self, exposure, param):
        if self.time_series and param == "spectrum":
            return exposure.key
        return super().get_key(exposure, param)

    def update_source(self, model, exposure):
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("spectrum.basis_weights", spectrum_coeffs)
        source = source.set("flux", source.spectrum.flux)
        source = source.set("position", np.zeros(2))#model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        
        return source    




# %%
def L1_loss(arr):
    """L1 norm loss for array-like inputs."""
    return np.nansum(np.abs(arr))


def L2_loss(arr):
    """L2 (quadratic) loss for array-like inputs."""
    return np.nansum(arr**2)


def tikhinov(arr):
    """Finite-difference approximation used by several regularisers."""
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :], axis=1)
    dy = np.diff(pad_arr[:, 0:-1], axis=0)
    return dx**2 + dy**2


def TV_loss(arr, eps=1e-16):
    """Total variation (approx.) loss computed from finite differences."""
    return np.sqrt(tikhinov(arr) + eps**2).sum()


def TSV_loss(arr):
    """Total squared variation (quadratic) loss."""
    return tikhinov(arr).sum()


def ME_loss(arr, eps=1e-16):
    """Maximum-entropy inspired loss (negative entropy of distribution)."""
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S

# %%
class CursedResolvedSource(dl.sources.Source):
    distribution: Array
    position: Array
    pitch: float
    roll: Array

    def __init__(self, distribution, pitch, position=np.zeros(2), roll=0., **kwargs):
        self.distribution = distribution
        self.pitch = float(pitch)
        self.position = position
        self.roll = roll
        super().__init__(**kwargs)
    
    def normalise(self):
        return self
    
    def model(self, optics, return_wf=False, return_psf=False):
        R, TH = dlu.pixel_coords(self.distribution.shape[0], pixel_scale=self.pitch, polar=True)
        coords = dlu.polar2cart(np.array([R, TH+self.roll]))
        # coords = dlu.nd_coords(self.distribution.shape, self.pitch, self.position)
        xs = coords[0].flatten()
        ys = coords[1].flatten()
        ds = self.distribution.flatten()


        conv_psf = np.sum(
            jax.lax.map(
                lambda x: x[2]*jax.lax.stop_gradient(optics.propagate(self.wavelengths, np.array([x[0], x[1]]), self.weights)),
                np.stack((xs, ys, ds)).T,
                batch_size=256,
            ), 
            axis=0
        )

        wf = optics.propagate(self.wavelengths, np.array([xs.mean(), ys.mean()]), self.weights, return_wf=True)
        if return_psf:
            return dl.PSF(conv_psf, wf.pixel_scale.mean())
        return conv_psf

# %%
class InterpolatedResolvedSource(dl.sources.Source):
    """Cheap stand-in for CursedResolvedSource: K anchor PSFs instead of one PSF per source pixel.

    The PSF of a source at p is approximated by a weighted sum of shifted anchor PSFs,
        PSF(p) ~ sum_k w_k(p) * shift(PSF(a_k), p - a_k),     sum_k w_k = 1,
    so the image is sum_k PSF(a_k) (*) S_k, with S_k the deltas {w_k(p_j) d_j} at (p_j - a_k) / pixel_scale,
    placed at exact sub-pixel positions (Fourier-domain shifts).
    The anchors sit on a polar grid in the detector frame (a centre point plus n_angular points on each
    ring of anchor_radii, in arcsec), densest near the occulter where the PSF varies fastest; the weights
    are bilinear in (r, theta). The only approximation is the shift-invariance of the PSF between neighbouring
    anchors (exact at anchors). The image is linear in
    `distribution`, so its gradient is exact for the approximate model. Cost: K propagations + K FFTs.

    Hybrid: pixels with radius < exact_radius (arcsec) are propagated exactly, one PSF each, because the PSF
    varies too fast there for interpolation. Radius does not depend on roll, so these pixels are a static set;
    the anchor rings then start at exact_radius (no centre anchor). exact_radius=0 gives pure interpolation.

    Assumes the last layer of `optics` is the MFT onto the detector grid (NICMOSCoronagraph's "prop1"), whose
    npixels / pixel_scale / focal_length define the grid; a source offset (x, y) moves the PSF by (y, x)
    pixels about the grid centre (N - 1) / 2. Source positions are rolled as in CursedResolvedSource.
    """
    distribution: Array
    position: Array
    pitch: float = eqx.field(static=True)    # static: the inner/outer pixel split below is fixed at trace time
    roll: Array
    anchor_radii: tuple = eqx.field(static=True)
    n_angular: int = eqx.field(static=True)
    include_centre: bool = eqx.field(static=True)
    grad_optics: bool = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    exact_radius: float = eqx.field(static=True)

    def __init__(self, distribution, pitch, position=np.zeros(2), roll=0.,
                 anchor_radii=(0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.85, 1.1, 1.5, 2.2), n_angular=8,
                 include_centre=True, grad_optics=False, batch_size=4, exact_radius=0.6, **kwargs):
        self.distribution = distribution
        self.pitch = float(pitch)
        self.position = position
        self.roll = roll
        self.anchor_radii = tuple(float(r) for r in anchor_radii)
        self.n_angular = int(n_angular)
        self.include_centre = bool(include_centre)
        self.grad_optics = bool(grad_optics)
        self.batch_size = int(batch_size)
        self.exact_radius = float(exact_radius)
        super().__init__(**kwargs)

    def normalise(self):
        return self

    def layout(self):
        """(centre anchor?, ring radii in arcsec). With an exact core, rings start at exact_radius."""
        if self.exact_radius > 0:
            return False, (self.exact_radius,) + tuple(r for r in self.anchor_radii if r > self.exact_radius)
        return self.include_centre, self.anchor_radii

    def anchors(self):
        """Anchor offsets (K, 2) in radians, (x, y); the centre anchor (if any) comes first."""
        centre, radii = self.layout()
        ang = np.arange(self.n_angular) * 2 * np.pi / self.n_angular
        ring = np.array([[r * np.cos(t), r * np.sin(t)] for r in radii for t in ang])
        a = np.concatenate([np.zeros((1, 2)), ring]) if centre else ring
        return dlu.arcsec2rad(a)

    def weights_matrix(self, r, theta):
        """(K, M) interpolation weights for sources at polar position (r, theta): linear in r between rings
        (centre to first ring included, constant beyond the last), periodic-linear in theta."""
        centre, radii = self.layout()
        nodes = np.array(([0.] if centre else []) + [dlu.arcsec2rad(x) for x in radii])
        w_r = jax.vmap(lambda e: np.interp(r, nodes, e), out_axes=1)(np.eye(len(nodes)))   # (M, nodes)
        n = self.n_angular
        u = np.mod(theta, 2 * np.pi) * n / (2 * np.pi)
        j, f = np.floor(u).astype(int) % n, u % 1
        w_t = (1 - f)[:, None] * jax.nn.one_hot(j, n) + f[:, None] * jax.nn.one_hot((j + 1) % n, n)
        if centre:
            w_r, w_c = w_r[:, 1:], w_r[:, :1]
        w = (w_r[:, :, None] * w_t[:, None, :]).reshape(len(r), -1)
        return (np.concatenate([w_c, w], axis=1) if centre else w).T

    def model(self, optics, return_wf=False, return_psf=False):
        final = optics.layers["prop1"]
        N, ps = final.npixels, final.pixel_scale / final.focal_length   # output grid, radians / pixel
        wid = self.distribution.shape[0]
        R, TH = dlu.pixel_coords(wid, pixel_scale=self.pitch, polar=True)
        TH = TH + self.roll
        pos = np.stack(dlu.polar2cart(np.array([R, TH])), -1).reshape(-1, 2)    # (M, 2) source (x, y)
        d = self.distribution.flatten()
        # inner (exact) / outer (interpolated) split, in numpy so that it is concrete under jit (pitch is static)
        xs = (onp.arange(wid) - (wid - 1) / 2) * self.pitch
        inner = onp.flatnonzero(onp.hypot(*onp.meshgrid(xs, xs)).flatten() < dlu.arcsec2rad(self.exact_radius))
        outer = onp.setdiff1d(onp.arange(wid ** 2), inner)
        image = np.zeros((N, N))

        if len(inner):
            sg = (lambda x: x) if self.grad_optics else jax.lax.stop_gradient
            exact = lambda x: x[2] * sg(optics.propagate(self.wavelengths, x[:2], self.weights))
            xyd = np.concatenate([pos[inner], d[inner, None]], axis=1)
            image += np.sum(jax.lax.map(exact, xyd, batch_size=self.batch_size), axis=0)

        if len(outer):
            W = self.weights_matrix(R.flatten()[outer], TH.flatten()[outer])
            image += np.sum(jax.lax.map(self._anchor_image(optics, pos[outer], d[outer], N, ps),
                                        (self.anchors(), W), batch_size=self.batch_size), axis=0)
        if return_psf:
            return dl.PSF(image, final.pixel_scale)
        return image

    def _anchor_image(self, optics, pos, d, N, ps):
        """Returns the function mapping (anchor offset, its weights over the outer pixels) to its image term."""
        fy, fx = np.fft.fftfreq(2 * N), np.fft.rfftfreq(2 * N)

        def one(args):
            a, w = args
            psf = optics.propagate(self.wavelengths, a, self.weights)
            psf = psf if self.grad_optics else jax.lax.stop_gradient(psf)
            # S_k in the Fourier domain, exactly: sum_j c_j exp(-2 pi i f . t_j) with t_j the (row, col) pixel shift
            # of source j from the anchor (a separable matmul; no sub-pixel interpolation error). A shift beyond
            # the window contributes nothing to it and would wrap around, so it is dropped.
            t = ((pos - a) / ps)[:, ::-1]
            c = w * d * np.all(np.abs(t) < N, axis=1)
            A = np.exp(-2j * np.pi * fy[:, None] * t[:, 0]) * c                 # (2N, M)
            B = np.exp(-2j * np.pi * t[:, 1, None] * fx)                        # (M, N + 1)
            # zero padding to 2N keeps the circular convolution free of wrap-around inside the N x N window
            return np.fft.irfft2(np.fft.rfft2(psf, (2 * N, 2 * N)) * (A @ B), (2 * N, 2 * N))[:N, :N]

        return one

# %%
class PointResolvedFit(ModelFit):
    PARAM_KEYS = ModelFit.PARAM_KEYS | {"positions": "exposure", "spectrum": "target", "resolved": "target"}
    wid: float
    regulariser: Array

    def __init__(self, spectrum_basis, filter, wid, regulariser=np.zeros(2), resolved_source="exact", resolved_kwargs=None):
        """resolved_source: "exact" (CursedResolvedSource, one PSF per pixel) or "interp"
        (InterpolatedResolvedSource(**resolved_kwargs), anchor PSFs + interpolation)."""
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)

        wvr, intenr = calc_throughput(filter, 1)
        resolved_cls = {"exact": CursedResolvedSource, "interp": InterpolatedResolvedSource}[resolved_source]

        self.source = dl.Scene([            
            ("resolved", resolved_cls(
                wavelengths=wvr,
                spectrum=dl.Spectrum(wvr, intenr), 
                distribution=np.ones((wid, wid)),
                pitch=dlu.arcsec2rad(0.0432*2),
                **(resolved_kwargs or {}),
            )),
            ("point", dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))),
        ])
        self.wid = wid
        self.regulariser=regulariser
    
    def get_distribution(self, model, exposure):
        return 10**(model.get(exposure.fit.map_param(exposure, "resolved")))

    def update_source(self, model, exposure):
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("point.spectrum.basis_weights", spectrum_coeffs)
        source = source.set("point.flux", source.point.spectrum.flux)        

        distribution = self.get_distribution(model, exposure)

        source = source.set("resolved.distribution",  distribution)
        source = source.set("resolved.roll", -np.deg2rad(exposure.orient))
        
        return source

    def loglike(self, model, exposure, per_pix=False, return_im=False):


        if "resolved" in model.params.keys():
            dist = self.get_distribution(model, exposure)
            return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im) + self.regulariser[0]* L2_loss(dist) +  self.regulariser[1]*TSV_loss(dist)
        
        return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im)



class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values

class NICMOSModel(BaseModeller):
    optics: NICMOSOptics
    detector: NICMOSDetector

    def __init__(self, exposures, params, optics, detector):
        self.optics = optics
        self.detector = detector
        self.params = params


class ModelParams(BaseModeller):

    def __getitem__(self, key):
        return self.params[key]

    def __getattr__(self, key):

        # Make the object act like a real dictionary
        if hasattr(self.params, key):
            return getattr(self.params, key)

        if key in self.params.keys():
            return self.params[key]

        for sub_key, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)

        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys()]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys()]))

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def map(self, fn):
        return jax.tree.map(lambda x: fn(x), self)

    # Re-name this donate, and it counterpart accept, receive?
    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(list(self.keys()), list(self.values()))

    def partition(self, params):
        """params can be a model params object or a list of keys"""
        if isinstance(params, ModelParams):
            params = list(params.params.keys())
        return (
            ModelParams({param: self[param] for param in params}),
            ModelParams({param: self[param] for param in self.keys() if param not in params}),
        )

    def combine(self, params2):
        return ModelParams({**self.params, **params2.params})

    def jacfwd(self, fn, n_batch=1):
        X, unravel_fn = ravel_pytree(self)
        Xs = np.array_split(X, n_batch)
        rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
        lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
        starts = np.concatenate([np.array([0]), lens])

        @eqx.filter_jacfwd
        def batched_jac_fn(x, index):
            model_params = unravel_fn(rebuild(x, index))
            return eqx.filter_jit(fn)(model_params)

        return np.concatenate([batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1)
