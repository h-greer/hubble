import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
import jax

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
    pam: Array#object = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array
    orient: Array


    fit: object = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, err, bad, fit, mjd, exptime, wcs, pam, orient):
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

class LoadedExposure(Exposure):
    def __init__(self, name, filter, fit, data, err, bad):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        self.data = data
        self.err = err
        self.bad = bad
        self.exptime = 0.
        self.pam = 0.
        self.orient = 0.

def exposure_from_file(fname, fit, extra_bad=None, crop=None):

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)

    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    detector_mask = np.full((256, 256), False, dtype=bool).at[127:130, :].set(True)#.at[:, 127:130].set(True)

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32) | detector_mask)
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    wcs = WCS(image_hdr)


    pam = hdr['NPFOCUSP']

    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']

    exptime = float(hdr['EXPTIME'])
    gain = float(hdr['ADCGAIN'])
    orient = float(hdr["ORIENTAT"])
    print(exptime, gain)

    mjd = hdr['EXPSTART']

    print(hdr["CAL_VER"])

    print(hdr["ORIENTAT"])

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

    err_with_poisson = np.sqrt(data/(gain*exptime) + err**2)

    bad_with_poisson = np.isnan(err_with_poisson)

    return Exposure(filename, name, filter, data, err_with_poisson, bad_with_poisson, fit, mjd, exptime, wcs, pam, orient)

class ModelFit(zdx.Base):
    source: dl.Telescope

    @abstractmethod
    def update_source(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "primary_low" | "primary_tilt":
                return exposure.key            
            case "primary_opd" | "cold_mask_opd" | "cold_mask_tilt":
                return "global"
            case "cold_mask_shift" | "cold_mask_rot" | "cold_mask_shear" | "cold_mask_scale":
                return "global"
            case "bias":
                return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        if param in ["primary_opd", "primary_low", "cold_mask_opd", "primary_tilt", "cold_mask_tilt", "cold_mask_shift", "cold_mask_rot", "cold_mask_shear", "cold_mask_scale", "bias"]:
            return f"{param}.{exposure.get_key(param)}"
        return param
    
    def update_optics(self, model, exposure):
        optics = model.optics
        if "primary_opd" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "primary_opd"))*1e-9
            coefficients = coefficients.at[0,0].set(0.)
            optics = optics.set("primary_opd.coefficients", coefficients)
        
        if "primary_low" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "primary_low"))*1e-9
            optics = optics.set("primary_low.coefficients", coefficients)
        
        if "primary_tilt" in model.params.keys():
            angles = dlu.arcsec2rad(model.get(self.map_param(exposure, "primary_tilt")))
            optics = optics.set("primary_tilt.angles", angles)

        if "cold_mask_tilt" in model.params.keys():
            angles = dlu.arcsec2rad(model.get(self.map_param(exposure, "cold_mask_tilt")))
            optics = optics.set("cold_mask_tilt.angles", angles)
        
        if "cold_mask_opd" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "cold_mask_opd"))*1e-9
            optics = optics.set("cold_mask_opd.coefficients", coefficients)
        
        if "cold_mask_shift" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_shift"))*1e-2
            optics = optics.set("cold_mask.transformation.translation", translation)
            optics = optics.set("cold_mask_opd.aperture.transformation.translation", translation)
        
        if "cold_mask_shear" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_shear"))
            optics = optics.set("cold_mask.transformation.shear", translation)
            optics = optics.set("cold_mask_opd.aperture.transformation.shear", translation)

        if "cold_mask_scale" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_scale"))
            optics = optics.set("cold_mask.transformation.compression", translation)
            optics = optics.set("cold_mask_opd.aperture.transformation.compression", translation)
        
        if "cold_mask_rot" in model.params.keys():
            translation = dlu.deg2rad(model.get(self.map_param(exposure, "cold_mask_rot")))+np.pi/4
            optics = optics.set("cold_mask.transformation.rotation", translation)
            optics = optics.set("cold_mask_opd.aperture.transformation.rotation", translation)

        if "occulter_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "occulter_radius"))*dlu.arcsec2rad(0.3)*24*2.4
            optics = optics.set("occulter.layers.occulter.r", radius)
        
        if "occulter_coeffs" in model.params.keys():
            coeffs = model.get(self.map_param(exposure, "occulter_coeffs"))*dlu.arcsec2rad(0.3)*24*2.4
            optics = optics.set("occulter.layers.occulter.cc", coeffs[::2])
            optics = optics.set("occulter.layers.occulter.ss", coeffs[1::2])
        
        if "fnumber" in model.params.keys():
            fnumber = model.get(self.map_param(exposure, "fnumber"))
            optics = optics.set("prop1.focal_length", fnumber*2.4)

        return optics

    def update_detector(self, model, exposure):
        detector = model.detector

        if "bias" in model.params.keys():
            bias = model.get(self.map_param(exposure, "bias"))
            detector = detector.set("bias.value", bias)
        if "jitter" in model.params.keys():
            jitter = model.get(self.map_param(exposure, "jitter"))
            detector = detector.set("jitter.sigma", np.abs(jitter))
        return detector

    def __call__(self, model, exposure):
        source = self.update_source(model, exposure)
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
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
    #nwavels: int = eqx.field(static=True)
    #spectrum: CombinedSpectrum
    time_series: bool = eqx.field(static=True)

    def __init__(self, spectrum_basis, filter, time_series=False, precombined=False, wavels=None):
        nwavels, nbasis = spectrum_basis.shape
        if precombined:
            self.source = dl.PointSource(spectrum=PreCombinedBasisSpectrum(wavels, np.zeros(nbasis), spectrum_basis))
        else:
            wv, inten = calc_throughput(filter, nwavels)
            self.source = dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))
        self.time_series=time_series
    
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "spectrum" or param == "flux":
            if self.time_series:
                return exposure.key
            else:    
                return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("spectrum.basis_weights", spectrum_coeffs)
        source = source.set("flux", source.spectrum.flux*exposure.exptime)
        source = source.set("position", np.zeros(2))#model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        
        return source    



# class SpectrumVisFit(ModelFit):
#     vis_model: LogVisModel
#     def __init__(self, spectrum, nwavels, vis_model):
#         super().__init__(spectrum, nwavels)
#         self.vis_model = vis_model

#     def get_key(self, exposure, param):
#         if param == "phases":
#             return exposure.key
#         elif param == "amplitudes":
#             return exposure.key
#         else:
#             return super().get_key(exposure, param)
    
#     def map_param(self, exposure, param):
#         if param == "phases":
#             return f"{param}.{exposure.get_key(param)}"
#         elif param == "amplitudes":
#             return f"{param}.{exposure.get_key(param)}"
#         else:
#             return super().map_param(exposure, param)


#     def __call__(self, model, exposure):

#         source = self.update_source(model, exposure)
#         optics = self.update_optics(model, exposure)
#         detector = self.update_detector(model, exposure)

#         wfs = optics.model(source, return_wf=True)

#         phases = model.get(exposure.fit.map_param(exposure, "phases"))
#         amplitudes = model.get(exposure.fit.map_param(exposure, "amplitudes"))

#         psfs = self.vis_model.model_vis(wfs, amplitudes, phases, exposure.filter)

#         psf = psfs.data.sum(tuple(range(psfs.ndim)))
#         pixel_scale = psfs.pixel_scale.mean()

#         psf_obj = dl.PSF(psf, pixel_scale)
        
#         return detector.model(psf_obj, return_psf=False)


class BreathingFit(ModelFit):
    ns: int = eqx.field(static=True)
    def __init__(self, ns):
        self.source = dl.PointSource(wavelengths=[1])
        self.ns = ns

    def get_key(self, exposure, param):
        if param == "breathing":
            return exposure.key
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param == "breathing":
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def __call__(self, model, exposure):
        source = self.update_source(model, exposure)
        detector = self.update_detector(model, exposure)

        breathing = model.get(exposure.fit.map_param(exposure, "breathing"))
        aberrations = model.get(exposure.fit.map_param(exposure, "aberrations"))

        defocuses = np.linspace(-breathing, breathing, self.ns)

        psf = 0.0

        for i in range(self.ns):
            ab = aberrations.at[0].add(defocuses[i])
            model = model.set(exposure.fit.map_param(exposure, "aberrations"), ab)
            optics = self.update_optics(model, exposure)
            psfs = optics.model(source, return_psf=True)
            psf = psf + psfs.data.sum(tuple(range(psfs.ndim)))/self.ns

        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
class BreathingSinglePointFit(SinglePointFit, BreathingFit):
    def __init__(self, spectrum, nwavels, ns):
        SinglePointFit.__init__(self, spectrum, nwavels)
        BreathingFit.__init__(self, ns)


class BinaryFit(ModelFit):
    def __init__(self, spectrum_basis, filter):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.source = dl.Scene([
            ("primary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))), 
            ("secondary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis)))
        ])
            
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "primary_spectrum" or param == "secondary_spectrum":
            return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "primary_spectrum", "secondary_spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        primary_coeffs = model.get(exposure.fit.map_param(exposure, "primary_spectrum"))
        secondary_coeffs = model.get(exposure.fit.map_param(exposure, "secondary_spectrum"))

        source = self.source.set("primary.spectrum.basis_weights", primary_coeffs)
        source = source.set("primary.flux", source.primary.spectrum.flux)
        source = source.set("secondary.spectrum.basis_weights", secondary_coeffs)
        source = source.set("secondary.flux", source.secondary.spectrum.flux)


        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)
        separation = model.get(exposure.fit.map_param(exposure, "separation"))*dlu.arcsec2rad(0.0432)
        position_angle = dlu.deg2rad(model.get(exposure.fit.map_param(exposure, "position_angle")))


        positions = dlu.positions_from_sep(position, separation, position_angle)

        source = source.set("primary.position", positions[0])
        source = source.set("secondary.position", positions[1])
        
        return source

class PointSourceContrastFit(ModelFit):
    def __init__(self, spectrum_basis, filter):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.source = dl.Scene([
            ("primary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))), 
            ("secondary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis)))
        ])
            
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "spectrum" or param == "secondary_spectrum" or param == "secondary_position":
            return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "spectrum", "secondary_spectrum", "secondary_position"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        primary_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))
        secondary_coeffs = model.get(exposure.fit.map_param(exposure, "secondary_spectrum"))

        source = self.source.set("primary.spectrum.basis_weights", primary_coeffs)
        source = source.set("primary.flux", source.primary.spectrum.flux)
        source = source.set("secondary.spectrum.basis_weights", secondary_coeffs)
        source = source.set("secondary.flux", source.secondary.spectrum.flux)


        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)

        secondary_position = model.get(exposure.fit.map_param(exposure, "secondary_position"))*dlu.arcsec2rad(0.0432)

        source = source.set("primary.position", position)

        source = source.set("secondary.position", secondary_position)
        
        return source


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
    filters: dict
    optics: NICMOSOptics
    detector: NICMOSDetector

    def __init__(self, exposures, params, optics, detector):
        self.optics = optics
        self.detector = detector
        self.params = params
        self.filters = {}

        for filter in [e.filter for e in exposures]:
            #print(filter)
            spec = filter_files[filter]
            spec = spec.at[:,0].divide(1e10)
            self.filters[filter] = spec[::5,:]    





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
