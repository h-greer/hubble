import jax.numpy as np
import jax.random as jr
from jax import Array

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
#from astrocut import fits_cut
from astropy.nddata import Cutout2D
import numpy
import pandas as pd

from abc import abstractmethod

from apertures import *
from detectors import *


filter_files = {
    'F170M': np.asarray(pd.read_csv("../data/HST_NICMOS1.F170M.dat", sep=' ')),
    'F095N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F095N.dat", sep=' ')),
    'F145M': np.asarray(pd.read_csv("../data/HST_NICMOS1.F145M.dat", sep=' '))[::5,:],
    'F190N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F190N.dat", sep=' ')),
    'F108N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F108N.dat", sep=' ')),
    'F187N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F187N.dat", sep=' '))
}

class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array

    fit: object = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, err, bad, fit):
        """
        Initialise exposure
        """
        self.filename = filename
        self.target = name
        self.filter = filter
        self.data = data
        self.err = err
        self.bad = bad

        self.fit = fit
    
    def get_key(self, param):
        return self.fit.get_key(self, param)

    def map_param(self, param):
        return self.fit.map_param(self, param)
    
    @property
    def key(self):
        return self.filename

class InjectedExposure(Exposure):
    def __init__(self, name, filter, fit):
        self.filter = filter
        self.filename = f"{name}_{filter}"
        self.target = name
        self.fit = fit
        self.bad = None
        self.data = None
        self.err = None
    
    def inject(self, model, n_exp):
        generated_data = self.fit(model,self)
        err = (np.sqrt(generated_data) + 10)/np.sqrt(n_exp)
        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data
        object.__setattr__(self, 'data', np.flip(data))
        object.__setattr__(self, 'bad', np.flip(np.zeros(self.data.shape)))
        object.__setattr__(self, 'err', np.flip(err)) 


tf = lambda x: x#np.flip(x, axis=0)

def exposure_from_file(fname, fit, extra_bad=None, crop=None):
    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)

    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']

    if crop:
        w = WCS(image_hdr)
        y,x = numpy.unravel_index(numpy.argmax(data),data.shape)
        centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units
        data = Cutout2D(data, centre, crop, wcs=w).data
        err = Cutout2D(err, centre, crop, wcs=w).data
        info = Cutout2D(info, centre, crop, wcs=w).data

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    if extra_bad is not None:
        bad = bad | extra_bad

    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    return Exposure(filename, name, filter, tf(data), tf(err), tf(bad), fit)

class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "fluxes":
                return exposure.key
            case "positions":
                return exposure.key
            case "aberrations":
                return exposure.key
            case "cold_mask_shift":
                return exposure.key
            case "cold_mask_rot":
                return exposure.key
            #case _:
            #    return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        if param in ["fluxes", "positions", "aberrations", "cold_mask_shift", "cold_mask_rot"]:
            return f"{param}.{exposure.get_key(param)}"
        return param
    
    def update_optics(self, model, exposure):
        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "aberrations"))
            optics = optics.set("AberratedAperture.coefficients", coefficients)
        
        if "cold_mask_shift" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_shift"))
            optics = optics.set("cold_mask.transformation.translation", translation)

        if "cold_mask_rot" in model.params.keys():
            rotation = model.get(self.map_param(exposure, "cold_mask_rot"))
            optics = optics.set("cold_mask.transformation.rotation", rotation)
        
        if "outer_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "outer_radius"))
            optics = optics.set("cold_mask.outer.radius", radius)
        
        if "secondary_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "secondary_radius"))
            optics = optics.set("cold_mask.secondary.radius", radius)
        
        if "spider_width" in model.params.keys():
            radius = model.get(self.map_param(exposure, "spider_width"))
            optics = optics.set("cold_mask.spider.width", radius)
        if "scale" in model.params.keys():
            scale = model.get(self.map_param(exposure, "scale"))
            optics = optics.set("psf_pixel_scale", scale)
        return optics

class SinglePointFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    def __init__(self):
        self.source = dl.PointSource(wavelengths=[1])
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        source = self.source.set("spectrum", dl.Spectrum(filter[:,0], filter[:,1]))
        source = source.set("flux", model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions")))
        #print(source.flux, source.spectrum)

        #source = self.source
        optics = self.update_optics(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        return model.detector.model(psf_obj, return_psf=False)

    
class BinaryFit(ModelFit):
    source: dl.BinarySource = eqx.field(static=True)
    def __init__(self):
        self.source = dl.BinarySource(wavelengths=[1])
    
    def get_key(self, exposure, param):
        if param == "contrast":
            return exposure.key
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param == "contrast":
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)
    
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        source = self.source.set("wavelengths", filter[:,0])
        source = source.set("weights", filter[:,1])
        source = source.set("mean_flux", model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("contrast", model.get(exposure.fit.map_param(exposure, "contrast")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions")))
        source = source.set("separation", model.get(exposure.fit.map_param(exposure, "separation")))
        source = source.set("position_angle", model.get(exposure.fit.map_param(exposure, "position_angle")))
        
        optics = self.update_optics(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        return model.detector.model(psf_obj, return_psf=False)


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

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def __getattr__(self, key):
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys]))

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)