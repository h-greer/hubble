import jax.numpy as np
from jax import Array

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astrocut import fits_cut
from astropy.nddata import Cutout2D
import numpy
import pandas as pd

from abc import abstractmethod

from apertures import *
from detectors import *


filter_files = {
    'F170M': np.asarray(pd.read_csv("../data/HST_NICMOS1.F170M.dat", sep=' ')),
    'F095N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F095N.dat", sep=' ')),
    'F145M': np.asarray(pd.read_csv("../data/HST_NICMOS1.F145M.dat", sep=' ')),
    'F190N': np.asarray(pd.read_csv("../data/HST_NICMOS1.F190N.dat", sep=' '))
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

def exposure_from_file(fname, crop=None):
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

    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    return Exposure(filename, name, filter, data, err, bad, SinglePointFit())

class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "fluxes":
                return exposure.key
            #case _:
            #    return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        if param == "fluxes":
            return f"{param}.{exposure.get_key(param)}"
        return param

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

        psfs = model.optics.model(source, return_psf=True)
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