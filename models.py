import jax.numpy as np
import jax.random as jr
from jax import Array
import jax

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

def get_filter(file):
    flt = np.asarray(pd.read_csv(file, sep=' '))#[::20,:]

    wv = flt[:,0]
    bp = flt[:,1]

    ebp = bp/(wv/1e4)

    nebp = ebp/np.sum(ebp)*(np.max(wv)-np.min(wv))*0.01
    final = flt.at[:,1].set(nebp)
    return final


filter_files = {
    'F170M': get_filter("../data/HST_NICMOS1.F170M.dat")[::5,:],
    'F095N': get_filter("../data/HST_NICMOS1.F095N.dat"),
    'F145M': get_filter("../data/HST_NICMOS1.F145M.dat")[::5,:],
    'F190N': get_filter("../data/HST_NICMOS1.F190N.dat"),
    'F108N': get_filter("../data/HST_NICMOS1.F108N.dat"),
    'F187N': get_filter("../data/HST_NICMOS1.F187N.dat"),
    'F090M': get_filter("../data/HST_NICMOS1.F090M.dat")[::5,:],
    #'F110W': np.asarray(pd.read_csv("../data/HST_NICMOS1.F110W.dat", sep=' '))[::20,:],
    'F110W': get_filter("../data/HST_NICMOS1.F110W.dat")[::20,:],
    'F160W': get_filter("../data/HST_NICMOS1.F160W.dat")[::20,:],
    'POL0S': get_filter("../data/HST_NICMOS1.POL0S.dat")[::20,:],
    'POL240S': get_filter("../data/HST_NICMOS1.POL240S.dat")[::20,:],
    'POL120S': get_filter("../data/HST_NICMOS1.POL120S.dat")[::20,:],
}

class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    mjd: str = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array

    fit: object = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, err, bad, fit, mjd):
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
    
    def get_key(self, param):
        return self.fit.get_key(self, param)

    def map_param(self, param):
        return self.fit.map_param(self, param)
    
    @property
    def key(self):
        return self.filename

"""
class InjectedExposure(Exposure):
    def __init__(self, name, filter, fit):
        self.filter = filter
        self.filename = f"{name}_{filter}"
        self.target = name
        self.fit = fit
        self.bad = None
        self.data = None
        self.err = None
        self.mjd = 0.0
    
    def inject(self, model, n_exp):
        generated_data = self.fit(model,self)
        err = (np.sqrt(generated_data) + 10)/np.sqrt(n_exp)
        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data
        object.__setattr__(self, 'data', data)#np.flip(data))
        object.__setattr__(self, 'bad', np.zeros(self.data.shape))#np.flip(np.zeros(self.data.shape)))
        object.__setattr__(self, 'err',err)#np.flip(err)) 

"""

class InjectedExposure(Exposure):
    def __init__(self, name, filter, fit, model, t_exp, n_exp):
        self.filter = filter
        self.filename = f"{name}_{filter}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0

        generated_data = self.fit(model, self) * t_exp
        err = np.sqrt(generated_data + 10**2)/np.sqrt(n_exp)
        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data

        self.data = data/t_exp
        self.err = err/t_exp
        self.bad = np.zeros(self.data.shape)




tf = lambda x: np.flip(x)#np.rot90(x,k=3)#np.flip(x)#, axis=0)

def exposure_from_file(fname, fit, extra_bad=None, crop=None):

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)
    #print(image_hdr)

    with fits.open(fname) as hdul:
        print(hdul.info())

    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))


    

    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']

    mjd = hdr['EXPSTART']

    if crop:
        w = WCS(image_hdr)
        y,x = numpy.unravel_index(numpy.nanargmax(data),data.shape)
        centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units
        data = Cutout2D(data, centre, crop, wcs=w).data
        err = Cutout2D(err, centre, crop, wcs=w).data
        info = Cutout2D(info, centre, crop, wcs=w).data

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    #if extra_bad is not None:
    #    bad = bad | extra_bad

    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    return Exposure(filename, name, filter, tf(data), tf(err), tf(bad), fit, mjd)

class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "fluxes":
                return f"{exposure.target}_{exposure.filter}"
            case "positions":
                return exposure.key
            case "aberrations":
                return exposure.key
            case "breathing":
                return exposure.key
            case "cold_mask_shift":
                return exposure.key
            case "cold_mask_rot":
                return exposure.filter
            case "cold_mask_scale":
                return exposure.filter
            case "cold_mask_shear":
                return exposure.filter
            case "primary_rot":
                return exposure.filter
            case "primary_scale":
                return exposure.filter
            case "primary_shear":
                return exposure.filter
            case "slope":
                return f"{exposure.target}_{exposure.filter}"
            case "spectrum":
                return f"{exposure.target}_{exposure.filter}"
            #case "displacement":
            #    return exposure.filter
            #case _:
            #    return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        if param in ["fluxes", "positions", "aberrations", "cold_mask_shift", "cold_mask_rot", "cold_mask_scale", "cold_mask_shear", "primary_rot", "primary_scale", "primary_shear", "breathing", "slope", "spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        return param
    
    def update_optics(self, model, exposure):
        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "aberrations"))*1e-9
            optics = optics.set("AberratedAperture.coefficients", coefficients)
        
        if "cold_mask_shift" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_shift"))*1e-2
            optics = optics.set("cold_mask.transformation.translation", translation)

        if "cold_mask_scale" in model.params.keys():
            compression = model.get(self.map_param(exposure, "cold_mask_scale"))
            optics = optics.set("cold_mask.transformation.compression", compression)

        if "cold_mask_rot" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "cold_mask_rot")))
            optics = optics.set("cold_mask.transformation.rotation", rotation)

        if "cold_mask_shear" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "cold_mask_shear")))
            optics = optics.set("cold_mask.transformation.shear", rotation)
        
        if "outer_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "outer_radius"))
            optics = optics.set("cold_mask.outer.radius", radius)
        
        if "secondary_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "secondary_radius"))
            optics = optics.set("cold_mask.secondary.radius", radius)
        
        if "spider_width" in model.params.keys():
            radius = model.get(self.map_param(exposure, "spider_width"))
            optics = optics.set("cold_mask.spider.width", radius)

        if "primary_scale" in model.params.keys():
            compression = model.get(self.map_param(exposure, "primary_scale"))
            optics = optics.set("main_aperture.transformation.compression", compression)
            optics = optics.set("AberratedAperture.aperture.transformation.compression", compression)

        if "primary_rot" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "primary_rot")))
            optics = optics.set("main_aperture.transformation.rotation", rotation)
            optics = optics.set("AberratedAperture.aperture.transformation.rotation", rotation)

        if "primary_shear" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "primary_shear")))
            optics = optics.set("main_aperture.transformation.shear", rotation)
            optics = optics.set("AberratedAperture.aperture.transformation.shear", rotation)
        
        if "rot" in model.params.keys():
            rot = dlu.deg2rad(model.get(self.map_param(exposure, "rot")))
            optics = optics.set("CompoundAperture.transformation.rotation", rot)
        if "scale" in model.params.keys():
            scale = model.get(self.map_param(exposure, "scale"))
            optics = optics.set("psf_pixel_scale", scale)
        if "softening" in model.params.keys():
            softening = model.get(self.map_param(exposure, "softening"))
            optics = optics.set("main_aperture.softening", softening)
            optics = optics.set("cold_mask.softening", softening)
            optics = optics.set("AberratedAperture.aperture.softness", softening)
        if "displacement" in model.params.keys():
            disp = model.get(self.map_param(exposure, "displacement"))
            optics = optics.set("defocus", disp)
        return optics

class SinglePointFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    def __init__(self):
        self.source = dl.PointSource(wavelengths=[1])
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        slope = model.get(exposure.fit.map_param(exposure, "spectrum"))

        wv = filter[:,0]
        inten = filter[:,1]

        wmax = np.max(wv)
        wmin = np.min(wv)

        swv = (wv-wmin)/(wmax-wmin)

        poly = dl.PolySpectrum(swv, slope)
        sloped = inten * poly.weights

        #sloped = inten * (1 + slope[0]*1e-3*swv + slope[1]*1e-6*swv**2 
        #                  + slope[2]*1e-9*swv**3 + slope[3]*1e-12*swv**4 + slope[4]*1e-15*swv**5)

        #sloped = inten * (1 + slope[0] + slope[1]*1e-3*swv + slope[2]*1e-6*swv**2 
        #                  + slope[3]*1e-9*swv**3 + slope[4]*1e-12*swv**4 + slope[5]*1e-15*swv**5)

        #sloped = sloped/np.sum(sloped)

        source = self.source.set("spectrum", dl.Spectrum(wv, sloped))
        source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        #print(source.flux, source.spectrum)

        #source = self.source
        optics = self.update_optics(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        return model.detector.model(psf_obj, return_psf=False)

def nearest_interpolate(x, xp, fp):
    dists = x - xp.reshape((-1,1))
    locs = np.argmin(np.abs(dists),axis=0)
    return fp[locs]    


class SinglePointSpectrumFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        spectrum = np.abs(model.get(exposure.fit.map_param(exposure, "spectrum")))


        wv = filter[:,0]
        inten = filter[:,1]
        swv = wv/1e-9

        wmin = np.min(swv)
        wmax = np.max(swv)
        woff = (wmax-wmin)*0.1


        wavels = np.linspace(wmin+woff, wmax-woff, self.nwavels)
        #inten = inten * nearest_interpolate(swv, wavels, spectrum)
        inten = inten * np.interp(swv, wavels, spectrum)

        source = self.source.set("spectrum", dl.Spectrum(wv, inten))
        source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        #print(source.flux, source.spectrum)

        #source = self.source
        optics = self.update_optics(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return model.detector.model(psf_obj, return_psf=False)
    
    #def get_spectrum(self, model, exposure):



class BreathingSinglePointFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    ns: int = eqx.field(static=True)
    def __init__(self):
        self.source = dl.PointSource(wavelengths=[1])
        self.ns = 6
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        slope = model.get(exposure.fit.map_param(exposure, "slope"))

        wv = filter[:,0]
        inten = filter[:,1]
        swv = wv/1e-9

        sloped = inten * (1 + slope[0]*1e-3*swv + slope[1]*1e-6*swv**2 
                          + slope[2]*1e-9*swv**3 + slope[3]*1e-12*swv**4 + slope[4]*1e-15*swv**5)# + slope[2]*swv**2 + slope[3]*swv**3)

        sloped = sloped/np.sum(sloped)

        source = self.source.set("spectrum", dl.Spectrum(wv, sloped))
        source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))

        breathing = model.get(exposure.fit.map_param(exposure, "breathing"))#*1e-9
        aberrations = model.get(exposure.fit.map_param(exposure, "aberrations"))

        psf = 0.0

        for i in range(self.ns):
            ab = aberrations - breathing/2 + i * (breathing)/(self.ns-1)
            model = model.set(exposure.fit.map_param(exposure, "aberrations"), ab)
            #print(model.params)
            optics = self.update_optics(model, exposure)
            psfs = optics.model(source, return_psf=True)
            psf = psf + psfs.data.sum(tuple(range(psfs.ndim)))/self.ns
        
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        return model.detector.model(psf_obj, return_psf=False)
    
class BinaryFit(ModelFit):
    source: dl.BinarySource = eqx.field(static=True)
    def __init__(self):
        self.source = dl.BinarySource(wavelengths=[1])
    
    def get_key(self, exposure, param):
        if param == "contrast":
            return exposure.filter
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param == "contrast":
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)
    
    def __call__(self, model, exposure):
        filter = model.filters[exposure.filter]
        slope = model.get(exposure.fit.map_param(exposure, "slope"))

        wv = filter[:,0]
        inten = filter[:,1]
        swv = wv/1e-9

        sloped = inten * (1 + slope[0]*1e-3*swv + slope[1]*1e-6*swv**2 
                          + slope[2]*1e-9*swv**3 + slope[3]*1e-12*swv**4 + slope[4]*1e-15*swv**5)# + slope[2]*swv**2 + slope[3]*swv**3)

        sloped = sloped/np.sum(sloped)

        source = self.source.set("spectrum", dl.Spectrum(wv, sloped))
        source = source.set("mean_flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("contrast", model.get(exposure.fit.map_param(exposure, "contrast")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        source = source.set("separation", model.get(exposure.fit.map_param(exposure, "separation"))*dlu.arcsec2rad(0.0432))
        source = source.set("position_angle", dlu.deg2rad(model.get(exposure.fit.map_param(exposure, "position_angle"))))
        
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