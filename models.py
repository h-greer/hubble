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
from vis_models import LogVisModel


class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    mjd: str = eqx.field(static=True)
    exptime: str = eqx.field(static=True)
    wcs: object = eqx.field(static=True)
    pam: object = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array


    fit: object = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, err, bad, fit, mjd, exptime, wcs, pam):
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
        self.filename = f"{name}_{filter}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        self.data = 0.
        self.err = 0.
        self.bad = 0.
        self.exptime = 0.
        self.pam = 0.


class InjectedExposure(Exposure):
    def __init__(self, name, filter, fit, model, t_exp, n_exp):
        self.filter = filter
        self.filename = f"{name}_{filter}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        generated_data = self.fit(model, self) * t_exp * 5
        err = np.sqrt(generated_data + 3**2)/np.sqrt(n_exp)
        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data

        self.data = data/t_exp/5
        self.err = err/t_exp/5
        self.bad = np.zeros(self.data.shape)

        self.exptime = t_exp
        self.pam = 0.




tf = lambda x: x#np.flip(x)#np.rot90(x,k=3)#np.flip(x)#, axis=0)

def exposure_from_file(fname, fit, extra_bad=None, crop=None):

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)
    #print(image_hdr)

    #with fits.open(fname) as hdul:
    #    print(hdul.info())

    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    wcs = WCS(image_hdr)


    pam = hdr['NPFOCUSP']

    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']

    exptime = float(hdr['EXPTIME'])
    gain = float(hdr['ADCGAIN'])

    mjd = hdr['EXPSTART']

    #print(hdr['CAL_VER'])
    #print(hdr['NPFOCUSP'])

    if crop:
        w = WCS(image_hdr)
        y,x = numpy.unravel_index(numpy.nanargmax(data),data.shape)
        centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units
        data = Cutout2D(data, centre, crop, wcs=w).data
        err = Cutout2D(err, centre, crop, wcs=w).data
        info = Cutout2D(info, centre, crop, wcs=w).data

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32))
    if extra_bad is not None:
        #print("extra bad")
        bad = bad | tf(extra_bad)

    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    err_with_poisson = np.sqrt(data/(gain*exptime) + err**2)

    bad_with_poisson = np.isnan(err_with_poisson)

    return Exposure(filename, name, filter, tf(data), tf(err_with_poisson), tf(bad_with_poisson), fit, mjd, exptime, wcs, pam)

class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "fluxes":
                return exposure.key#f"{exposure.target}_{exposure.filter}"
            case "positions":
                return exposure.key
            case "aberrations":
                return exposure.key
            case "breathing":
                return exposure.key
            case "cold_mask_shift":
                #return "global"#exposure.key#"global"
                return str(round(exposure.mjd))
            case "cold_mask_rot":
                return "global"#exposure.key#"global"
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
            case "spectrum" | "primary_spectrum" | "secondary_spectrum":
                #return exposure.key
                return f"{exposure.target}_{exposure.filter}"

            case "primary_distortion" | "cold_mask_distortion":
                return "global"
            case "defocus":
                return exposure.key
            #    return exposure.filter
            #case _:
            #    return exposure.key
            case "bias":
                return exposure.key
            case "jitter":
                return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        if param in ["fluxes", "positions", "aberrations", "cold_mask_shift", "cold_mask_rot", "cold_mask_scale", "cold_mask_shear", "primary_rot", "primary_scale", "primary_shear", "breathing", "slope", "spectrum", "primary_spectrum", "secondary_spectrum", "bias", "jitter", "primary_distortion", "cold_mask_distortion", "defocus"]:
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
            optics = optics.set("displacement", disp)

        if "defocus" in model.params.keys():
            disp = model.get(self.map_param(exposure, "defocus"))
            optics = optics.set("defocus", disp)

        if "primary_distortion" in model.params.keys():
            dist = model.get(self.map_param(exposure, "primary_distortion"))
            optics = optics.set("main_aperture.transformation.distortion", dist)

        if "cold_mask_distortion" in model.params.keys():
            dist = model.get(self.map_param(exposure, "cold_mask_distortion"))
            optics = optics.set("cold_mask.transformation.distortion", dist)        

        return optics

    def update_detector(self, model, exposure):
        detector = model.detector

        if "bias" in model.params.keys():
            bias = model.get(self.map_param(exposure, "bias"))
            detector = detector.set("bias.value", bias)
        if "jitter" in model.params.keys():
            jitter = model.get(self.map_param(exposure, "jitter"))
            detector = detector.set("jitter.sigma", jitter)
        return detector

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

"""
class NonNormalisedSpectrum(dl.SimpleSpectrum):
    wavelengths: Array
    weights: Array
    def __init__(self, wavels, weights):
        super().__init__(wavels)
        self.weights = np.asarray(weights, float)

        #self.wavelengths = np.asarray(wavels, float)
    @property
    def weights(self):
        return self.weights#10**self.weights
    def normalise(self):
        return self
"""

class SinglePointSpectrumFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels
    def __call__(self, model, exposure):

    
        spectrum = model.get(exposure.fit.map_param(exposure, "spectrum"))

        #wv_small, _ = calc_throughput(exposure.filter, nwavels=len(spectrum))
        
        wv, inten = calc_throughput(exposure.filter, nwavels=len(spectrum))

        """

        wv = filter[:,0]
        inten = filter[:,1]
        swv = wv/1e-9

        wmin = np.min(swv)
        wmax = np.max(swv)
        woff = (wmax-wmin)*0.1


        wavels = np.linspace(wmin+woff, wmax-woff, self.nwavels)
        #inten = inten * nearest_interpolate(swv, wavels, spectrum)
        inten = inten * np.interp(swv, wavels, spectrum, left=0., right=0.)

        #source = self.source.set("spectrum", NonNormalisedSpectrum(wv, inten))
        """

        #spec_full = np.interp(wv, wv_small, spectrum)

        source = self.source.set("spectrum", dl.Spectrum(wv, inten*10**spectrum))

        source = source.set("flux", np.sum(10**spectrum))

        #source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        #print(source.flux, source.spectrum)

        #source = self.source
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
    #def get_spectrum(self, model, exposure):


class SinglePointFourierSpectrumFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels
    def __call__(self, model, exposure):

        nw = self.nwavels


        coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        wv, filt = calc_throughput(exposure.filter, nwavels=nw)

        inten = np.zeros(nw)
        xs = np.linspace(0, 2*np.pi, nw)


        for i,c in enumerate(coeffs):
            inten = inten + np.cos(xs * i/2)*c
            #inten = inten +  jax.lax.select(i == 0, np.ones(nw), np.zeros(nw))*c
            #inten = inten +  jax.lax.select((i % 2) == 0, np.cos(xs*i/2), np.zeros(nw))*c
            #inten = inten +  jax.lax.select((i % 2) == 1, np.sin(xs*(i+1)/2), np.zeros(nw))*c
            """if c == 0:
                inten += np.ones(nw)
            elif c % 2:
                inten += np.cos(xs*c//2)
            else:
                inten += np.sin(xs*(c+1)//2)"""
            
        #inten = np.maximum(inten, 0.0)#np.zeros(nw))
        #inten = np.abs(inten)

        inten = 10**inten
        inten *= filt

        source = self.source.set("flux", np.nansum(inten))

        source = source.set("spectrum", dl.Spectrum(wv, inten))
        #source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))


        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    

class SinglePointPolySpectrumFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels
    def __call__(self, model, exposure):

        source = self.source

        nw = self.nwavels


        coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        wv, filt = calc_throughput(exposure.filter, nwavels=nw)

        inten = NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nw), coeffs).weights

        #inten = jax.nn.softplus(inten/10)*10
        inten = 10**inten

        source = source.set("flux", np.sum(inten))


        inten *= filt

        source = source.set("spectrum", dl.Spectrum(wv, inten))
        #source = source.set("flux", 10**model.get(exposure.fit.map_param(exposure, "fluxes")))


        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)


class SpectrumVisFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    vis_model: LogVisModel
    def __init__(self, nwavels, vis_model):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels

        self.vis_model = vis_model

    def get_key(self, exposure, param):
        if param == "phases":
            return exposure.key
        elif param == "amplitudes":
            return exposure.key
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param == "phases":
            return f"{param}.{exposure.get_key(param)}"
        elif param == "amplitudes":
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)


    def __call__(self, model, exposure):

        source = self.source

        nw = self.nwavels


        spectrum = model.get(exposure.fit.map_param(exposure, "spectrum"))
        wv, inten = calc_throughput(exposure.filter, nwavels=len(spectrum))
        source = self.source.set("spectrum", dl.Spectrum(wv, inten*10**spectrum))
        source = source.set("flux", np.sum(10**spectrum))

        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        wfs = optics.model(source, return_wf=True)

        print(exposure.fit.map_param(exposure, "phases"))
        phases = model.get(exposure.fit.map_param(exposure, "phases"))
        amplitudes = model.get(exposure.fit.map_param(exposure, "amplitudes"))
        #phases = model.get(exposure.fit.map_param(exposure, "phases"))

        psfs = self.vis_model.model_vis(wfs, amplitudes, phases, exposure.filter)

        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)


class BreathingSinglePointFit(ModelFit):
    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    ns: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.PointSource(wavelengths=[1])
        self.nwavels = nwavels
        self.ns = 5
    def __call__(self, model, exposure):
        source = self.source
        nw = self.nwavels

        coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))
        wv, filt = calc_throughput(exposure.filter, nwavels=nw)
        inten = NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nw), coeffs).weights
        inten = 10**inten
        source = source.set("flux", np.sum(inten))
        inten *= filt

        source = source.set("spectrum", dl.Spectrum(wv, inten))

        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        detector = self.update_detector(model, exposure)

        breathing = model.get(exposure.fit.map_param(exposure, "breathing"))
        aberrations = model.get(exposure.fit.map_param(exposure, "aberrations"))

        defocuses = np.linspace(-breathing, breathing, self.ns)

        psf = 0.0

        for i in range(self.ns):
            ab = aberrations.at[0].set(defocuses[i])
            model = model.set(exposure.fit.map_param(exposure, "aberrations"), ab)
            optics = self.update_optics(model, exposure)
            psfs = optics.model(source, return_psf=True)
            psf = psf + psfs.data.sum(tuple(range(psfs.ndim)))/self.ns

        #psfs = optics.model(source, return_psf=True)
        #psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
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

class BinaryPolySpectrumFit(ModelFit):
    source: dl.Scene = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)
    def __init__(self, nwavels):
        self.source = dl.Scene([("primary",dl.PointSource(wavelengths=[1])), ("secondary",dl.PointSource(wavelengths=[1]))])
        self.nwavels = nwavels
    
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

        source = self.source
        nw = self.nwavels
        wv, filt = calc_throughput(exposure.filter, nwavels=nw)

        flux = 10**model.get(exposure.fit.map_param(exposure, "fluxes"))
        contrast = model.get(exposure.fit.map_param(exposure, "contrast"))

        flux_scales = dlu.fluxes_from_contrast(flux, contrast)


        primary_coeffs = model.get(exposure.fit.map_param(exposure, "primary_spectrum"))
        inten = NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nw), primary_coeffs).weights
        inten = 10**inten
        #inten = jax.nn.softplus(inten/10)*10
        source = source.set("primary.flux", np.sum(inten)*flux_scales[0])
        inten *= filt
        source = source.set("primary.spectrum", dl.Spectrum(wv, inten))

        secondary_coeffs = model.get(exposure.fit.map_param(exposure, "secondary_spectrum"))
        inten = NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nw), secondary_coeffs).weights
        #inten = jax.nn.softplus(inten/10)*10
        inten = 10**inten
        source = source.set("secondary.flux", np.sum(inten)*flux_scales[1])
        inten *= filt
        source = source.set("secondary.spectrum", dl.Spectrum(wv, inten))

        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)
        separation = model.get(exposure.fit.map_param(exposure, "separation"))*dlu.arcsec2rad(0.0432)
        position_angle = dlu.deg2rad(model.get(exposure.fit.map_param(exposure, "position_angle")))

        positions = dlu.positions_from_sep(position, separation, position_angle)

        source = source.set("primary.position", positions[0])
        source = source.set("secondary.position", positions[1])
        
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        return detector.model(psf_obj, return_psf=False)


"""class BaseModeller(zdx.Base):
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

        return values"""

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


"""class ModelParams(BaseModeller):

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
        return other.set(self.keys, self.values)"""



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
