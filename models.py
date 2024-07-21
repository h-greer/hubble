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

from abc import abstractmethod


class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array

    fit: object = eqx.field(static=True)

    # WHERE DOES THE FIT GO

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

    def get_keys(self, exposure, param):
        """
        fallthrough at this stage, until we have more than single-filter-single-star images
        """
        match param:
            case _:
                return exposure.key
            #case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_keys(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        return param

class SinglePointFit(ModelFit):
    def __call__(self, model, exposure):
        return model.model()