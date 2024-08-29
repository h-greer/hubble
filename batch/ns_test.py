# import trick

import sys
sys.path.insert(0, '../')

# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import numpy

import numpyro as npy
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler

# Optimisation imports
import zodiax as zdx
import optax

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72

from detectors import *
from apertures import *

# the real world
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import pandas as pd

import chainconsumer as cc

ddir = "../data/MAST_2024-08-27T07_49_07.684Z/"
fname = ddir + 'HST/n8ku01ffq_cal.fits'


#ddir = '../data/MAST_2024-07-11T09_26_05.575Z/'
#fname = ddir + 'HST/N43CA5020/n43ca5020_mos.fits'

data = fits.getdata(fname, ext=1)
err = fits.getdata(fname, ext=2)
info = fits.getdata(fname, ext=3)

image_hdr = fits.getheader(fname, ext=1) # this is the header for just the image in particular

# rather than manipulate WCS coordinates ourselves, can use https://docs.astropy.org/en/stable/wcs/
w = WCS(image_hdr)

y,x = numpy.unravel_index(numpy.argmax(data),data.shape)
centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units

wid = 64


cropped = Cutout2D(data, centre, wid, wcs=w).data
err_cropped = Cutout2D(err, centre, wid, wcs=w).data
info_cropped = Cutout2D(info, centre, wid, wcs=w).data

cropped_data = np.asarray(cropped, dtype=float)#+0.1#.clip(min=0)
cropped_err = np.asarray(err_cropped, dtype=float)

bad_pix = cropped_err==0.0


bad_pix_2 = (info_cropped&256) | (info_cropped&64) | (info_cropped&32)

cropped_err = np.where(bad_pix | bad_pix_2, 1e10,cropped_err)
cropped_data = np.where(bad_pix | bad_pix_2 , 0, cropped_data)

#f170m = np.asarray(pd.read_csv("../data/HST_NICMOS1.F170M.dat", sep=' '))
#f095n = np.asarray(pd.read_csv("../data/HST_NICMOS1.F095N.dat", sep=' '))
f145m = np.asarray(pd.read_csv("../data/HST_NICMOS1.F145M.dat", sep=' '))



wavels = f145m[::20,0]/1e10
weights = f145m[::20,1]


source = dl.PointSource(
    #wavelengths=wavels,
    spectrum=dl.Spectrum(wavels, weights),
    flux = 180369.28,
    #position = np.asarray([-5e-7,5e-7])
)

"""source = dl.BinarySource(
    #position = np.asarray([-5e-7,5e-7]),
    wavelengths=wavels,
    weights=weights,
    spectrum=dl.Spectrum(wavels, weights),
    mean_flux=5000,
    separation=dlu.arcsec2rad(0.042),
    #position_angle=7*np.pi/4,
    #contrast = 0.3,
)"""

oversample = 4

optics = dl.AngularOpticalSystem(
    512,
    2.4,
    [
        dl.CompoundAperture([
            ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=0.5)),
            ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4), softening=0.5))
        ],normalise=True),
        dl.AberratedAperture(
            dl.layers.CircularAperture(1.2),
            noll_inds=np.asarray([4,5,6,7,8,9,10,11]),#,12,13,14,15,16,17,18,19,20]),
            coefficients = np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
        )
    ],
    wid,
    0.0432,
    oversample
)

detector = dl.LayeredDetector(
    [
        #("detector_response", ApplyNonlinearity(coefficients=np.zeros(1), order = 3)),
        #("constant", dl.layers.AddConstant(value=0.0)),
        #("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid*oversample,wid*oversample)))),
        #("jitter", dl.layers.ApplyJitter(sigma=0.1)),
        ("downsample", dl.layers.Downsample(oversample))
     ]
)


telescope = dl.Telescope(
    optics,
    source,
    detector
)


def psf_model(data, model):

    f = npy.sample("flux", dist.Uniform(1e5,1e6))

    x = npy.sample("X", dist.Uniform(-1,1))
    y = npy.sample("Y", dist.Uniform(-1,1))

    samplers = {
        "flux": f,
        "position": np.asarray([
            x*dlu.arcsec2rad(0.042),
            y*dlu.arcsec2rad(0.042),
        ]),
        #"separation": npy.sample("Separation", dist.Uniform(0,1e-6)),
        #"contrast": npy.sample("Contrast", dist.Uniform(0,20)),
        #"position_angle": npy.sample("Position Angle", dist.Uniform(0,np.pi)),
        #"cold_mask.transformation.translation": np.asarray([
        #    npy.sample("Cold X", dist.Uniform(0,0.15)),
        #    npy.sample("Cold Y", dist.Uniform(0,0.15))
        #]),
        #"cold_mask.transformation.rotation": npy.sample("Cold Rotation", dist.Uniform(0, np.pi/2)),
        #"AberratedAperture.coefficients":
        #"constant.value": npy.sample("Detector Offset", dist.Uniform(-1,1))
    }

    model = model.set(list(samplers.keys()),list(samplers.values()))

    #for key in samplers:
    #    model = model.set(key, samplers[key])

    model_data = model.model().flatten()

    img, err, bad = data

    #img = np.where(bad, 0, img)
    #err = np.where(bad, 1e10, err)

    with npy.plate("data", size=len(img.flatten())):
        image = dist.Normal(img.flatten(), err.flatten())
        return npy.sample("psf", image, obs=model_data)


ns = NestedSampler(psf_model)

ns.run(jr.PRNGKey(100),(cropped_data, cropped_err, bad_pix), telescope)

ns.print_summary()

samples = ns.get_samples(jr.PRNGKey(1), num_samples=500)

chain = cc.Chain.from_numpyro(samples, "numpyro chain", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)

fig = consumer.plotter.plot()
fig.savefig("ns_test.png")
plt.show()