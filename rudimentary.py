# Basic imports
from jax._src.api_util import flat_out_axes
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import numpy

# Optimisation imports
import zodiax as zdx
import optax

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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
from astrocut import fits_cut

ddir = './data/MAST_2024-07-03T0023/'
fname = ddir + 'HST/n9nk01010/n9nk01010_mos.fits'
#fname = ddir + 'HST/n9nk01010/n9nk01010_mos.fits'


data = fits.getdata(fname, ext=1)

image_hdr = fits.getheader(fname, ext=1) # this is the header for just the image in particular

# rather than manipulate WCS coordinates ourselves, can use https://docs.astropy.org/en/stable/wcs/
w = WCS(image_hdr)

y,x = numpy.unravel_index(numpy.argmax(data),data.shape)
centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units


# use fits_cut because we want WCS metadata cropped appropriately
cutout = fits_cut(fname, centre, 64, single_outfile=False, memory_only=True)[0] 
cropped = cutout[1].data
cropped_image_hdr = cutout[1].header


#########################################################

wavels = 1e-6 * np.linspace(1.60, 1.80, 20)

weights = np.concatenate([np.linspace(0., 1., 10), np.linspace(1., 0., 10)])

source = dl.PointSource(
    wavelengths=wavels,
    #spectrum=dl.Spectrum(wavels, weights),
    flux = 15000
)

optics = dl.AngularOpticalSystem(
    512,
    2.4,
    [
        dl.CompoundAperture([
            ("main_aperture",HSTMainAperture(softening=0.01)),
            ("cold_mask",NICMOSColdMask(softening=0.01))
        ],normalise=True),
        dl.AberratedAperture(
            dl.layers.CircularAperture(1.2), noll_inds=np.asarray([4,5,6,7,8,9,10])
        )
    ],
    64,
    0.043,
    1
)

detector = dl.LayeredDetector(
    [#("pixel_response",dl.layers.ApplyPixelResponse(np.ones((64,64)))),
     ("detector_response", ApplyNonlinearity(coefficients=np.ones(5)))
     ]
)


telescope = dl.Telescope(
    optics,
    source,
    #detector
)

paths = [
    "flux", "position",
    "cold_mask.transformation.translation", "cold_mask.transformation.rotation",
    "AberratedAperture.coefficients",
    "cold_mask.outer.radius", "cold_mask.secondary.radius",
    "main_aperture.pad_1.translation",
    "main_aperture.pad_2.translation",
    "main_aperture.pad_3.translation",
    "cold_mask.pad_1.translation",
    "cold_mask.pad_2.translation",
    "cold_mask.pad_3.translation",
    "cold_mask.pad_1.rotation",
    "cold_mask.pad_2.rotation",
    "cold_mask.pad_3.rotation",
    "cold_mask.pad_1.width",
    "cold_mask.pad_2.width",
    "cold_mask.pad_3.width",
]

cropped_data = np.asarray(cropped, dtype=float).clip(min=0)

@zdx.filter_jit
@zdx.filter_value_and_grad(paths)
def loss_fn(model,data):
    psf = model.model()
    return -jsp.stats.poisson.logpmf(data,psf).mean()

groups = [
    "flux", "position",
    "cold_mask.transformation.translation", "cold_mask.transformation.rotation",
    "AberratedAperture.coefficients",
    [
        "cold_mask.outer.radius",
        "cold_mask.secondary.radius"
    ],
    [
        "main_aperture.pad_1.translation",
        "main_aperture.pad_2.translation",
        "main_aperture.pad_3.translation",
        "cold_mask.pad_1.translation",
        "cold_mask.pad_2.translation",
        "cold_mask.pad_3.translation"
    ],
    [
        "cold_mask.pad_1.rotation",
        "cold_mask.pad_2.rotation",
        "cold_mask.pad_3.rotation"
    ],
    [
        "cold_mask.pad_1.width",
        "cold_mask.pad_2.width",
        "cold_mask.pad_3.width"
    ],
]

optimisers = [optax.adam(1e4),optax.adam(1e-8), optax.adam(8e-3), optax.adam(2e-4), optax.adam(2e-10), optax.adam(2e-3), optax.adam(2e-3), optax.adam(2e-4),optax.adam(2e-4)]#, optax.adam(1e-5)]

optim, opt_state = zdx.get_optimiser(
    telescope, groups, optimisers
)

losses, models = [], []
for i in tqdm(range(300)):
    loss, grads = loss_fn(telescope,cropped_data)
    updates, opt_state = optim.update(grads, opt_state)
    telescope = zdx.apply_updates(telescope, updates)

    models.append(telescope)
    losses.append(loss)

print(float(losses[0]), float(losses[-1]))

#print(telescope.get(paths))

for g in groups:
    if type(g) == list:
        for s in g:
            print(s, telescope.get(s))
    else:
        print(g, telescope.get(g))

coords = dlu.pixel_coords(512, 2.4)
plt.imshow(telescope.optics.transmission(coords,2.4/512))
plt.show()
#plt.imshow(np.abs(telescope.model()-cropped_data)**0.25)
