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
from astropy.nddata import Cutout2D

ddir = './data/MAST_2024-07-03T0023-1/'
#fname = ddir + 'HST/n9nk01010/n9nk01010_mos.fits'
#fname = ddir + 'HST/n8yj02010/n8yj02010_mos.fits'
fname = ddir + 'HST/n9nk14010/n9nk14010_mos.fits'


data = fits.getdata(fname, ext=1)
err = fits.getdata(fname, ext=2)

image_hdr = fits.getheader(fname, ext=1) # this is the header for just the image in particular

# rather than manipulate WCS coordinates ourselves, can use https://docs.astropy.org/en/stable/wcs/
w = WCS(image_hdr)

y,x = numpy.unravel_index(numpy.argmax(data),data.shape)
centre = SkyCoord(w.pixel_to_world(x,y), unit='deg') # astropy wants to keep track of units

wid = 80

# use fits_cut because we want WCS metadata cropped appropriately
#cuts = fits_cut(fname, centre, 64, single_outfile=True, memory_only=True)
#cutout = cuts[0]
#cropped = cutout[1].data
#cropped_image_hdr = cutout[1].header

cropped = Cutout2D(data, centre, wid, wcs=w).data
err_cropped = Cutout2D(err, centre, wid, wcs=w).data


#########################################################

# F170M
wavels = 1e-6 * np.linspace(1.60, 1.80, 20)

# F110W
#wavels = 1e-6 * np.linspace(0.8, 1.4, 20)

weights = np.concatenate([np.linspace(0., 1., 10), np.linspace(1., 0., 10)])

source = dl.PointSource(
    wavelengths=wavels,
    #spectrum=dl.Spectrum(wavels, weights),
    flux = 5000
)

optics = dl.AngularOpticalSystem(
    512,
    2.4,
    [
        dl.CompoundAperture([
            ("main_aperture",HSTMainAperture(softening=0.25)),
            ("cold_mask",NICMOSColdMask(softening=0.25))
        ],normalise=True),
        dl.AberratedAperture(
            dl.layers.CircularAperture(1.2), noll_inds=np.asarray([4,5,6,7,8,9,10])
        )
    ],
    wid,
    0.043,
    1
)

detector = dl.LayeredDetector(
    [
        ("detector_response", ApplyNonlinearity(coefficients=np.zeros(3), order = 5)),
        ("constant", dl.layers.AddConstant(value=0.0)),
        ("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid,wid)))),
        #("jitter", dl.layers.ApplyJitter(sigma=0.1))
     ]
)


telescope = dl.Telescope(
    optics,
    source,
    detector
)

paths = [
    "flux",
    "position",
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
    "cold_mask.pad_1.height",
    "cold_mask.pad_2.height",
    "cold_mask.pad_3.height",
    "cold_mask.spider.width",
    "constant.value",
    "detector_response.coefficients",
    #"jitter.sigma",
    #"pixel_response.pixel_response"
]

cropped_data = np.asarray(cropped, dtype=float)#+0.1#.clip(min=0)
cropped_err = np.asarray(err_cropped, dtype=float)

cropped_err = np.where(cropped_err==0.0, 1e6,cropped_err)

@zdx.filter_jit
@zdx.filter_value_and_grad(paths)
def loss_fn(model,data):
    img, err = data
    psf = model.model()
    return -jsp.stats.norm.logpdf(psf, img, err).sum()

groups = [
    "flux", "position",
    "cold_mask.transformation.translation", "cold_mask.transformation.rotation",
    "AberratedAperture.coefficients",
    [
        "cold_mask.outer.radius",
        "cold_mask.secondary.radius"
    ],
    [
        #"main_aperture.pad_1.translation",
        #"main_aperture.pad_2.translation",
        #"main_aperture.pad_3.translation",
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
        "cold_mask.pad_3.width",
        #"cold_mask.pad_1.height",
        #"cold_mask.pad_2.height",
        #"cold_mask.pad_3.height",
    ],
    "cold_mask.spider.width",
    "constant.value",
    "detector_response.coefficients",
    #"jitter.sigma"
    #"pixel_response.pixel_response"
]

"""optimisers = [
    optax.adam(1e4),
    optax.adam(1e-8),
    optax.adam(8e-3),
    optax.adam(2e-4),
    optax.adam(2e-3),
    optax.adam(1e-10),
    optax.adam(2e-3),
    optax.adam(2e-4),
    optax.adam(2e-4)
]"""


pixel_opt = optax.piecewise_constant_schedule(init_value=1e-2*1e-8, 
                             boundaries_and_scales={100 : int(1e8)})

optimisers = [
    optax.adam(1e3),
    optax.adam(1e-8),
    optax.adam(1e-2),
    optax.adam(1e-4),
    optax.adam(1e-10),
    optax.adam(1e-3),
    optax.adam(1e-3),
    optax.adam(1e-4),
    optax.adam(1e-3),
    optax.adam(1e-3),
    optax.adam(1e-2),
    optax.adam(0e-10),
    #optax.adam(1e-2)
    #optax.adam(pixel_opt)
]


#optimisers = [optax.adam(1e4),optax.adam(1e-8), optax.adam(8e-3), optax.adam(2e-4), optax.adam(2e-9), optax.adam(2e-3), optax.adam(2e-3), optax.adam(2e-4),optax.adam(2e-4)]#, optax.adam(1e-5)]


optim, opt_state = zdx.get_optimiser(
    telescope, groups, optimisers
)

losses, models = [], []
for i in tqdm(range(100)):
    loss, grads = loss_fn(telescope,(cropped_data,cropped_err))
    updates, opt_state = optim.update(grads, opt_state)
    telescope = zdx.apply_updates(telescope, updates)

    models.append(telescope)
    losses.append(loss)

print(f"{float(losses[0]):e}, {float(losses[-1]):e}")

#print(telescope.get(paths))

for g in groups:
    if type(g) == list:
        for s in g:
            print(s, telescope.get(s))
    else:
        print(g, telescope.get(g))

fig, axs = plt.subplots(1,4)

coords = dlu.pixel_coords(512, 2.4)
axs[0].imshow((cropped_data+0.1)**0.25)
axs[1].imshow((telescope.model()+0.1)**0.25)
#axs[2].imshow(cropped_err)
axs[2].imshow(telescope.optics.transmission(coords,2.4/512))
#axs[2].imshow(telescope.detector.pixel_response.pixel_response)
axs[3].imshow(abs(cropped_data - telescope.model())**0.25)

#axs[1].imshow(telescope.optics.aberrations.eval_basis(coords)*1e9)
plt.show()
#plt.imshow(np.abs(telescope.model()-cropped_data)**0.25)
