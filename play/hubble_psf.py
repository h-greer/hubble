# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from optax._src.alias import transform

# Optimisation imports
import zodiax as zdx
import optax

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72

# pixel sampling
npix = 512


# Hubble mirror parameters
diam = 2.4
spider_width = 0.038
spider_angles = np.asarray([0, 90, 180, 270])
secondary_diam = 0.305

pad_size = 0.15
pad_distance = 1
pad_angles = np.asarray([0, 2*np.pi/3, 4*np.pi/3]) + np.pi/7

coords = dlu.pixel_coords(npix, diam)

def hubble_secondaries():
    spider = dl.layers.Spider(
        width=spider_width,
        angles=spider_angles
    )

    secondary = dl.layers.CircularAperture(
        radius=secondary_diam / 2,
        occulting=True
    )

    return [secondary, spider]

def hubble_pupil():

    primary = dl.layers.CircularAperture(radius=diam/2)

    secondaries = hubble_secondaries()

    return dl.layers.CompoundAperture([primary,*secondaries], normalise=True)

def hubble_pads():
    pad_locs = [dl.CoordTransform(translation = (pad_distance*np.cos(angle),pad_distance*np.sin(angle)), rotation=-angle) for angle in pad_angles]

    pads = [dl.layers.RectangularAperture(width=pad_size, height=pad_size, occulting=True, transformation=transformation) for transformation in pad_locs]

    return pads


psf_npix = 64  # Number of pixels in the PSF
psf_pixel_scale = 50e-3  # 50 mili-arcseconds
oversample = 3  # Oversampling factor for the PSF

# add some abberations
indices = np.array([2, 3, 7, 8, 9, 10])
coefficients = 30e-9 * jr.normal(jr.PRNGKey(0), indices.shape)


layers = [
    (
        "main_aperture",
        hubble_pupil(),
    ),
    (
        "mask",
        dl.layers.CompoundAperture([*hubble_secondaries(),*hubble_pads()], normalise=True,transformation=dl.CoordTransform(np.asarray([0.08,0]))),
    ),
    (
        "aberrations",
        dl.layers.AberratedAperture(dl.layers.CircularAperture(diam/2), noll_inds=indices, coefficients=coefficients)
    )
]

optics = dl.AngularOpticalSystem(
    npix, diam, layers, psf_npix, psf_pixel_scale, oversample
)

transmission = optics.mask.transmission(coords,diam/npix) * optics.main_aperture.transmission(coords,diam/npix)

spatial_extent = (-diam/2, diam/2, -diam/2, diam/2)

ang = psf_npix*psf_pixel_scale/2 * 1e3
angular_extent = (-ang/2, ang/2, -ang/2, ang/2)


plt.figure(figsize=(14, 4))
plt.suptitle("Hubble Optics")
plt.subplot(1, 3, 1)
plt.title("Aperture Transmission")
plt.imshow(transmission, extent=spatial_extent)
plt.colorbar(label="Transmission")
plt.xlabel("x (m)")
plt.ylabel("y (m)")


plt.subplot(1, 3, 2)
plt.title("Aperture Abberations")
plt.imshow(optics.aberrations.eval_basis(coords)*1e9, extent=spatial_extent)
plt.colorbar(label="OPD (nm)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")


wavels = 1e-6 * np.linspace(1, 1.2, 10)
psf = optics.propagate(wavels)
plt.subplot(1, 3, 3)
plt.title("Sqrt PSF")
plt.imshow(psf**0.5, extent=angular_extent)
plt.colorbar(label="Sqrt Intensity")
plt.xlabel("x (mas)")
plt.ylabel("y (mas)")

plt.tight_layout()
plt.show()
