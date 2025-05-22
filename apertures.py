from dLux.optical_systems import OpticalSystem
import jax.numpy as np

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx

"""

Custom apertures for the features of the HST/NICMOS optical system
Values in here *ARE NOT TO BE TRUSTED*, they are taken out of TinyTim and my imagination where relevant.

"""

class HSTMainAperture(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "mirror" : dl.CircularAperture(
                radius = 1.2,
                softening=self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.022*1.2,#0.038*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330*1.2,
                occulting = True,
                softening = self.softening
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921*1.2, 0),
                ),
                softening = self.softening
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, -0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, 0.7606*1.2),
                ),
                softening=self.softening
            )
        }



class NICMOSColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.955,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.077*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372*1.2,
                occulting = True,
                softening = self.softening
            ),

            # "pad_1" : dl.RectangularAperture(
            #     width = 0.1650*1.2,
            #     height = 0.1410*1.2,
            #     occulting = True,
            #     transformation=dl.CoordTransform(
            #         translation = (0.9021*1.2, 0),
            #         rotation=np.deg2rad(0)
            #     ),
            #     softening = self.softening
            # ),
            # "pad_2" : dl.RectangularAperture(
            #     width = 0.1650*1.2,
            #     height = 0.1410*1.2,
            #     occulting = True,
            #     transformation=dl.CoordTransform(
            #         translation = (-0.4615*1.2, 0.7655*1.2),
            #         rotation=np.deg2rad(-121.15)
            #     ),
            #     softening = self.softening
            # ),
            # "pad_3" : dl.RectangularAperture(
            #     width = 0.1650*1.2,
            #     height = 0.1410*1.2,
            #     occulting = True,
            #     transformation=dl.CoordTransform(
            #         translation = (-0.4564*1.2, -0.7706*1.2),
            #         rotation=np.deg2rad(121.52)
            #     ),
            #     softening = self.softening
            # )
        }

"""
optics = dl.AngularOpticalSystem(
    512,
    2.4,
    [
        dl.CompoundAperture([
            ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=0.1)),
            ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4), softening=0.1))
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
"""


class NICMOSOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, psf_oversample=1):
        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,30),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431/psf_oversample,
            oversample
        )

### Fresnel propagators ###
def transfer_fn(coords, npixels, wavelength, pscale, distance):
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return _fftshift(np.exp(-1.0j * np.pi * wavelength * distance * rho_sq))

def transfer(wf, distance, pad=2):
    coords = dlu.pixel_coords(pad * wf.npixels, pad * wf.diameter)
    return transfer_fn(coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance)


def _fft(phasor, pad=2):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return 1 / padded.shape[0] * np.fft.fft2(padded)


def _ifft(phasor, pad=1):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return phasor.shape[0] * np.fft.ifft2(padded)


def _fftshift(phasor):
    return np.fft.fftshift(phasor)


def plane_to_plane(wf, distance, pad=2):
    fft_wf = _fft(wf.phasor, pad=pad)
    tf = transfer(wf, distance, pad=pad)
    phasor = dlu.resize(_ifft(fft_wf * tf), wf.npixels)
    return wf.set(["amplitude", "phase"], [np.abs(phasor), np.angle(phasor)])




class NICMOSFresnelOptics(dl.AngularOpticalSystem):
    defocus: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus

        layers = []

        layers += [("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2))]

        layers += [
            ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,30),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                )]
        
        
    

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample

        wf *= list(self.layers.values())[0]

        # Apply layers
        #for layer in list(self.layers.values())[0:2]:
            #print(layer)
        #    wf *= layer
        
        # cold mask offset
        #print(wf)
        #wf *= list(self.layers.values())[2]
        #print(wf)
        #wf = plane_to_plane(wf, 1e-6 * self.defocus, pad=2)
        print(wf.amplitude)

        wf = plane_to_plane(wf, self.defocus/1e3, pad=2)
        print(wf.amplitude)


        wf *= list(self.layers.values())[1]
        wf *= list(self.layers.values())[2]

        wf = wf.propagate(psf_npixels, pixel_scale)


        if return_wf:
            return wf
        return wf.psf
        

def gen_powers(degree):
    """
    Generates the powers required for a 2d polynomial
    """
    n = dlu.triangular_number(degree)
    vals = np.arange(n)

    # Ypows
    tris = dlu.triangular_number(np.arange(degree))
    ydiffs = np.repeat(tris, np.arange(1, degree + 1))
    ypows = vals - ydiffs

    # Xpows
    tris = dlu.triangular_number(np.arange(1, degree + 1))
    xdiffs = np.repeat(n - np.flip(tris), np.arange(degree, 0, -1))
    xpows = np.flip(vals - xdiffs)

    return xpows, ypows


def distort_coords(coords, coeffs, pows):
    pow_base = np.multiply(*(coords[:, None, ...] ** pows[..., None, None]))
    distortion = np.sum(coeffs[..., None, None] * pow_base[None, ...], axis=1)
    return coords + distortion

class DistortedCoords(zdx.Base):
    powers: np.ndarray
    distortion: np.ndarray

    def __init__(self, order=1, distortion=None):
        self.powers = np.array(gen_powers(order + 1))#[:, 1:]

        if distortion is None:
            distortion = np.zeros_like(self.powers)
        if distortion is not None and distortion.shape != self.powers.shape:
            raise ValueError("Distortion shape must match powers shape")
        self.distortion = distortion

    def calculate(self, npix, diameter):
        coords = dlu.pixel_coords(npix, diameter)
        return distort_coords(coords, self.distortion, self.powers)

    def apply(self, coords):
        return distort_coords(coords, self.distortion, self.powers)

class NICMOSDistortedOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, distortion_orders=5):

        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=DistortedCoords(order=distortion_orders),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=DistortedCoords(order=distortion_orders), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=np.pi/4)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,30),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431,
            oversample
        )
    #def apply(self, wavefront):


