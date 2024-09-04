import jax.numpy as np

import dLux as dl
import dLux.utils as dlu

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
                width = 0.025,#0.038*1.2,
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
                    translation = (-0.4615*1.2, 0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7606*1.2),
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

            "pad_1" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.9021*1.2, 0),
                    rotation=np.deg2rad(0)
                ),
                softening = self.softening
            ),
            "pad_2" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7655*1.2),
                    rotation=np.deg2rad(-121.15)
                ),
                softening = self.softening
            ),
            "pad_3" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7706*1.2),
                    rotation=np.deg2rad(121.52)
                ),
                softening = self.softening
            )
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
    def __init__(self, wf_npixels, psf_npixels, oversample):
        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2),
                    noll_inds=np.asarray([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                )
            ],
            psf_npixels,
            0.0431,
            oversample
        )