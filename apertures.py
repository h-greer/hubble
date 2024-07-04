import jax.numpy as np

import dLux as dl
import dLux.utils as dlu

"""

Custom apertures for the features of the HST/NICMOS optical system
Values in here *ARE NOT TO BE TRUSTED*, they are taken out of TinyTim and my imagination where relevant.

"""

class HSTMainAperture(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(rotation=-np.pi/4), softening=0.25):
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
                width = 0.038,
                angles = np.asarray([0, 90, 180, 270]),
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330,
                occulting = True,
                softening = self.softening
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921, 0),
                ),
                softening = self.softening
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615, 0.7555),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564, -0.7606),
                ),
                softening=self.softening
            )
        }



class NICMOSColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((0.0,0.0)),rotation=-np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 0.955,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.077,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372,
                occulting = True,
                softening = self.softening
            ),

            "pad_1" : dl.RectangularAperture(
                width = 0.065*2,
                height = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921, 0),
                    rotation=np.deg2rad(0)
                ),
                softening = self.softening
            ),
            "pad_2" : dl.RectangularAperture(
                width = 0.065*2,
                height = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615, 0.7555),
                    rotation=np.deg2rad(-121)
                ),
                softening = self.softening
            ),
            "pad_3" : dl.RectangularAperture(
                width = 0.065*2,
                height = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564, -0.7606),
                    rotation=np.deg2rad(121)
                ),
                softening = self.softening
            )
        }
