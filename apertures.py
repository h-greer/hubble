import jax.numpy as np

import dLux as dl
import dLux.utils as dlu

"""

Custom apertures for the features of the HST/NICMOS optical system
Values in here *ARE NOT TO BE TRUSTED*, they are taken out of TinyTim and my imagination where relevant.

"""

class HSTAperture(dl.CompoundAperture):
    def __init__(self, transformation=dl.CoordTransform(rotation=-np.pi/4)):
        self.normalise = False
        self.transformation = transformation
        self.apertures = {
            "mirror" : dl.CircularAperture(
                radius = 1.2,
            ),
            "spider" : dl.Spider(
                width = 0.038,
                angles = np.asarray([0, 90, 180, 270]),
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330,
                occulting = True,
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921, 0),
                )
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615, 0.7555),
                )
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564, -0.7606),
                )
            )
        }


class NICMOSColdMask(dl.CompoundAperture):
    def __init__(self, transformation=dl.CoordTransform(rotation=-np.pi/4)):
        self.normalise = False
        self.transformation = transformation
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 0.955,
            ),
            "spider" : dl.Spider(
                width = 0.077,
                angles = np.asarray([0, 90, 180, 270]) #+45
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372,
                occulting = True,
            ),

            "pad_1" : dl.SquareAperture(
                width = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921, 0),
                    rotation=np.deg2rad(0)
                )
            ),
            "pad_2" : dl.SquareAperture(
                width = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615, 0.7555),
                    rotation=np.deg2rad(-121)
                )
            ),
            "pad_3" : dl.SquareAperture(
                width = 0.065*2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564, -0.7606),
                    rotation=np.deg2rad(121)
                )
            )
        }