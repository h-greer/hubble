import jax.numpy as np
from jaxtyping import Array

import dLux as dl
import dLux.utils as dlu

"""
Detector layers and models
"""

class NICMOSDetector(dl.LayeredDetector):
    """
    NICMOS detector model.  Since we use pipeline-calibrated images, 
    our detector model is quite simple, only accounting for HST pointing jitter
    and residual detector bias/dark current? that isn't fully subtracted
    """
    def __init__(self: dl.LayeredDetector, oversample, wid):
        super().__init__(
            [
                ("jitter", dl.layers.ApplyJitter(sigma=7/43*oversample)),
                ("downsample", dl.layers.Downsample(oversample)),
                ("bias", dl.layers.AddConstant(value=0.0)),
            ]
        )