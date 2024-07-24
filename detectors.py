import jax.numpy as np
from jaxtyping import Array

import dLux as dl
import dLux.utils as dlu

"""
models for extra weird things that the NICMOS detector does that aren't in base dLux
should be fitted to data first
"""

class ApplyNonlinearity(dl.detector_layers.DetectorLayer):
    order : int
    coefficients : Array
    def __init__(self, coefficients=np.zeros(5), order=5):
        super().__init__()
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.order = int(order)

    def apply(self, psf):

        psf_data = psf.data

        res = psf_data#*0.0

        for i in range(2,self.order):
            res = res + self.coefficients[i]*psf_data**i
        return psf.set("data", res)

"""
detector = dl.LayeredDetector(
    [
        ("detector_response", ApplyNonlinearity(coefficients=np.zeros(1), order = 3)),
        ("constant", dl.layers.AddConstant(value=0.0)),
        ("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid*oversample,wid*oversample)))),
        #("jitter", dl.layers.ApplyJitter(sigma=0.1)),
        ("downsample", dl.layers.Downsample(oversample))
     ]
)
"""
class NICMOSDetector(dl.LayeredDetector):
    def __init__(self: dl.LayeredDetector, oversample, wid):
        super().__init__(
            [
                ("detector_response", ApplyNonlinearity(coefficients=np.zeros(1), order = 3)),
                ("constant", dl.layers.AddConstant(value=0.0)),
                ("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid*oversample,wid*oversample)))),
                #("jitter", dl.layers.ApplyJitter(sigma=0.1)),
                ("downsample", dl.layers.Downsample(oversample))
            ]
        )