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
        res = np.zeros(psf.shape)

        for i in range(self.order):
            res = res + self.coefficients[i]*psf**i
        return res
