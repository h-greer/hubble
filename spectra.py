import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

from dLux import Spectrum as Spectrum

import zodiax as zdx
import equinox as eqx




class SimpleSpectrum(dl.BaseSpectrum):

    wavelengths: Array

    def __init__(self: dl.Spectrum, wavelengths: Array):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        super().__init__()


class NonNormalisedClippedPolySpectrum(SimpleSpectrum):


    coefficients: Array

    def __init__(self: Spectrum, wavelengths: Array, coefficients: Array):
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.coefficients.ndim != 1:
            raise ValueError("Coefficients must be a 1d array.")

    def _eval_weight(self: Spectrum, wavelength: Array) -> Array:
        return np.array(
            [
                self.coefficients[i] * wavelength**i
                for i in range(len(self.coefficients))
            ]
        ).sum()

    @property
    def weights(self: Spectrum) -> Array:
        weights = jax.vmap(self._eval_weight)(self.wavelengths)
        return weights# / weights.sum()

    def normalise(self: Spectrum) -> Spectrum:

        return self
