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
from abc import abstractmethod

def nearest_interpolate(x, xp, fp):
    dists = x - xp.reshape((-1,1))
    locs = np.argmin(np.abs(dists),axis=0)
    return fp[locs]    


class SimpleSpectrum(dl.BaseSpectrum):
    wavelengths: Array

    def __init__(self: dl.Spectrum, wavelengths: Array):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        super().__init__()


class NonNormalisedBasisSpectrum(SimpleSpectrum):
    coefficients: Array

    def __init__(self: Spectrum, wavelengths: Array, coefficients: Array):
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.coefficients.ndim != 1:
            raise ValueError("Coefficients must be a 1d array.")

    @abstractmethod
    def _eval_weight(self: Spectrum, wavelength: Array) -> Array:
        pass

    @property
    def weights(self: Spectrum) -> Array:
        weights = jax.vmap(self._eval_weight)(self.wavelengths)
        return weights# / weights.sum()

    def normalise(self: Spectrum) -> Spectrum:

        return self
class NonNormalisedClippedPolySpectrum(NonNormalisedBasisSpectrum):
    def _eval_weight(self: Spectrum, wavelength: Array) -> Array:
        return np.array(
            [
                self.coefficients[i] * wavelength**i
                for i in range(len(self.coefficients))
            ]
        ).sum()


class NonNormalisedSpectrum(SimpleSpectrum):
    wavelengths: Array
    weights: Array
    def __init__(self, wavels, weights):
        super().__init__(wavels)
        self.weights = np.asarray(weights, float)

    @property
    def weights(self):
        return self.weights
    def normalise(self):
        return self


class CombinedSpectrum(SimpleSpectrum):
    filt_weights: Array

    def __init__(self, wavels, filt_weights):
        super().__init__(wavels)
        self.filt_weights = np.asarray(filt_weights, float)

    @property
    def spec_weights(self):
        pass

    @property
    def weights(self):
        weights = self.filt_weights*self.spec_weights()
        return weights/weights.sum()

    @property
    def flux(self):
        return np.sum(self.filt_weights*self.spec_weights())

    def normalise(self):
        return self

class CombinedBinnedSpectrum(CombinedSpectrum):
    spectrum_weights: Array
    def __init__(self, wavels, filt_weights, spec_weights):
        super().__init__(wavels, filt_weights)
        self.spectrum_weights = np.asarray(spec_weights, float)
    
    def spec_weights(self):
        return spectrum_weights


class CombinedFourierSpectrum(CombinedSpectrum):
    fourier_weights: Array
    def __init__(self, wavels, filt_weights, fourier_weights):
        super().__init__(wavels, filt_weights)
        self.fourier_weights = np.asarray(fourier_weights, float)
    

    def spec_weights(self):
        nw = len(self.wavelengths)
        inten = np.zeros(nw)
        xs = np.linspace(0, 2*np.pi, nw)

        for i,c in enumerate(self.fourier_weights):
            inten = inten + np.cos(xs * i/2)*c
        
        return 10**inten

class CombinedPolySpectrum(CombinedSpectrum):
    poly_weights: Array
    def __init__(self, wavels, filt_weights, poly_weights):
        super().__init__(wavels, filt_weights)
        self.poly_weights = np.asarray(poly_weights, float)
    
    def spec_weights(self):
        nw = len(self.wavelengths)
        inten = np.zeros(nw)
        xs = np.linspace(-1, 1, nw)

        for i,c in enumerate(self.poly_weights):
            inten = inten + c*nw**i
        
        return 10**inten

