import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

from dLux import Spectrum as Spectrum
from dLux.spectra import SimpleSpectrum





class CombinedSpectrum(SimpleSpectrum):
    wavelengths: Array
    filt_weights: Array
    basis_weights: Array

    def __init__(self, wavels, filt_weights, basis_weights):
        self.wavelengths = np.asarray(wavels, dtype=float)
        self.filt_weights = np.asarray(filt_weights, dtype=float)
        self.basis_weights = np.asarray(basis_weights, dtype=float)

    @property
    def flux(self):
        spec_w = self.spec_weights()
        detected_w = self.filt_weights * spec_w

        return detected_w.sum()

    @property
    def weights(self):
        spec_w = self.spec_weights()
        detected_w = self.filt_weights * spec_w

        flux = detected_w.sum()
        weights = detected_w / self.flux

        return weights

    def spec_weights(self):
        raise NotImplementedError

    def normalise(self):
        return self


eps = 1e-8
class CombinedBasisSpectrum(CombinedSpectrum):
    basis_vects: Array

    def __init__(self, wavels, filt_weights, basis_weights, basis):
        self.basis_vects = np.asarray(basis, dtype=float)
        super().__init__(wavels, filt_weights, basis_weights)

    def spec_weights(self):
        return np.maximum(np.sum(self.basis_vects * self.basis_weights, axis=1), eps)


class PreCombinedBasisSpectrum(CombinedSpectrum):
    basis_vects: Array
    def __init__(self, wavels, basis_weights, basis):
        filt_weights=np.ones_like(wavels)
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights)
    
    def spec_weights(self):
        return np.maximum(np.sum(self.basis_vects*self.basis_weights, axis=1), eps)