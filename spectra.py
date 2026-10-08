import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array

import dLux as dl
import dLux.utils as dlu

from dLux import Spectrum as Spectrum
from dLux.spectra import SimpleSpectrum

import zodiax as zdx
import equinox as eqx
from abc import abstractmethod

eps = 1e-5
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

        return detected_w / self.flux

    def spec_weights(self):
        raise NotImplementedError

    def normalise(self):
        return self


class CombinedBasisSpectrum(CombinedSpectrum):
    basis_vects: Array

    def __init__(self, wavels, filt_weights, basis_weights, basis):
        self.basis_vects = np.asarray(basis, dtype=float)
        super().__init__(wavels, filt_weights, basis_weights)

    def spec_weights(self):
        return np.maximum(np.sum(self.basis_vects * self.basis_weights, axis=1), eps)
