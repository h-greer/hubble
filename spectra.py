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

class CombinedSpectrum(dl.BaseSpectrum):
    wavelengths: Array
    filt_weights: Array

    def __init__(self, wavels, filt_weights):
        self.filt_weights = np.asarray(filt_weights, float)
        self.wavelengths = np.asarray(wavels, dtype=float)
        super().__init__()

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

