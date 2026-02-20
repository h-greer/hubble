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

class CombinedSpectrum(dl.Spectrum):
    wavelengths: Array
    filt_weights: Array
    basis_weights: Array

    def __init__(self, wavels, filt_weights, basis_weights):
        self.filt_weights = np.asarray(filt_weights, float)
        self.wavelengths = np.asarray(wavels, dtype=float)
        self.basis_weights = np.asarray(basis_weights, dtype=float)

    @property
    def spec_weights(self):
        pass

    @property
    def weights(self):
        weights = self.filt_weights*self.spec_weights()
        return weights/weights.sum()

    @property
    def flux(self):
        return np.sum(self.spec_weights())
    
    def proper_flux(self):
        return np.sum(self.spec_weights()*self.filt_weights)

    def normalise(self):
        return self

class CombinedBasisSpectrum(CombinedSpectrum):
    basis_vects: Array
    def __init__(self, wavels, filt_weights, basis_weights, basis):
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights)
    
    def spec_weights(self):
        return 10**np.sum(self.basis_vects*self.basis_weights, axis=1)

def build_dct_basis(nx, nf):
    xs = np.arange(nx)*2*np.pi/nx
    return jax.vmap(lambda i, x: np.cos(x * i/2), in_axes=(0,None), out_axes = (1))(np.arange(nf), xs)


