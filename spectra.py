import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

from dLux import Spectrum as Spectrum
from dLux.spectra import SimpleSpectrum

import zodiax as zdx
import equinox as eqx
import interpax as ipx
from abc import abstractmethod

def nearest_interpolate(x, xp, fp):
    dists = x - xp.reshape((-1,1))
    locs = np.argmin(np.abs(dists),axis=0)
    return fp[locs]    

class CombinedSpectrum(SimpleSpectrum):
    filt_weights: Array
    basis_weights: Array

    def __init__(self, wavels, filt_weights, basis_weights):
        super().__init__(wavels)
        self.filt_weights = np.asarray(filt_weights, float)
        #self.wavelengths = np.asarray(wavels, dtype=float)
        self.basis_weights = np.asarray(basis_weights, dtype=float)

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
    proj: Array
    def __init__(self, wavels, filt_weights, basis_weights, basis, proj=None):
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights)
        if proj is None:
            self.proj = np.eye(basis_weights.shape[0])
        else:
            self.proj = proj
    
    def spec_weights(self):
        #return 10**np.sum(self.basis_vects*np.dot(self.proj, self.basis_weights), axis=1)
        #return 10**np.sum(self.basis_vects*self.basis_weights, axis=1)
        return np.maximum(np.sum(self.basis_vects*self.basis_weights, axis=1), 1e-8)



class PreCombinedBasisSpectrum(CombinedSpectrum):
    basis_vects: Array
    def __init__(self, wavels, basis_weights, basis):
        filt_weights=np.ones_like(wavels)
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights)
    
    def spec_weights(self):
        return np.maximum(np.sum(self.basis_vects*self.basis_weights, axis=1), 1e-8)


def build_dct_basis(nx, nf):
    xs = np.arange(nx)*2*np.pi/nx
    return jax.vmap(lambda i, x: np.cos(x * i/2), in_axes=(0,None), out_axes = (1))(np.arange(nf), xs)



def load_spectrum_basis(filt, nwavels, npoly):
    basis_file = np.load(f"../data/spectrum_basis_{filt}.npy")[:,:npoly]
    spectrum_basis = ipx.interp1d(np.linspace(0,1,nwavels), np.linspace(0,1,basis_file.shape[0]), basis_file)
    return spectrum_basis/np.sqrt(np.mean(spectrum_basis**2, axis=0))

def load_custom_spectrum_basis(file, nwavels, npoly, norm=True, direct=False):
    basis_file = np.load(file)[:,:npoly]
    spectrum_basis = basis_file if direct else ipx.interp1d(np.linspace(0,1,nwavels), np.linspace(0,1,basis_file.shape[0]), basis_file)
    if norm:
        return spectrum_basis/np.sqrt(np.mean(spectrum_basis**2, axis=0))
    return spectrum_basis
