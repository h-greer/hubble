import jax
import jax.numpy as np
import numpy as onp
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


def hermite_node_coeffs(wavelengths, weights, n_nodes):
    """Cubic-Hermite interpolation of the PSF in wavelength (after milolangker/nircam_halo): knots at n_nodes of
    the (sorted) fine wavelengths, evenly spaced in index with both ends included, and value / derivative
    coefficients a, b such that sum_j w_j PSF(l_j) ~= sum_k a_k PSF(L_k) + b_k dPSF/dl(L_k), exact for cubics."""
    assert n_nodes >= 2
    nodes = wavelengths[onp.unique(onp.round(onp.linspace(0, len(wavelengths) - 1, n_nodes)).astype(int))]
    i = np.clip(np.searchsorted(nodes, wavelengths, side="right") - 1, 0, len(nodes) - 2)
    h = nodes[i + 1] - nodes[i]
    t = (wavelengths - nodes[i]) / h
    h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
    zeros = np.zeros(len(nodes))
    a = zeros.at[i].add(weights*h00).at[i + 1].add(weights*h01)
    b = zeros.at[i].add(weights*h*h10).at[i + 1].add(weights*h*h11)
    return nodes, a, b


def hermite_propagate(mono, wavelengths, weights, n_nodes, batch_size=None):
    """sum_j w_j PSF(l_j) from the PSF and its exact wavelength derivative (forward-mode jvp, ~2.5x one wavelength)
    at n_nodes knots only; the error falls ~n_nodes^-4 and barely depends on the number of fine wavelengths
    (on-axis ~2e-4 of the peak at 9 knots, ~2e-5 at 15). mono(wavelength) -> (psf, pixel_scale) at unit weight.
    Returns (psf, pixel_scale)."""
    nodes, a, b = hermite_node_coeffs(wavelengths, weights, n_nodes)
    node = lambda wl: jax.jvp(mono, (wl,), (np.ones_like(wl),))
    if batch_size is None:
        (psfs, ps), (dpsfs, _) = jax.vmap(node)(nodes)
    else:
        (psfs, ps), (dpsfs, _) = jax.lax.map(node, nodes, batch_size=batch_size)
    return np.tensordot(a, psfs, 1) + np.tensordot(b, dpsfs, 1), ps.mean()
