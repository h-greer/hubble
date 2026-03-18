# %%
import sys
sys.path.insert(0, '..')

# %%
# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import jax
import numpy

jax.config.update("jax_enable_x64", True)


# Optimisation imports
import zodiax as zdx
import optax
import optimistix as optx

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

from detectors import *
from apertures import *
from models import *
from fisher import *
from stats import posterior
from fitting import *
from plotting import *
from spectra import *

import jax.tree_util as jtu
import interpax as ipx

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

# %%
wid = 90
oversample = 4

nwavels_f110w = 20
nwavels_f160w = 20
nbasis_f110w = 20
nbasis_f160w = 10

n_zernikes = 20

optics = NICMOSFresnelOptics(512, wid, oversample, n_zernikes = n_zernikes, defocus=0., fnumber=80.)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"

spectrum_basis_f110w = load_spectrum_basis("F110W", nwavels_f110w, nbasis_f110w)
spectrum_basis_f160w = load_spectrum_basis("F160W", nwavels_f160w, nbasis_f160w)


ddir = "../data/MAST_2024-09-26T22_53_13.719Z/HST/"

exposures_single = [
    exposure_from_file(ddir + "n8ry01tkq_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry01tmq_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry02tpq_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry02tqq_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),

    #exposure_from_file(ddir + "n8ry03vbq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry03vcq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry04vfq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry04vgq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry05vmq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry05vnq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry06vpq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
    #exposure_from_file(ddir + "n8ry06vqq_cal.fits", SinglePointFit(spectrum_basis_f160w, "F160W"), crop=wid),
]


# %%
for e in exposures_single:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)
    print(e.exptime)

# %%
params = {
    #"fluxes": {},
    "positions": {},
    "spectrum": {},
    "aberrations": {},

    #"rot": 0.,

    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_scale": {},
    "cold_mask_shear": {},
    "primary_scale": {},
    "primary_rot": {},
    "primary_shear": {},
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,

    "softening": 20.,#0.1,
    "bias": {},
    "jitter": {},

    "defocus": {},#1e5#{}
    "fnumber": 80.,
    "quadrature": {},
}


for idx, exp in enumerate(exposures_single):
    npoly = nbasis_f110w if exp.filter == "F110W" else nbasis_f160w
    nwavels = nwavels_f110w if exp.filter == "F110W" else nwavels_f160w
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set(np.log10(np.nansum(exp.data)/nwavels))
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([6.,6.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 0.#-0.233#2.4#800.#160.*20
    

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample
    params["quadrature"][exp.fit.get_key(exp, "quadrature")] = 0.


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)

g = 5e-2

things = {
    "positions": sgd(g*2.5, 0),
    "spectrum": sgd(g*1, 10),
    "cold_mask_shift": sgd(g*3, 30),
    
    "bias": sgd(g*3, 20),
    "aberrations": sgd(g*0.05, 70),
    #"jitter": sgd(g*1, 120),

    "defocus": sgd(g*5, 30),
    #"fnumber": sgd(g*100, 100),
    "cold_mask_shear": sgd(g*0.5, 100),
    "quadrature": sgd(g*10, 400),
}


things_start = {
    "positions": sgd(g*5, 0),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 10)

# %%
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False)

# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things, 500, nbatches=10*len(exposures_single))

# %%
plt.plot(np.asarray(losses[-50:])/(len(exposures_single)*wid**2))

# %%
params_history_relative = [jax.tree.map(lambda x, y: x-y, x, params_history[0]) for x in params_history]

# %%
plot_params(params_history_relative, groups, xw = 3, save="wd-spectrum-params")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False)

# %%
final_params = optimise_optimistix(params_history[-1], model_single, exposures_single, project=True, diag=True, nbatches=10*len(exposures_single))

# %%
final_params.params

# %%
#sol.stats

# %%
plot_comparison(final_params.inject((model_single)), final_params, exposures_single)

# %%
# calculate spectrum

# %%
def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))


# %%
f = lambda params: loss_fn(ModelParams(params), exposures_single, final_params.inject((model_single)))
F, unflatten = zdx.batching.hessian(f, {"spectrum":final_params["spectrum"]}, nbatches=2)

# %%
cov_all = np.linalg.inv(F)
cov_f110w = cov_all[:nbasis_f110w, :nbasis_f110w]
cov_f160w = cov_all[nbasis_f110w:, nbasis_f110w:]

# %%
plt.imshow(np.linalg.inv(F), cmap='seismic', vmin=-np.max(np.abs(cov_all)), vmax=np.max(np.abs(cov_all)))
plt.colorbar()


# %%
vals, vects = np.linalg.eig(F[:nbasis_f110w, :nbasis_f110w])#+fsh['n8yj02wyq.spectrum'])

order = np.argsort(vals)[::-1]

plt.figure()
plt.semilogy(np.sort(np.real(vals))[::-1])
plt.savefig("wd-eigenmodes-f110w.png")

vals, vects = np.linalg.eig(F[nbasis_f110w:, nbasis_f110w:])

order = np.argsort(vals)[::-1]

plt.figure()
plt.semilogy(np.sort(np.real(vals))[::-1])
plt.savefig("wd-eigenmodes-f160w.png")


# %%
plt.figure(figsize=(10,10))

wv, filt = calc_throughput("F110W", nwavels=nwavels_f110w)

spec = CombinedBasisSpectrum(wv, filt, final_params.get("spectrum.HZ4_F110W"), spectrum_basis_f110w)

wv2, filt2 = calc_throughput("F160W", nwavels=nwavels_f160w)

spec2 = CombinedBasisSpectrum(wv2, filt2, final_params.get("spectrum.HZ4_F160W"), spectrum_basis_f160w)


sp = spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)
sp2 = spec2.spec_weights()/spec2.flux*spec2.proper_flux()/(wv2*1e6)

#plt.plot(wavels, params.get("spectrum.U10764_F110W"))
plt.plot(wv*1e6, sp)
plt.plot(wv2*1e6, sp2)

plt.xlabel("Wavelength (um)")


# %%
from scipy.optimize import curve_fit
from scipy.constants import h, c, k

def planck_wavelength(wav, T, scale, bg):
    """
    Planck's Law as a function of wavelength (m) and temperature (K).
    Returns intensity in arbitrary units with a scaling factor.
    """
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    intensity = scale * a / ( (wav**4) * (np.exp(b) - 1.0) ) + bg# * (wav/1e-6)
    return intensity

def rayleigh_jeans(wav, scale):#, bg):
    return scale/wav**4# + bg

# %%
wavels = np.concat((wv, wv2))
spectrum = np.concat((sp, sp2))

initial_guesses = [14500.0, 1e-12, 0.] 

# Perform the curve fit
popt, pcov = curve_fit(planck_wavelength, wavels, spectrum, p0=initial_guesses, maxfev=10000)
print(popt)


# %%
from jax.numpy import linalg as la
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# %%
popt

# %%
np.sqrt(pcov[0,0])

# %%
np.round(153.2, -2)

# %%
spectrum

# %%
planck_wavelength(wavels, *popt) - spectrum #+ popt[2]

# %%
plt.figure(figsize=(10,10))
plt.plot(wavels, planck_wavelength(wavels, *popt) - spectrum)#+  popt[2])

# %%
plt.figure(figsize=(10,7))

plt.plot(np.sort(wavels)*1e6, planck_wavelength(np.sort(wavels), *popt), label=fr"Blackbody Curve")# $T = {np.round(popt[0], -3):3.0f}$")
plt.plot(wv*1e6, sp, label = "F110W Spectrum")
plt.plot(wv2*1e6, sp2, label = "F160W Spectrum")

plt.axvline(1.282)
plt.axvline(1.094)
plt.axvline(0.9546)
plt.axvline(1.005)
plt.axvline(0.923)
plt.axvline(0.901)

plt.axvline(1.51)



plt.xlabel("Wavelength (um)")
#plt.legend()

# %%
plt.figure(figsize=(10,7))

plt.plot(wv*1e6, sp/(wv*1e6), color='orange')
plt.plot(wv2*1e6, sp2/(wv2*1e6), color='orange')

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.HZ4_F110W"), nearestPD(cov_f110w))
    spec = CombinedBasisSpectrum(wv, filt, coeffs, spectrum_basis_f110w)#CombinedFourierSpectrum(wv, filt, coeffs)
    plt.plot(wv*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6), color='b', alpha=0.01, zorder=0)

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.HZ4_F160W"), nearestPD(cov_f160w))
    spec = CombinedBasisSpectrum(wv2, filt2, coeffs, spectrum_basis_f160w)#CombinedFourierSpectrum(wv, filt, coeffs)
    plt.plot(wv2*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv2*1e6), color='b', alpha=0.01, zorder=0)

plt.plot(wavels*1e6, planck_wavelength(wavels, *popt)/(wavels*1e6), label=fr"Blackbody Curve")# $T = {np.round(popt[0], -3):3.0f}$")


plt.xlabel("Wavelength (um)")
#plt.legend()


