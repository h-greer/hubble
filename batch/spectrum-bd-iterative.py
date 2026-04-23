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
wid = 72
oversample = 4

nwavels = 100#13#6
npoly=15

n_zernikes = 20

optics = NICMOSFresnelOptics(512, wid, oversample, n_zernikes = n_zernikes, defocus=0., fnumber=80.)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"



# spectrum_basis_f110w = load_spectrum_basis("F110W", nwavels, npoly)
# spectrum_basis_f110w = load_custom_spectrum_basis("../data/iterative_spectrum_basis.npy", nwavels, npoly, direct=True)


spectrum_data = np.load("../data/iterative_basis_binned.npz")

wavels_binned=spectrum_data["wavels_binned"]
wavels_binned_upsampled=spectrum_data["wavels_binned_upsampled"]
vects_binned=spectrum_data["vects_binned"][:,:npoly]
vects_filt_binned=np.array(spectrum_data["vects_filt_binned"])[:,:npoly]
vects_binned_upsampled=spectrum_data["vects_binned_upsampled"][:,:npoly]

ddir = "../data/MAST_2025-12-15T00_12_09.074Z/HST/"

exposures_single = [
    exposure_from_file(ddir + "n9nk29c8q_cal.fits", SinglePointFit(vects_binned, "F110W", precombined=False, wavels=wavels_binned), crop=wid),
    exposure_from_file(ddir + "n9nk29d1q_cal.fits", SinglePointFit(vects_binned, "F110W", precombined=False, wavels=wavels_binned), crop=wid),
    #exposure_from_file(ddir + "n8yj49k6q_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),
    #exposure_from_file(ddir + "n8yj49k8q_cal.fits", SinglePointFit(spectrum_basis_f110w, "F110W"), crop=wid),
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
    "fnumber": 79.68,
    "quadrature": {},
}



for idx, exp in enumerate(exposures_single):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels))#*0.6
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([6.,6.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 0.
    

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
    "spectrum": sgd(g*0.5, 10),
    "bias": sgd(g*3, 20),
    "cold_mask_shift": sgd(g*0.3, 30),
    "defocus": sgd(g*2, 30),
    "aberrations": sgd(g*0.08, 70),

    
    # #"fnumber": sgd(g*3, 100),
    "cold_mask_shear": sgd(g*0.5, 100),

    "quadrature": sgd(g*25, 250),
}

things_start = {
    "positions": sgd(g*5, 0),
    "spectrum": sgd(g*0.2, 10),
}

groups = list(things.keys())

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
opt_params

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 20)

# %%
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single)

# %%
plt.plot(np.asarray(losses[-10:])/(len(exposures_single)*wid**2))

# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things, 300, nbatches=20)

# %%
len(losses)

# %%


# %%
plt.plot(np.asarray(losses[-50:])/(len(exposures_single)*wid**2))

# %%
params_history_relative = [jax.tree.map(lambda x, y: x-y, x, params_history[0]) for x in params_history]

# %%
plot_params(params_history_relative, groups, xw = 3)
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=True)

# %%
params_history[-1]["spectrum"]["U20581_F110W"]

# %%
spec = PreCombinedBasisSpectrum(wavels_binned_upsampled, params_history[19]["spectrum"]["U20581_F110W"], vects_binned_upsampled)
plt.plot(spec.spec_weights())

# %%
# stop

# %%
final_params = optimise_optimistix(params_history[-1], model_single, exposures_single, project=True, diag=True)

# %%
final_params.params

# %%
#sol.stats

# %%
plot_comparison(final_params.inject((model_single)), final_params, exposures_single, quadrature=True)

# %%
# calculate spectrum

# %%
def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))


# %%


# %%
final_params.params

# %%
f = lambda params: loss_fn(ModelParams(final_params.params|params), exposures_single, final_params.inject((model_single)))
F, unflatten = zdx.batching.hessian(f, {"spectrum":params_history[-1]["spectrum"]}, nbatches=10)

# %%
plt.imshow(F[-(npoly):, -(npoly):])

# %%
cov_f110w = np.linalg.inv(F)[-(npoly):, -(npoly):]
spectrum_err = np.diag(np.sqrt(np.abs(cov_f110w)))


# %%
final_params

# %%
plt.imshow(cov_f110w, cmap='seismic', vmin=-np.max(np.abs(cov_f110w)), vmax=np.max(np.abs(cov_f110w)))
plt.colorbar()


# %%
np.sqrt(np.diag(cov_f110w))

# %%
print("mode covariances")
print(final_params.get("spectrum.U20581_F110W")[0]/np.sqrt(np.diag(cov_f110w)))

print(final_params.get("spectrum.U20581_F110W")[0]*np.sqrt(np.real(np.linalg.eigvals(F[-(npoly):, -(npoly):]))))


# %%
# final_params.get("spectrum.U20581_F110W")[0]/np.sqrt(vals)

# %%
vals, vects = np.linalg.eig(F[-(npoly):, -(npoly):])#+fsh['n8yj02wyq.spectrum'])

order = np.argsort(vals)[::-1]

#plt.figure(figsize=(10,10))
#plt.xlabel("Coefficient")

#for i in range(5):
#    plt.plot(np.arange(npoly),np.real(vects[:,order[i]]), label=f"{i}")
#plt.legend()

plt.semilogy(np.sort(np.real(vals))[::-1])

# %%
wavels, bandpass = calc_throughput("F110W", nwavels=nwavels*2)

# %%
#final_params = ModelParams(params_history[-1])

# %%
fname = "../data/2M1439.fits"
data = fits.getdata(fname, ext=0).astype(np.float32)

# %%
cov_f110w.shape

# %%
plt.figure(figsize=(10,5))



spec = PreCombinedBasisSpectrum(wavels_binned_upsampled, final_params.get("spectrum.U20581_F110W"), vects_binned_upsampled)
wv = wavels_binned_upsampled

sp = spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)

smax = np.max(sp)

sp = sp/smax

plt.plot(wv*1e6, sp, color='orange')

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.U20581_F110W"), (cov_f110w))
    spec = PreCombinedBasisSpectrum(wavels_binned_upsampled, coeffs, vects_binned_upsampled)
    plt.plot(wv*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)/smax, color='b', alpha=0.01, zorder=0)


wv_range = (data[0]*1e-6 > np.min(wv)) & (data[0]*1e-6 < np.max(wv))
#plt.errorbar(data[0][wv_range], data[1][wv_range]/np.mean(data[1][wv_range]), data[2][wv_range]/np.mean(data[1][wv_range]))
plt.plot(wavels*1e6, bandpass/np.max(bandpass)*0.5)

plt.xlabel("Wavelength (um)")
plt.ylim(0,2)
plt.axvline(0.81)
plt.axvline(1.4)

# %%


# %%
plt.figure(figsize=(10,5))



spec = PreCombinedBasisSpectrum(wavels_binned_upsampled, final_params.get("spectrum.U20581_F110W"), vects_binned_upsampled)
wv = wavels_binned_upsampled

sp = spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)

smax = np.mean(sp)*0.9

sp = sp/smax

#plt.plot(wv*1e6, sp, color='orange')

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.U20581_F110W"), (cov_f110w))
    spec = PreCombinedBasisSpectrum(wavels_binned_upsampled, coeffs, vects_binned_upsampled)
    plt.plot(wv*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)/smax, color='b', alpha=0.01, zorder=0)


wv_range = (data[0]*1e-6 > np.min(wv)) & (data[0]*1e-6 < np.max(wv))
plt.errorbar(data[0][wv_range], data[1][wv_range]/np.mean(data[1][wv_range]), data[2][wv_range]/np.mean(data[1][wv_range]))

plt.xlabel("Wavelength (um)")
plt.axvline(0.81)
plt.axvline(1.385)

plt.savefig("spectrum-bd.png")

# # %%
# final_params.params

# # %%
# plt.figure(figsize=(10,10))

# nw = nwavels
# wv, filt = calc_throughput("F110W", nwavels=nw)
# big_basis = spectrum_basis_f110w#load_spectrum_basis("F110W", nw, npoly)

# spec = CombinedBasisSpectrum(wv, filt, final_params.get("spectrum.U20581_F110W"), big_basis)

# sp = spec.spec_weights()/spec.flux*spec.proper_flux()#/(wv*1e6)


# # %%
# np.savez("spectrum_iterative.npz", weights=sp,params=final_params.params)


