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
def L1_loss(arr):
    """L1 norm loss for array-like inputs."""
    return np.nansum(np.abs(arr))


def L2_loss(arr):
    """L2 (quadratic) loss for array-like inputs."""
    return np.nansum(arr**2)


def tikhinov(arr):
    """Finite-difference approximation used by several regularisers."""
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :], axis=1)
    dy = np.diff(pad_arr[:, 0:-1], axis=0)
    return dx**2 + dy**2


def TV_loss(arr, eps=1e-16):
    """Total variation (approx.) loss computed from finite differences."""
    return np.sqrt(tikhinov(arr) + eps**2).sum()


def TSV_loss(arr):
    """Total squared variation (quadratic) loss."""
    return tikhinov(arr).sum()


def ME_loss(arr, eps=1e-16):
    """Maximum-entropy inspired loss (negative entropy of distribution)."""
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S

# %%
class PointResolvedFit(ModelFit):
    wid: float
    oversample: float

    def __init__(self, spectrum_basis, filter, wavels, wid, oversample):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)

        wvr, intenr = calc_throughput(filter, 3)

        self.source = dl.Scene([
            ("point", dl.PointSource(spectrum=CombinedBasisSpectrum(wavels, inten, np.zeros(nbasis), spectrum_basis))),
            ("resolved", dl.ResolvedSource(
                wavelengths=wv,
                flux=1.,
                spectrum=dl.Spectrum(wvr, intenr), 
                distribution=np.ones((wid*oversample, wid*oversample))
            ))
        ])
        self.wid = wid
        self.oversample = oversample
    
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "spectrum" or param == "flux":
            return f"{exposure.target}_{exposure.filter}"
        elif param == "resolved":
            return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "spectrum", "resolved"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def get_distribution(self, model, exposure):
        return 10**(model.get(exposure.fit.map_param(exposure, "resolved")))

    def update_source(self, model, exposure):
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("point.spectrum.basis_weights", spectrum_coeffs)
        source = source.set("point.flux", source.point.spectrum.flux)
        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)
        source = source.set("point.position", position)
        source = source.set("resolved.position", position)
        

        distribution = self.get_distribution(model, exposure)

        source = source.set("resolved.distribution",  distribution)
        source = source.set("resolved.flux",  distribution.sum())
        
        return source

    def loglike(self, model, exposure, per_pix=False, return_im=False):


        if "resolved" in model.params.keys():
            dist = self.get_distribution(model, exposure)
            return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im) + 0.05* L1_loss(dist) +  0.2*TV_loss(dist)
        
        return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im)

# %%
wid = 100
oversample = 2

nwavels = 20#13#6
npoly=4

n_zernikes = 20

optics = NICMOSFresnelOptics(512, wid, oversample, n_zernikes = n_zernikes, defocus=0., fnumber=80.)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"


# data = np.load("spectrum_iterative.npz", allow_pickle=True)
# weights = data["weights"]
# params = data["params"][()]

# weights_downsampled = ipx.interp1d(np.linspace(0,1,nwavels), np.linspace(0,1,weights.shape[0]), weights)

spectrum_basis_f110w = load_spectrum_basis("F110W", nwavels, npoly)
spectrum_basis_f110w = load_custom_spectrum_basis("../data/iterative_spectrum_basis.npy", nwavels, npoly, direct=True)


spectrum_data = np.load("../data/iterative_basis_binned.npz")

wavels_binned=spectrum_data["wavels_binned"]
wavels_binned_upsampled=spectrum_data["wavels_binned_upsampled"]
vects_binned=spectrum_data["vects_binned"][:,:npoly]
vects_filt_binned=np.array(spectrum_data["vects_filt_binned"])[:,:npoly]
vects_binned_upsampled=spectrum_data["vects_binned_upsampled"][:,:npoly]
big_basis=spectrum_data["big_basis"][:,:npoly]


ddir = "../data/MAST_2025-12-15T00_12_09.074Z/HST/"

ddir = '../data/MAST_2024-09-19T06_48_02.332Z/HST/'

ddir = '../data/MAST_2024-09-19T06_48_02.332Z/HST/'


exposures_single = [
    exposure_from_file(ddir + "na2a05ttq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05tuq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05txq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05tzq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),

    exposure_from_file(ddir + "na2a05u2q_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u4q_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u7q_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u9q_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),

    exposure_from_file(ddir + "na2a05ucq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05ueq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05uhq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05ujq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),

    exposure_from_file(ddir + "na2a05umq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05uoq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05urq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05utq_cal.fits", PointResolvedFit(vects_binned, "F110W", wavels_binned, wid=wid, oversample=oversample), crop=wid),

]

# %%
wavels_binned.shape

# %%
vects_binned.shape

# %%
np.array(vects_filt_binned)

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
    "resolved": {}
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
    

    params["bias"][exp.fit.get_key(exp, "bias")] = -0.2
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample
    params["quadrature"][exp.fit.get_key(exp, "quadrature")] = 0.
    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((wid*oversample, wid*oversample))-2.


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%
# stop

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)

g = 5e-2

things = {
    "positions": sgd(g*2.5, 0),
    "spectrum": sgd(g*0.5, 10),
    #"bias": sgd(g*3, 20),
    "cold_mask_shift": sgd(g*2, 30),
    "defocus": sgd(g*0.5, 30),
    "aberrations": sgd(g*0.08, 70),

    
    # #"fnumber": sgd(g*3, 100),
    "cold_mask_shear": sgd(g*0.5, 100),

    #"quadrature": sgd(g*25, 950),
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
losses, params_history, C = optimise_new(opt_params, model_single, exposures_single, things, 300, nbatches=64, return_c=True)

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
_, flat_fn = jax.flatten_util.ravel_pytree(params_history[-1])

# %%
flat, _ = jax.flatten_util.ravel_pytree(dict(sorted((flat_fn(C) | {"resolved": {"N458_F110W": np.ones(wid**2 * oversample**2)}}).items())))

# %%
C_new = flat#np.diag(flat)#np.diag(np.concat((np.diag(C), np.ones(wid**2 * oversample**2))))

# %%
C_new.shape

# %%
65567**2 * 8 / 1e9

# %%
def adam(lr, delay, momentum=0.5):
    return optax.adam(zdx.optimisation.delay(lr, delay))


things_all = things | {"resolved": adam(3e-2, 0)}

# %%
orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_all})

# %%
opt_params

# %%
opt_params

# %%
losses, params_history_resolved = optimise_new(opt_params, model_single, exposures_single, things_all, 300, nbatches=15, use_c=C_new)

# %%
inten = 10**params_history_resolved[-1]["resolved"]["N458_F110W"]
plt.figure(figsize=(10,10))
plt.imshow((inten-np.min(inten)))
plt.colorbar()
plt.savefig("deconvolution.png")

# %%
