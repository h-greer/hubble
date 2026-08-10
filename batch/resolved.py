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
import chainconsumer as cc


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
np.vstack((np.ones(5), np.arange(5))).T

# %%
class CursedResolvedSource(dl.sources.Source):
    distribution: Array
    position: Array
    pitch: float

    def __init__(self, distribution, pitch, position=np.zeros(2), **kwargs):
        self.distribution = distribution
        self.pitch = float(pitch)
        self.position = position
        super().__init__(**kwargs)
    
    def normalise(self):
        return self
    
    def model(self, optics, return_wf=False, return_psf=False):
        coords = dlu.nd_coords(self.distribution.shape, self.pitch, self.position)
        xs = coords[0].flatten()
        ys = coords[1].flatten()
        ds = self.distribution.flatten()
        # conv_psf = np.sum(
        #     jax.vmap(
        #         lambda xs, ys, ds: ds*optics.propagate(self.wavelengths, np.array([xs, ys]), self.weights)
        #     )(xs, ys, ds), 
        #     axis=0
        # )

        conv_psf = np.sum(
            jax.lax.map(
                lambda x: x[2]*optics.propagate(self.wavelengths, np.array([x[0], x[1]]), self.weights),
                np.stack((xs, ys, ds)).T,
                batch_size=64,
            ), 
            axis=0
        )





        # print(self.wavelengths)

        # conv_psf = jax.vmap(lambda xs, ys, ds: ds*optics.propagate(self.wavelengths, np.array([xs, ys]), self.weights))(xs, ys, ds)
        
        # print(conv_psf.shape)

        wf = optics.propagate(self.wavelengths, np.array([xs.mean(), ys.mean()]), self.weights, return_wf=True)
        if return_psf:
            return dl.PSF(conv_psf, wf.pixel_scale.mean())
        return conv_psf

# %%
class PointResolvedFit(ModelFit):
    wid: float

    def __init__(self, spectrum_basis, filter, wid):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)

        wvr, intenr = calc_throughput(filter, 1)

        self.source = dl.Scene([            
            ("resolved", CursedResolvedSource(
                wavelengths=wvr,
                spectrum=dl.Spectrum(wvr, intenr), 
                distribution=np.ones((wid, wid)),
                pitch=dlu.arcsec2rad(0.0432*2)#(1.6e-6/2.4)/2,
            )),
            ("point", dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))),
        ])
        self.wid = wid
    
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

        distribution = self.get_distribution(model, exposure)

        source = source.set("resolved.distribution",  distribution)
        
        return source

    def loglike(self, model, exposure, per_pix=False, return_im=False):


        if "resolved" in model.params.keys():
            dist = self.get_distribution(model, exposure)
            return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im)# + 0.1* L2_loss(dist) +  0.005*TV_loss(dist)
        
        return super().loglike(model, exposure, per_pix=per_pix, return_im=return_im)

# %%
wid = 40
oversample = 4

nwavels = 20
npoly=3

n_modes = 14

resolved_wid = 20

optics = NICMOSCoronagraph(512, wid, oversample, n_modes=n_modes)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))


ddir = '../data/data/'


vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]

assert vects.shape == (nwavels, npoly)


spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

extra_bad = np.full((wid,wid), False).at[20:30, 0:10].set(True)



exposures_single = [
    exposure_from_file(ddir + 'n8zu11epq_m_clc_calf.fits', PointResolvedFit(spectrum_basis, "F160W", wid=resolved_wid), crop=wid),
]


# %%
params = {
    "spectrum": {},
    "primary_opd": {},
    "primary_tilt": {},
    "cold_mask_opd": {},
    "cold_mask_tilt": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_shear": {},
    "cold_mask_scale": {},

    "bias": {},
    "occulter_radius": 0.76,
    "fnumber": 45.3,
    "resolved": {},
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*6)

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([0.00, 0.])
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([-0.3, -0.15])

    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.array([-160.])

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-13.,-7.])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 0.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.

    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((resolved_wid,resolved_wid))+0.
    


    # params["quadrature"][exp.fit.get_key(exp, "quadrature")] = np.array(0.)


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
# plot_comparison(model_single, params, exposures_single)

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)

def adam(lr, delay):
    return optax.adam(zdx.optimisation.delay(lr, delay))


g = 5e-2

things = {
    "spectrum": sgd(g*4., 20),
    # "cold_mask_shift": sgd(g*3, 100),
    
    "bias": sgd(g*6, 30),
    # "primary_opd": sgd(g*.05, 20),
    # "primary_opd": adam(0.1, 10),

    "cold_mask_opd": sgd(g*5, 20),

    "primary_tilt": sgd(g*1., 10),
    "cold_mask_tilt": sgd(g*1, 0),
    # #"jitter": opt(g*1, 120),


    # # "cold_mask_shear": sgd(g*2, 200),
    # "cold_mask_rot": sgd(g*10, 200),
    # "cold_mask_scale": sgd(g*20, 200),

    # # "quadrature": sgd(g*20, 400)

    # "occulter_radius": sgd(g*1, 200),
    # "fnumber": sgd(g*0.1, 220),

    "resolved": adam(3e-2, 50)
}

things_start = {
    "positions": sgd(g*5, 0),
}

groups = list(things.keys())

# %%
# orig_params = params.params
# opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
# losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 10)

# %%
# plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single)

# %%
orig_params = params.params #| params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new_resolved(opt_params, model_single, exposures_single, things, 100, nbatches=5)

# %%
plt.plot(losses[:])

# %%
losses[-1]

# %%
# plot_params(params_history, groups, xw = 3)
# plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, quadrature=False)

# %%
params_history[-1]

# %%
plt.figure(figsize=(10,10))
plt.imshow(np.flip(params_history[-1]["resolved"]["SZ-82_F160W"]))
plt.savefig("sz82.png")

# # %%
# model = ModelParams(params_history[-1]).inject(model_single)
# fit = exp.fit(model, exposures_single[0])
# exp = exposures_single_no_bad[0]

# # %%
# resid = (exp.data - fit)/exp.err
# rlim = np.nanmax(np.abs(resid))
# plt.imshow(resid, cmap="bwr", vmin=-rlim, vmax=rlim)
# plt.colorbar()

# # %%



