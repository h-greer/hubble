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

#jax.config.update("jax_enable_x64", True)


# Optimisation imports
import zodiax as zdx
import optax

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

def interp(
    image: Array,
    knot_coords: Array,
    sample_coords: Array,
    method: str = "linear",
    fill: float = 0.0,
):

    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    return ipx.interp2d(
        ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill
    ).reshape(sample_coords[0].shape)

class PointResolvedFit(ModelFit):
    wid: float
    oversample: float
    def __init__(self, spectrum_basis, filter, wid, oversample):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)

        wvr, intenr = calc_throughput(filter, 3)

        self.source = dl.Scene([
            ("point", dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))),
            ("resolved", dl.ResolvedSource(
                wavelengths=wv,
                flux=1.,
                spectrum=dl.Spectrum(wvr, intenr), 
                distribution=np.ones((64*4, 64*4))
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

    def update_source(self, model, exposure):
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("point.spectrum.basis_weights", spectrum_coeffs)
        source = source.set("point.flux", source.point.spectrum.flux)
        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)
        source = source.set("point.position", position)
        source = source.set("resolved.position", position)
        

        distribution = 10**interp(model.get(exposure.fit.map_param(exposure, "resolved")), dlu.pixel_coords(self.wid, 1), dlu.pixel_coords(self.wid*self.oversample, 1))

        source = source.set("resolved.distribution",  distribution)
        source = source.set("resolved.flux",  distribution.sum())
        
        return source  

# %%
wid = 100
oversample = 4

nwavels = 10#13#6
npoly=5#10#2

n_zernikes = 15#12

optics = NICMOSOptics(1024, wid, oversample, n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)

basis_file = np.load("spectrum_basis.npy")[:,:npoly]
spectrum_basis = ipx.interp1d(np.linspace(0,1,nwavels), np.linspace(0,1,basis_file.shape[0]), basis_file)
spectrum_basis = spectrum_basis/np.sqrt(np.mean(spectrum_basis**2, axis=0))

#spectrum_basis = build_dct_basis(nwavels, npoly)

ddir = '../data/MAST_2024-09-19T06_48_02.332Z/HST/'


exposures_single = [
    exposure_from_file(ddir + "na2a05ttq_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05tuq_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05txq_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05tzq_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),

    exposure_from_file(ddir + "na2a05u2q_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u4q_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u7q_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),
    exposure_from_file(ddir + "na2a05u9q_cal.fits", PointResolvedFit(spectrum_basis, "F110W", wid=wid, oversample=oversample), crop=wid),

    # exposure_from_file(ddir + "na2a05ucq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05ueq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05uhq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05ujq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),

    # exposure_from_file(ddir + "na2a05umq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05uoq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05urq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),
    # exposure_from_file(ddir + "na2a05utq_cal.fits", PointResolvedFit(spectrum_basis, "F110W"), crop=wid),

    #exposure_from_file(ddir + "na2a05tvq_cal.fits", PointResolvedFit(spectrum_basis, "F160W"), crop=wid),
    #exposure_from_file(ddir + "na2a05twq_cal.fits", PointResolvedFit(spectrum_basis, "F160W"), crop=wid),
]


# %%
for e in exposures_single:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)
    print(e.exptime)
    print(e.data.shape)

# %%


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
    "scale": 0.043142,

    "softening": 2.,#0.1,
    "bias": {},
    "jitter": {},
    "resolved": {}
    #"displacement": 1.#1e5#{}
}


for idx, exp in enumerate(exposures_single):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])#positions_dict[exp.fit.get_key(exp, "positions")]#np.asarray(positions[idx])#np.asarray([0.49162114, -0.5632928])#np.asarray([ 0.45184505, -0.8391668 ])#np.asarray([-0.2,0.4])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = np.zeros(npoly).at[0].set(np.log10(np.nansum(exp.data)/nwavels))
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)#np.asarray([0., 24.884588  , -25.489779  , -17.15699   , -21.790146  ,
    #      -4.592212  ,  -4.832893  ,  19.196083  ,   0.37983412,
    #       7.0756216 ,   0.30277824,  -6.330534])#np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8.,8.])#np.asarray([9.599048, 6.196583])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    #params["displacement"][exp.fit.get_key(exp, "displacement")] = 1e6

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.06
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample

    params["resolved"][exp.fit.get_key(exp, "resolved")] = np.zeros((wid,wid))-2.
    


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
print(np.nansum(exp.data))

# %%
np.log10(7336/np.sum(10**spectrum_basis[:,0]))

# %%
print(params.params)

# %%
plot_comparison(model_single, params, exposures_single, save="deconvolution/startup")


# %%
def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

base_sgd2 = lambda vals: optax.noisy_sgd(vals)

opts = lambda lr, start, *schedule: base_sgd2(scheduler(lr, start, *schedule))


base_adam = lambda vals: optax.adam(vals)
opta = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))





def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 5e-2

things = {
    "positions": opt(g*5, 0),
    "spectrum": opt(g*2, 10),
    "cold_mask_shift": opt(g*10, 30),
    "bias": opt(g*5, 20),
    "aberrations": opt(g*0.1, 50),
}

things_all = {
    "positions": opt(g*5, 0),
    "spectrum": opt(g*2, 0),#opt(g*2, 10),#opt(g*2, 10),#, (20, 1.5)),
    "cold_mask_shift": opt(g*10, 0),
    "bias": opt(g*5, 0),
    #"aberrations": opt(g*0.8, 0),
    "resolved": opta(1e-2, 0),
}

things_start = {
    "positions": opt(g*5, 0),
}

groups = list(things.keys())

# %%
list(params["resolved"].values())

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

def get_optimiser_new(model_params, optimisers):
    param_spec = ModelParams({param: param for param in model_params.keys()})
    optim = optax.multi_transform(optimisers, param_spec)
    return optim, optim.init(model_params)

def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    priors = 2e0*np.array([L2_loss(10**x) + 0.4*TV_loss(10**x) for x in params["resolved"].values()]).sum() if "resolved" in params.keys() else 0.
    loss = np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))
    return loss + priors

def optimise(params, model, exposures, things, niter, reduce_ram=False, recalculate=False):
    paths = [x for x in list(things.keys()) if x != "resolved"]
    paths_full = list(things.keys())
    optimisers = [things[i] for i in paths]

    print("Calculating Fishers")

    fish = lambda model, exposure, params: fisher_fn(model, exposure, params, reduce_ram=reduce_ram)

    #fishers = calc_fishers(model, exposures, paths)
    fishers = calc_fishers(model, exposures, paths, fisher_fn, recalculate=recalculate)
    model_params = ModelParams({p: model.get(p) for p in things.keys()})
    lrs = populate_lr_model(fishers, exposures, model_params)
    

    optim, opt_state = get_optimiser_new(
        model_params, things
    )

    jit_loss = zdx.filter_value_and_grad(paths_full)(loss_fn)

    print("Fitting Model")

    @zdx.filter_jit
    def update(model_params, exposures, model, lrs, opt_state):
        grads = jax.tree.map(lambda x: x * 0.0, model_params)

        loss, new_grads = jit_loss(model_params,exposures, model)
        grads += new_grads
        grads = jax.tree.map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs.params))
        updates, opt_state = optim.update(grads, opt_state)
        model_params = zdx.apply_updates(model_params, updates)
        return loss, model_params, opt_state



    losses, models = [], []
    for i in tqdm(range(niter)):
        loss, model_params, opt_state = update(model_params, exposures, model, lrs, opt_state)
        models.append(model_params)
        losses.append(loss)

    
    return losses, models

# %%
initial_losses, initial_models = optimise(params, model_single, exposures_single, things_start, 10, recalculate=True)

# %%
plot_comparison(model_single, initial_models[-1], exposures_single, save="deconvolution/initial_comparison")

# %%
initial_models[-1].params

# %%
int_losses, int_models = optimise(initial_models[-1].inject(params), initial_models[-1].inject(model_single), exposures_single, things, 300  , recalculate=True)

# %%


# %%
models_pd = [jax.tree.map(lambda x,y: (x-y)/y, int_models[i], int_models[-1]) for i in range(len(int_models))]

# %%
plot_params(int_models, groups, xw = 3, save="deconvolution/params")
plot_comparison(model_single, int_models[-1], exposures_single, save="deconvolution/intermediate_comparison")

plt.figure()
plt.plot(np.asarray(int_losses[:])/(len(exposures_single)*wid**2))
plt.savefig("deconvolution/loss-intermediate.png")

# %%
print(int_models[-1].params)

# %%
losses, models = optimise(int_models[-1].inject(params), int_models[-1].inject(model_single), exposures_single, things_all, 500, recalculate=False)

# %%
plot_comparison(model_single, models[-1].combine(int_models[-1]), exposures_single, save="deconvolution/final_comparison")

# %%

plt.figure()
plt.plot(np.asarray(losses[:])/(len(exposures_single)*wid**2))
plt.savefig("deconvolution/loss-final.png")

# %%
plt.figure(figsize=(10,10))
plt.imshow(10**models[-1]["resolved"]["N458_F110W"])
plt.colorbar()
plt.savefig("deconvolution/disk.png")