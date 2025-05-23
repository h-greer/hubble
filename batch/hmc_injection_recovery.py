# import trick

import sys
sys.path.insert(0, '../')

# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import jax
import jax.tree_util as jtu

jax.config.update("jax_enable_x64", True)

import numpy



import numpyro as npy
import numpyro.distributions as dist

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

from detectors import *
from apertures import *
from models import *

import chainconsumer as cc


def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

wid = 64
oversample = 4

optics = NICMOSOptics(512, wid, oversample)

detector = NICMOSDetector(oversample, wid)

flt = "F145M"

injected_params = {
    "fluxes": {f"injected_{flt}": np.asarray(5e9)},
    "positions": {f"injected_{flt}": np.asarray([-3e-7,1e-7])},
    "aberrations": {f"injected_{flt}":np.zeros(19).at[0].set(5e-9)},#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9},
    "cold_mask_shift": {f"injected_{flt}":np.asarray([-0.08, -0.08])},
    "cold_mask_rot": {f"injected_{flt}":np.asarray([np.pi/4])},#np.asarray([np.pi/4+dlu.deg2rad(0.8)])},
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
}

injected_exposure = InjectedExposure("injected",flt,SinglePointFit())
exposures = [injected_exposure]

model = set_array(NICMOSModel(exposures, injected_params, optics, detector))
for e in exposures:
    e.inject(model, 100)


pixel_scale = dlu.arcsec2rad(0.0432)


def psf_model(data, model):

    params = {
        "fluxes": {},
        "positions": {},
        "aberrations": {},
        "cold_mask_shift": {}, 
        "cold_mask_rot": {},
        "outer_radius": 1.2*0.955,
        "secondary_radius": 0.372*1.2,
        "spider_width": 0.077*1.2,
    }

    for exp in exposures:
        params["positions"]=injected_params["positions"]#[exp.fit.get_key(exp, "positions")] = np.asarray([npy.sample("X", dist.Normal(0, 0.8))*pixel_scale,npy.sample("Y", dist.Normal(0,0.8))*pixel_scale])
        params["fluxes"]=injected_params["fluxes"]#[exp.fit.get_key(exp, "fluxes")] = npy.sample("Flux", dist.Uniform(4, 6))*1e9
        params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(19).at[0].set(npy.sample("Defocus", dist.Uniform(-10, 10))*1e-9)
        params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([npy.sample("Cold X", dist.Uniform(-10, -6))*1e-2,npy.sample("Cold Y", dist.Uniform(-10, -6))*1e-2])
        params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = np.pi/4# npy.sample("Cold Rot", dist.Normal(np.pi/4, np.deg2rad(0.3)))


    params = ModelParams(params)

    mdl = params.inject(model)

    

    img, err, bad = data.data.flatten(), data.err.flatten(), data.bad.flatten()

    with npy.plate("data", size=len(img.flatten())):
        model_data = data.fit(mdl, data).flatten()
        image = dist.Normal(img.flatten(), err.flatten())
        return npy.sample("psf", image, obs=model_data)



sampler = npy.infer.MCMC(
    #npy.infer.NUTS(psf_model, init_strategy=npy.infer.init_to_value(site=None,values={"Cold X":-8,"Cold Y":-8, "X":0.0, "Y": 0.0, "Flux":np.nansum(exposures[0].data)/1e9, "Cold Rot": np.pi/4}), dense_mass=False),
    npy.infer.NUTS(psf_model, init_strategy=npy.infer.init_to_sample, dense_mass=False),
    num_warmup=1000,
    num_samples=1000,
    #num_chains=6,
    #chain_method='vectorized'
    #progress_bar=False,
)

sampler.run(jr.PRNGKey(1),exposures[0], model)

sampler.print_summary()

chain = cc.Chain.from_numpyro(sampler, name="numpyro chain", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)
consumer = consumer.add_truth(cc.Truth(location={"X":-3e-7/pixel_scale, "Y":1e-7/pixel_scale, "Flux":5,"Cold X":-0.08, "Cold Y":-0.08, "Defocus":5, "Cold Rot":np.pi/4}))

fig = consumer.plotter.plot()
fig.savefig("chains_hmc_new.png")
plt.close()