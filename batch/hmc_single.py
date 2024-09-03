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

ddir = "../data/MAST_2024-08-27T07_49_07.684Z/"
fname = ddir + 'HST/n8ku01ffq_cal.fits'


exposure = exposure_from_file(fname,SinglePointFit(),crop=wid)
exposures = [exposure]

#plt.imshow(exposure.data)
#plt.show()

params = {
    "fluxes": {},
    "positions": {},
    "aberrations": {},#np.zeros(8),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,
    "cold_mask_shift": {}, #np.asarray([-0.05, -0.05]),
    "cold_mask_rot": {},#np.asarray([np.pi/4]),
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
    "scale": 0.0431
}

for exp in exposures:
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["fluxes"][exp.fit.get_key(exp, "fluxes")] = np.nansum(exp.data)
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(19)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-0.12,-0.12])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = np.pi/4

model = set_array(NICMOSModel(exposures, params, optics, detector))


pixel_scale = dlu.arcsec2rad(0.0432)

print("yes")

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
        params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([npy.sample("X", dist.Normal(0, 1))*pixel_scale,npy.sample("Y", dist.Normal(0,1))*pixel_scale])
        params["fluxes"][exp.fit.get_key(exp, "fluxes")] = npy.sample("Flux", dist.Uniform(0, 3))*1e5
        params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(19).at[0].set(npy.sample("Defocus", dist.Normal(0, 10))*1e-9)
        params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-npy.sample("Cold X", dist.HalfNormal(0.1)),-npy.sample("Cold Y", dist.HalfNormal(0.1))])
        params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = npy.sample("Cold Rot", dist.Normal(np.pi/4, np.deg2rad(0.3)))


    params = ModelParams(params)

    mdl = params.inject(model)

    model_data = data.fit(mdl, data).flatten()


    img, err, bad = data.data.flatten(), data.err.flatten(), data.bad.flatten()

    image = np.where(bad, 0, img)
    error = np.where(bad, 1, err)


    with npy.plate("data", size=len(image)):
        image_d = dist.Normal(image, error)
        return npy.sample("psf", image_d, obs=model_data)



sampler = npy.infer.MCMC(
    npy.infer.NUTS(psf_model, init_strategy=npy.infer.init_to_mean, dense_mass=False),
    num_warmup=100,
    num_samples=100,
    #num_chains=6,
    #chain_method='vectorized'
    progress_bar=True,
)

sampler.run(jr.PRNGKey(1),exposures[0], model)

sampler.print_summary()

chain = cc.Chain.from_numpyro(sampler, name="numpyro chain", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)
#consumer = consumer.add_truth(cc.Truth(location={"X":-3e-7/pixel_scale, "Y":1e-7/pixel_scale, "Flux":5,"Cold X":0.08, "Cold Y":0.08, "Defocus":5, "Cold Rot":np.pi/4}))

fig = consumer.plotter.plot()
fig.savefig("chains_hmc_data.png")
plt.close()