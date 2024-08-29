# import trick

import sys
sys.path.insert(0, '../')

# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import jax
import jax.tree_util as jtu

#jax.config.update("jax_enable_x64", True)

import numpy



import numpyro as npy
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler

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
    "fluxes": {f"injected_{flt}": np.asarray(5e8)},
    "positions": {f"injected_{flt}": np.asarray([-3e-7,1e-7])},
    "aberrations": {f"injected_{flt}":np.zeros(19)},#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9},
    "cold_mask_shift": {f"injected_{flt}":np.asarray([-0.05, -0.05])},
    "cold_mask_rot": {f"injected_{flt}":np.asarray([np.pi/4+dlu.deg2rad(0.8)])},
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
}

injected_exposure = InjectedExposure("injected",flt,SinglePointFit())
exposures = [injected_exposure]

model = set_array(NICMOSModel(exposures, injected_params, optics, detector))
for e in exposures:
    e.inject(model, 5)


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
        params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([npy.sample("X", dist.Uniform(-2, 2))*pixel_scale,npy.sample("Y", dist.Uniform(-2,2))*pixel_scale])
        params["fluxes"][exp.fit.get_key(exp, "fluxes")] = 5e8#npy.sample("flux", dist.Uniform(4, 6))*1e8
        params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(19)
        params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-0.05,-0.05])
        params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = np.pi/4


    params = ModelParams(params)

    mdl = params.inject(model)

    model_data = data.fit(mdl, data).flatten()

    #model = model.set(list(samplers.keys()),list(samplers.values()))

    #for key in samplers:
    #    model = model.set(key, samplers[key])

    #model_data = model.model().flatten()

    img, err, bad = data.data.flatten(), data.err.flatten(), data.bad.flatten()

    #img = np.where(bad, 0, img)
    #err = np.where(bad, 1e10, err)

    with npy.plate("data", size=len(img.flatten())):
        image = dist.Normal(img.flatten(), err.flatten())
        return npy.sample("psf", image, obs=model_data)


nlive = (4000 + 4000)//4

ns = NestedSampler(psf_model,
                   constructor_kwargs={"num_live_points" : nlive, "max_samples": nlive*4},
                   termination_kwargs={'live_evidence_frac': 0.01})

ns.run(jr.PRNGKey(100),exposures[0], model)

ns.print_summary()


samples = ns.get_samples(jr.PRNGKey(1), num_samples=nlive)

chain = cc.Chain.from_numpyro(samples, "numpyro chain", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)

fig = consumer.plotter.plot()
fig.savefig("ns_test.png")
plt.show()