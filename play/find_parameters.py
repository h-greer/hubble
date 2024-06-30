# Set CPU count for numpyro multi-chain multi-thread
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'


# Basic imports
import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from optax._src.alias import transform

# Optimisation imports
import zodiax as zdx
import optax

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72


import numpyro as npy
import numpyro.distributions as dist

import chainconsumer as cc

# all the PSF things
from hubble_psf import *


# optical system parameters

hubble_optics = make_hubble_optics()

wavels = 1e-6 * np.linspace(1, 1.2, 10)

weights = np.array([np.linspace(0.5, 1.5, 10), np.linspace(1.5, 0.5, 10)])


binary = dl.BinarySource(
    wavels,
    mean_flux = 1e7,
    contrast = 5,
    weights=weights,
    separation = dlu.arcsec2rad(0.1)
)

"""
psf = binary.model(hubble_optics)

plt.imshow(psf**0.5)
plt.show()
"""

# TODO: add detector behaviours

img_telescope = dl.Telescope(hubble_optics, ('source',binary))

psf = img_telescope.model()
psf_photon = jr.poisson(jr.PRNGKey(0), psf)
bg_noise = 3*jr.normal(jr.PRNGKey(0), psf_photon.shape)

data = psf_photon + np.abs(bg_noise)

#plt.imshow(data)
#plt.show()

model_optics = make_hubble_optics()

# probably need to infer a few more things

paths = [
    "aberrations.coefficients",
    "mask.transformation.translation"]

new_model = model_optics.multiply(paths, 0)

model_binary = dl.BinarySource(
    wavels,
    mean_flux = 1e7,
    contrast = 1,
    weights=weights,
    separation=0,#dlu.arcsec2rad(1),
    position_angle=0,
)

model_system = dl.Telescope(new_model, model_binary)

paths = ["contrast","position_angle","mean_flux","separation","aberrations.coefficients", "mask.transformation.translation"]

@zdx.filter_value_and_grad(paths)
def loss_fn(model,data):
    psf = model.model()
    return -jsp.stats.poisson.logpmf(data,psf).mean()


jit_loss = zdx.filter_jit(loss_fn)

model = model_system

groups = [["contrast","position_angle"],"mean_flux","separation","aberrations.coefficients", "mask.transformation.translation"]

optimisers = [optax.adam(1e-1), optax.adam(1e6), optax.adam(1e-7), optax.adam(1e-9), optax.adam(5e-2)]

# Construct our optimiser objects
optim, opt_state = zdx.get_optimiser(
    model, groups, optimisers
    #model, paths, optax.adam(1e-8)
)


# A basic optimisation loop
losses, models = [], []
for i in tqdm(range(1000), desc="Loss: "):
    # Calculate the loss gradients, and update the model
    loss, grads = jit_loss(model, data)
    updates, opt_state = optim.update(grads, opt_state)
    model = zdx.apply_updates(model, updates)

    # save results
    models.append(model)
    losses.append(loss)


### HMC STUFF AAAAAAAAAAAA

source_paths = ["contrast","position_angle","separation","mean_flux"]

#source_paths = ["contrast", "position_angle"]


def psf_model(data, model):
    source_params = [
        npy.sample("contrast", dist.Uniform(4, 5)),
        npy.sample("theta", dist.Uniform(0, np.pi)),
        npy.sample("sep", dist.Uniform(dlu.arcsec2rad(0.09),dlu.arcsec2rad(0.12))),#dist.Uniform(dlu.arcsec2rad(0.01),dlu.arcsec2rad(1))),
        npy.sample("flux", dist.Uniform(0.9,1.1))
    ]
    aberration_params = np.asarray([
        npy.sample("astig_x", dist.Uniform(-1e-7, 1e-7)),
        npy.sample("astig_y", dist.Uniform(-1e-7, 1e-7)),
        npy.sample("defocus", dist.Uniform(-1e-7, 1e-7)),
    ])

    cold_mask_offset = np.asarray([
        npy.sample("cold_x", dist.Uniform(0.07,0.09)),
        npy.sample("cold_y", dist.Uniform(0.0, 0.02))
    ])

    with npy.plate("data", len(data.flatten())):
        poisson_model = dist.Poisson(
            model.set(source_paths,source_params)
            .set("aberrations.coefficients",aberration_params)\
            .set("mask.transformation.translation",cold_mask_offset).model().flatten()
        )
        return npy.sample("psf", poisson_model, obs=data.flatten())


"""
sampler = npy.infer.MCMC(
    npy.infer.NUTS(psf_model),
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
)

sampler.run(jr.PRNGKey(0), data, img_telescope)#, init_params = [5.0, dlu.arcsec2rad(0.1)])

chain = cc.Chain.from_numpyro(sampler, "test", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)
consumer.plotter.plot()
plt.show()
"""
