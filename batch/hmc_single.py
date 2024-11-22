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




import numpyro as npy
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

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

wid = 16
oversample = 4

optics = NICMOSOptics(512, wid, oversample)

detector = NICMOSDetector(oversample, wid)

#ddir = "../data/MAST_2024-08-27T07_49_07.684Z/"
#fname = ddir + 'HST/n8ku01ffq_cal.fits'

ddir = "../data/MAST_2024-09-08T07_59_18.213Z/"
fname = ddir + 'HST/n43ca5feq_cal.fits'

exposure = exposure_from_file(fname,SinglePointFit(),crop=wid)
exposures = [exposure]
params = {
    "fluxes": {},
    "positions": {},
    "contrast": {},
    "aberrations": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_scale": {},
    "cold_mask_shear": {},
    "primary_scale": {},
    "primary_rot": {},
    "primary_shear": {},
    "slope": {},
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
    "scale": 0.0434735,
    "rot": 0.0
}

for exp in exposures:
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.6,-0.7])#np.asarray([0.,-12.])#np.asarray([3,2.])#np.asarray([3.,2.])#np.asarray([-3.,-2.])##np.asarray([-8.,-2.])#np.asarray([-6.,-7.])
    params["fluxes"][exp.fit.get_key(exp, "fluxes")] = np.log10(np.nansum(exp.data))
    params["slope"][exp.fit.get_key(exp, "slope")] = np.zeros(5)#.at[0].set(1)
    #params["aberrations"] = injected_params["aberrations"]
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(26)#jr.uniform(jr.key(0), (8,),minval=-4e-8, maxval=4e-8) #np.asarray([-8.59023084e-10,  1.77049982e-09, -4.45293089e-09, -3.70890613e-08,2.03658617e-08,  1.08092528e-08, -2.77077727e-09,  1.86458672e-09])*0.9#jr.uniform(jr.key(0), (8,),minval=-2e-8, maxval=2e-8)#np.zeros(8)#np.ones(8)*1e-8
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([0.06, 0.06])*1e2
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = 45.
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    #params["contrast"][exp.fit.get_key(exp, "contrast")] = 1.

model = set_array(NICMOSModel(exposures, params, optics, detector))
params = ModelParams(params)

mdl = params.inject(model)
#print(mdl)

#psf = exposures[0].fit(mdl,exposures[0])

#plt.imshow(psf)#.fit(mdl, exposures[0]))
#plt.show()



#print(exposures[0].data)

#pixel_scale = dlu.arcsec2rad(0.0432)

#print("yes")

def psf_model(data, model):

    params = {
        "fluxes": {},
        "positions": {},
        #"aberrations": {},#np.zeros(8),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,
        #"cold_mask_shift": {}, #np.asarray([-0.05, -0.05]),
        #"cold_mask_rot": {},#np.asarray([np.pi/4]),
        #"slope": {},
        #"outer_radius": 1.2*0.955,
        #"secondary_radius": 0.372*1.2,
        #"spider_width": 0.077*1.2,
        #"scale": 0.0431
    }

    for exp in exposures:
        params["fluxes"][exp.fit.get_key(exp, "fluxes")] = npy.sample("Flux", dist.Uniform(-4,5))#*1e5
        #params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([npy.sample("Cold X", dist.Uniform(5, 12)), npy.sample("Cold Y", dist.Uniform(5, 12))])
        params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([npy.sample("X", dist.Uniform(-1,1)), npy.sample("Y", dist.Uniform(-1, 1))])

    params = ModelParams(params)

    
    
    with npy.plate("data", size=len(data.data.flatten())):

        mdl = params.inject(model)

        model_data = data.fit(mdl, data).flatten()


        img, err, bad = data.data.flatten(), data.err.flatten(), data.bad.flatten()

        image = np.where(bad, 0, img)
        error = np.where(bad, 1e5, err)
        
        image_d = dist.Normal(image, error*200)
        return npy.sample("psf", image_d, obs=model_data)#np.where(bad,0,model_data))


#config = {"flux": LocScaleReparam(centered=0), "X": LocScaleReparam(centered=0), "Y": LocScaleReparam(centered=0)}

#model2 = npy.handlers.reparam(psf_model, config=config)


sampler = npy.infer.MCMC(
    npy.infer.NUTS(psf_model, 
                   #init_strategy=npy.infer.init_to_value(site=None, values = {"X": 0.6, "Y":-0.7, "Flux": np.log10(np.nansum(exp.data)), "Cold X": 6., "Cold Y": 6.}),
                   init_strategy=npy.infer.init_to_sample,
                    dense_mass=False),
    num_warmup=200,
    num_samples=200,
    #num_chains=6,
    #chain_method='vectorized'
    progress_bar=True,
)

sampler.run(jr.PRNGKey(0),exposures[0], model)

sampler.print_summary()

chain = cc.Chain.from_numpyro(sampler, name="numpyro chain", color="teal")
consumer = cc.ChainConsumer().add_chain(chain)
#consumer = consumer.add_truth(cc.Truth(location={"X":-3e-7/pixel_scale, "Y":1e-7/pixel_scale, "Flux":5,"Cold X":0.08, "Cold Y":0.08, "Defocus":5, "Cold Rot":np.pi/4}))

fig = consumer.plotter.plot()
#fig.savefig("chains_hmc_data.png")
#plt.close()

plt.show()