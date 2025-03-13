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

# dLux imports
import dLux as dl
import dLux.utils as dlu

#jax.config.update("jax_enable_x64", True)


import chainconsumer as cc

import numpyro as npy
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

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
from fisher import *
from stats import posterior
from fitting import *
from plotting import *

#%matplotlib inline
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

# %%
wid = 64
oversample = 4

nwavels = 20
npoly=5

optics = NICMOSOptics(512, wid, oversample)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"


files = [
    #'n8yj53vfq_cal.fits'

    'n8yj59glq_cal.fits',

]

exposures_single = [exposure_from_file(ddir + file, SinglePointPolySpectrumFit(nwavels), crop=wid) for file in files]

exposures_binary = [exposure_from_file(ddir + file, BinaryPolySpectrumFit(nwavels), crop=wid) for file in files]

# %%
params = {
    #"fluxes": {},
    "positions": {},
    "spectrum": {},
    "aberrations": {},
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
    "scale": 0.0432,
    "rot": 0.,
    "softening": 2.,
    "bias": {}
}

for exp in exposures_single:
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = np.zeros(npoly).at[0].set(1)*np.log10(np.nansum(exp.data)/nwavels)
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(26)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = -np.asarray([-0.06, -0.06])*1e2
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. #+ 180.
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.

model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

base_adam = lambda vals: optax.adam(vals)
opta = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))


def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 5e-3

things = {
    #"fluxes" : opt(g*20,10),
    "positions": opt(g*30, 0),
    "spectrum": opt(g*20, 10),#, (20, 1.5)),
    "cold_mask_shift": opt(g*50, 120),
    "cold_mask_rot": opt(g*500, 120),
    #"cold_mask_scale": opt(g*3000, 150),
    #"cold_mask_shear": opt(g*1000, 150),
    #"primary_scale": opt(g*100, 150),
    #"primary_rot": opt(g*100, 150),
    #"primary_shear": opt(g*100, 150),
    #"aberrations": opt(g*100,20),#, (150, g*0.2)),
    #"spectrum": opt(g*50, 20)#, (150, g*200), (200, g*300), (250, g*400)),
    #"spectrum": opt(g*0.01, 20),
    #"softening": opt(g*1e3, 200),
    #"breathing": opt(g*1000,150),
    #"rot": opt(g*100, 50),
    "bias": opt(g*30, 20)
}

groups = list(things.keys())
paths = flatten(groups)
optimisers = [things[i] for i in groups]
groups = [list(x) if isinstance(x, tuple) else x for x in groups]

# %%
losses, models = optimise(params, model_single, exposures_single, things, 30)

# %%
#plt.plot(losses[-20:])

# %%
#plot_params(models, groups, xw = 3)
#plot_comparison(model_single, models[-1], exposures_single)

# %%


# %%
#potato

# %%
#plot_spectra(model_single, models[-1], exposures_single)

# %%
def tree_mul(spec, val):
    return jtu.tree_map(lambda x: x*val, spec)

def tree_sum(spec, val):
    return jtu.tree_map(lambda x: x+val, spec)


def extract_binary_params(params, exposures, x, y, theta, r, flux, contrast):
    #fluxes = dlu.fluxes_from_contrast(flux, contrast)
    param_dict = params.params.copy()
    param_dict["primary_spectrum"] = param_dict["spectrum"]
    param_dict["secondary_spectrum"] = param_dict["spectrum"]
    param_dict["fluxes"] = dlu.list2dictionary([(exp.fit.get_key(exp, "fluxes"), flux) for exp in exposures], ordered=True)#tree_mul(param_dict["spectrum"], fluxes[0])
    param_dict["contrast"] = dlu.list2dictionary([(exp.fit.get_key(exp, "contrast"), contrast) for exp in exposures], ordered=True) #tree_mul(param_dict["spectrum"], fluxes[1])
    param_dict["positions"] = tree_sum(param_dict["positions"], np.array([x,y]))
    param_dict["separation"] = r#dlu.list2dictionary([(exp.fit.get_key(exp, "separation"), r) for exp in exposures])
    param_dict["position_angle"] = theta #dlu.list2dictionary([(exp.fit.get_key(exp, "position_angle"), theta) for exp in exposures])
    return ModelParams(param_dict)


"""def inject_binary_values(x, y, theta, r, flux, contrast, initial_params):
    fluxes = dlu.fluxes_from_contrast(flux, contrast)
    injected_params = ModelParams({
        "primary_spectrum": spectra_mul(initial_params.get("primary_spectrum"),fluxes[0]),
        "secondary_spectrum": spectra_mul(initial_params.get("secondary_spectrum"),fluxes[1]),
        "positions": np.asarray([x,y]),
        "position_angle": theta,
        "separation": r
    })
    return injected_params.inject(initial_params)
"""

# %%
binary_params = extract_binary_params(models[-1], exposures_binary, 0., 0., 0., 0., 0., 1.)
model_binary = set_array(NICMOSModel(exposures_binary, binary_params.params, optics, detector))


# %%
@zdx.filter_jit
def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))


# %%
#jax.profiler.start_server(1234)

# %%
#with jax.profiler.trace("/tmp/tensorboard"):
#    x = loss_fn(params, exposures_single, model_single)
#    x.block_until_ready()


# %%
#stop

# %%

@zdx.filter_jit
def fit_binary_flux(params, exposures, x, y, theta, r, contrast):
    base_params = extract_binary_params(params, exposures, x, y, theta, r, 0., contrast)

    mdl = params.inject(model_binary)

    
    mean_flux = 0.

    for exp in exposures:

        #print(params.params["spectrum"])

        spec = params.get(exp.fit.map_param(exp, "spectrum"))[0]
        #print(spec)

        psf = exp.fit(mdl,exp).flatten()
        psf= np.where(exp.bad.flatten(), 0., psf)

        psf = psf/np.sum(psf)
        #psf.at[exp.bad.flatten()].set(0.)

        data = exp.data.flatten()
        #data= data.at[exp.bad.flatten()].set(0.)
        data= np.where(exp.bad.flatten(), 0., data)


        design = np.transpose(np.vstack((np.ones(len(psf)), psf)))

        flux_raw, _, _, _ = np.linalg.lstsq(design, data)
        
        true_flux= np.log10(flux_raw[1]/nwavels * 2/(1+contrast)) - spec
        #print(true_flux)

        mean_flux += true_flux/len(exposures)
    
    flux_params = extract_binary_params(params, exposures, x, y, theta, r, mean_flux, contrast)

    return loss_fn(flux_params, exposures, model_binary), flux_params  



# %%
binary_params

# %%
#x_vals = np.linspace(-5, 5, 4)
theta_vals = 360* np.arange(8)/8#np.arange(4)*np.pi/2#np.linspace(0, 2*np.pi, 4)
r_vals = np.arange(20)/2#np.linspace(0,15,20)#np.asarray([1.5, 3])#np.linspace(0, 5, 2)
contrast_vals = [1.]#10**np.linspace(-1, 1, 21)
min_loss = np.inf
best_params = None

#for x in x_vals:
    #for y in y_vals:
for theta in theta_vals:
    for r in r_vals:
        for cnt in contrast_vals:
            ang = dlu.deg2rad(theta)
            x = -r*np.sin(ang)/2
            y = -r*np.cos(ang)/2
            #print(dlu.positions_from_sep(np.asarray([x,y]), r, theta))
            loss, params = fit_binary_flux(models[-1], exposures_binary, x, y, theta, r, cnt)

            print(loss)
            if loss < min_loss and min_loss != 0.0:
                min_loss = loss
                best_params = params
        

# %%
best_params

# %%
#plot_comparison(model_binary, best_params, exposures_binary)

# %%
g = 5e-3
things = {
    #"fluxes" : opt(g*20,10),
    "positions": opt(g*100, 0),
    "separation": opt(g*30, 0),
    "position_angle": opt(g*1e-2, 10),
    "primary_spectrum": opt(g*50, 20),
    "secondary_spectrum": opt(g*50, 20),#, (20, 1.5)),
    "cold_mask_shift": opt(g*50, 60),
    "cold_mask_rot": opt(g*10, 60),
    "aberrations": opt(g*10,30),#, (150, g*0.2)),
    "bias": opt(g*20, 40)
}
groups = list(things.keys())


# %%
losses, models = optimise(best_params, set_array(model_binary), exposures_binary, things, 150)

# %%

# %%
losses[-1]

# %%
#plot_params(models, groups, xw = 3)
#plot_comparison(model_binary, models[-1], exposures_binary)

# %%


# %%
models[-1].params

# %%


# %%


# %%
rc = True
fishers = calc_fishers(models[-1].inject(model_binary), exposures_binary, groups, recalculate=rc)

# %%
fishers

# %%
aberration_names = [dlu.zernike_name(x) for x in range(40)]
poly_names = ["poly "+ x for x in ["0", "1", "2", "3", "4"]]

# %%
models[-1].params

# %%
new_params = ModelParams({"separation": 12.})

# %%
new_params + models[-1].params

# %%
ModelParams(models[-1].params | new_params).params

# %%
np.sqrt((np.linalg.inv(fishers['n8yj59glq']['position_angle'])))[0][0]

# %%
np.sqrt((np.linalg.inv(fishers['n8yj59glq']['separation'])))[0][0]

# %%
np.diag(np.sqrt(np.linalg.inv(fishers['n8yj59glq']['secondary_spectrum'])))

# %%


# %%
stop

# %%

def make_psf_model(modelparams, fishers):

    #@zdx.filter_jit
    def psf_model(data, model):

        params = {
            #"primary_spectrum": {},
            #"secondary_spectrum": {},
            "positions": {},
        }

        exp = exposures_binary[0]


        pos_mean = modelparams.get(exp.map_param("positions"))
        pos_std = np.diag(np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['positions']))))

        primary_mean = modelparams.get(exp.map_param("primary_spectrum"))
        primary_std = np.diag(np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['primary_spectrum']))))

        secondary_mean = modelparams.get(exp.map_param("secondary_spectrum"))
        secondary_std = np.diag(np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['secondary_spectrum']))))

        position_angle = npy.sample("position_raw", dist.Normal(0,1))
        separation = npy.sample("separation_raw", dist.Normal(0,1))
        X = npy.sample("x_raw", dist.Normal(0,1))
        Y = npy.sample("y_raw", dist.Normal(0,1))

        #primary_spectrum = npy.sample("primary_raw", dist.Normal(np.zeros(5), np.ones(5)))
        #secondary_spectrum = npy.sample("secondary_raw", dist.Normal(np.zeros(5), np.ones(5)))

        params["position_angle"] = npy.deterministic("Position Angle", modelparams.get("position_angle") + position_angle*np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['position_angle'])))[0][0])

        params["separation"] = npy.deterministic("Separation", modelparams.get("separation") + separation* np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['separation'])))[0][0])

        #params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([npy.sample("Cold X", dist.Normal(0, 1))*np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['cold_mask_shift'])))[0][0] + modelparams.get(exp.map_param("cold_mask_shift"))[0], npy.sample("Cold Y", dist.Normal(0, 1))*np.sqrt(np.abs(np.linalg.inv(fishers['n8yj59glq']['cold_mask_shift'])))[1][1] + modelparams.get(exp.map_param("cold_mask_shift"))[1]])

        
        
        params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([npy.deterministic("X", pos_mean[0]+ X*pos_std[0]), npy.deterministic("Y", pos_mean[1]+ Y*pos_std[1])])

            #params["primary_spectrum"][exp.fit.get_key(exp, "primary_spectrum")] = np.asarray([
            #    npy.deterministic("primary " +poly_names[x], primary_mean[i] + primary_spectrum[i]*primary_std[i]) for i, x in enumerate#(range(0,5))    
            #])

            #params["secondary_spectrum"][exp.fit.get_key(exp, "secondary_spectrum")] = np.asarray([
            #    npy.deterministic("secondary " +poly_names[x], secondary_mean[i] + secondary_spectrum[i]*secondary_std[i]) for i, x in enumerate(range(0,5))    
            #])




            #params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.asarray([
            #    npy.sample(aberration_names[x], dist.Normal(0,1))/np.sqrt(fishers['n8yj59glq']['aberrations'][i][i]) + modelparams.get(exp.map_param("aberrations"))[i] for i, x in enumerate(range(4,30))
                
            #])

        params = ModelParams(modelparams.params | params)

        #params.replace(modelparams)

        
        
        with npy.plate("data", size=len(data.data.flatten())):

            mdl = params.inject(model)
            model_data = data.fit(mdl, data).flatten()
            img, err, bad = data.data.flatten(), data.err.flatten(), data.bad.flatten()
            image = np.where(bad, 0, img)
            error = np.where(bad, 1e5, err)
            
            image_d = dist.Normal(image, error)
            return npy.sample("psf", image_d, obs=np.where(bad,0,model_data))
    
    return psf_model



sampler = npy.infer.MCMC(
    npy.infer.NUTS(make_psf_model(models[-1], jtu.tree_map(lambda x: np.abs(x), fishers)), 
                   init_strategy=npy.infer.init_to_mean,
                    dense_mass=False,
                    max_tree_depth = 5),
    num_warmup=500,
    num_samples=500,
    #num_chains=6,
    #chain_method='vectorized',
    progress_bar=True,
    #jit_model_args=True,
)

sampler.run(jr.PRNGKey(0),exposures_binary[0], model_binary)

sampler.print_summary()

chain = cc.Chain.from_numpyro(sampler, name="numpyro chain", color="blue")
consumer = cc.ChainConsumer().add_chain(chain)
#consumer = consumer.add_truth(cc.Truth(location={"X":-3e-7/pixel_scale, "Y":1e-7/pixel_scale, "Flux":5,"Cold X":0.08, "Cold Y":0.08, "Defocus":5, "Cold Rot":np.pi/4}))

fig = consumer.plotter.plot()
fig.savefig("fixed_turbo.png")
plt.close()
