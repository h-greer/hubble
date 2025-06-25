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

from detectors import *
from apertures import *
from models import *
from fisher import *
from stats import posterior
from fitting import *
import glob

import matplotlib.pyplot as plt
import matplotlib

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

extra_bad = None

wid = 64
oversample = 4

nwavels = 3
npoly=1

n_zernikes = 12-1#12-1

optics = NICMOSFresnelOptics(512, wid, oversample, defocus =0., n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)



number = int(sys.argv[1])-1

ddir = "../data/MAST_2025-06-24T0210/HST/"

dfiles = glob.glob(ddir+"*_asc.fits")
dfiles.sort()

files = [x[0]+"_cal.fits" for x in fits.getdata(dfiles[number], ext=1)[:-1]]
print(files)

"""

FILE THINGS


"""

exposures_single = [exposure_from_file(ddir + file, SinglePointPolySpectrumFit(nwavels), crop=wid, extra_bad=extra_bad) for file in files]

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
    "scale": 0.0432,

    "softening": 2.,#0.1,
    "bias": {},
    "jitter": {},
    "defocus": {}#1e5#{}
}

for exp in exposures_single:
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = np.zeros(npoly).at[0].set(1)*np.log10(np.nansum(exp.data)/nwavels)
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([6., 6.])#*1e2
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. #+ 180.
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 160.*20

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)

def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

base_adam = lambda vals: optax.adam(vals)
opta = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))

base_lbfgs = lambda vals: optax.lbfgs(vals)

optl = lambda lr, start, *schedule: base_lbfgs(scheduler(lr, start, *schedule))




def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 5e-2

things = {
    "positions": opt(g*5, 0),
    "spectrum": opt(g*8, 10),#, (20, 1.5)),
    #"cold_mask_shift": opt(g*100, 30),
    "cold_mask_shift": opt(g*10, 30),
    #"cold_mask_rot": opt(g*10, 100),
    "bias": opt(g*3, 20),
    #"aberrations": opt(g*0.15,300),#, (80, 2)),#, (150, g*0.2)),
    #"aberrations": opta(2, 50),
    "defocus": opt(g*5, 50),
    "aberrations": opta(2, 70),
    #"displacement": opt(g*30, 150),
}


groups = list(things.keys())
paths = flatten(groups)
optimisers = [things[i] for i in groups]
groups = [list(x) if isinstance(x, tuple) else x for x in groups]


losses, models = optimise(params, model_single, exposures_single, things, 130)


print(np.asarray(losses[-20:])/(len(exposures_single)*wid**2))


# %%
print(models[-1].params)

plot_params(models, groups, xw = 3, save="timeseries/{number}-fit.png")

# %%
fsh = calc_fishers(models[-1].inject(model_single), exposures_single, ["defocus"], recalculate=True, save=False)

# %%
fsh2= calc_fishers(models[-1].inject(model_single), exposures_single, ["cold_mask_shift"], recalculate=True, save=False)

# %%
defocuses = [x/20 for x in models[-1].params["defocus"].values()]
errs = [1/x['defocus']/20 for x in fsh.values()]
mjds = [exp.mjd for exp in exposures_single]
#mjds= [(x - mjds[0])*24*60 for x in mjds]



# %%
abb = np.zeros((len(exposures_single), n_zernikes+1))
abb = abb.at[:,1:].set(np.asarray([x for x in models[-1].params["aberrations"].values()]))#.transpose()
abb = abb.at[:,0].set(np.asarray([float(x)/20 for x in models[-1].params["defocus"].values()]))

cold_shift = np.asarray([x for x in models[-1].params["cold_mask_shift"].values()])


numpy.savez(f"timeseries/{number}.npz", defocuses=numpy.asarray(defocuses), errs=numpy.asarray(errs), mjds=numpy.asarray(mjds), aberrations=numpy.asarray(abb), cold_shift = numpy.asarray(cold_shift))