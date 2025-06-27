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

from astropy.io import fits
from sklearn.cluster import KMeans

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

cfiles = glob.glob(ddir+"*_cal.fits")
cfiles.sort()

mjds = [float(fits.getheader(fname, ext=0)["EXPSTART"]) for fname in cfiles]

kmeans = KMeans(n_clusters=20).fit(np.reshape(mjds, (-1, 1)))
idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
lut = np.zeros_like(idx)
lut[idx] = np.arange(20)

clumps = lut[kmeans.labels_]

files = [cfiles[i] for i in range(len(cfiles)) if clumps[i] == number]


print(files)

"""

FILE THINGS


"""

exposures_raw = [exposure_from_file(ddir + file, SinglePointPolySpectrumFit(nwavels), crop=wid, extra_bad=extra_bad) for file in files]

exposures_single = []
for exp in exposures_raw:
    if exp.data.shape == (wid, wid):
        exposures_single.append(exp)

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
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8., 8.])#*1e2
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



g = 1e-2

things = {
    "positions": opt(g*2, 0),
    "spectrum": opt(g*8, 10),#, (20, 1.5)),
    #"cold_mask_shift": opt(g*100, 30),
    "cold_mask_shift": opt(g*20, 30),
    #"cold_mask_rot": opt(g*10, 100),
    "bias": opt(g*3, 20),
    #"aberrations": opt(g*0.15,300),#, (80, 2)),#, (150, g*0.2)),
    #"aberrations": opta(2, 50),
    "defocus": opt(g*2, 50),
    "aberrations": opta(2, 70),
    #"displacement": opt(g*30, 150),
}


groups = list(things.keys())
paths = flatten(groups)
optimisers = [things[i] for i in groups]
groups = [list(x) if isinstance(x, tuple) else x for x in groups]


losses, models = optimise(params, model_single, exposures_single, things, 200)


print(np.asarray(losses[-20:])/(len(exposures_single)*wid**2))


# %%
print(models[-1].params)

plot_params(models, groups, xw = 3, save=f"timeseries-orbit/{number}-fit.png")

# %%
fsh = calc_fishers(models[-1].inject(model_single), exposures_single, ["defocus"], recalculate=True, save=False)

# %%
defocuses = [x/20 for x in models[-1].params["defocus"].values()]
errs = [1/x['defocus']/20 for x in fsh.values()]
mjds = [exp.mjd for exp in exposures_single]
spectra = np.asarray([x for x in models[-1].params["spectrum"].values()])

#mjds= [(x - mjds[0])*24*60 for x in mjds]



# %%
abb = np.zeros((len(exposures_single), n_zernikes+1))
abb = abb.at[:,1:].set(np.asarray([x for x in models[-1].params["aberrations"].values()]))#.transpose()
abb = abb.at[:,0].set(np.asarray([float(x)/20 for x in models[-1].params["defocus"].values()]))

cold_shift = np.asarray([x for x in models[-1].params["cold_mask_shift"].values()])


numpy.savez(f"timeseries-orbit/{number}.npz", defocuses=numpy.asarray(defocuses), errs=numpy.asarray(errs), mjds=numpy.asarray(mjds), aberrations=numpy.asarray(abb), cold_shift = numpy.asarray(cold_shift), spectra = numpy.asarray(spectra))