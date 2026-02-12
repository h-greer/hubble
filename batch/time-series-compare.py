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
#jax.config.update("jax_log_compiles", True)
#jax.config.update("jax_explain_cache_misses", True)


from sklearn.decomposition import PCA
from astropy.io import fits
from sklearn.cluster import KMeans


# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib

import glob

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

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

wid = 64#64
oversample = 4

nwavels = 3
npoly=1

n_zernikes = 26

optics = NICMOSSecondaryFresnelOptics(512, wid, oversample, mag=3.3, defocus=0., despace=0., n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)

# %%

spectrum_basis = np.ones((1, 3))


number = 18

ddir = "../data/MAST_2025-06-24T0210/HST/"

cfiles = glob.glob(ddir+"*_cal.fits")
cfiles.sort()

mjds = numpy.asarray([float(fits.getheader(fname, ext=0)["EXPSTART"]) for fname in cfiles])

kmeans = KMeans(n_clusters=20, random_state=0).fit(np.reshape(mjds, (-1, 1)))
idx = numpy.argsort(kmeans.cluster_centers_.sum(axis=1))
lut = numpy.zeros_like(idx)
lut[idx] = numpy.arange(20)

clumps = lut[kmeans.labels_]

files = [cfiles[i] for i in range(len(cfiles)) if clumps[i] == number]

exposures_raw = [exposure_from_file(file, SinglePointFit(spectrum_basis, "F187N"), crop=wid) for file in files]


exposures_single = []
for exp in exposures_raw:
    if exp.data.shape == (wid, wid):
        exposures_single.append(exp)

#exposures_single=exposures_single[1:50]

# %%
len(exposures_single)

# %%
for e in exposures_single:
    print(e.mjd*86400)
    print(e.exptime)
    print(e.target)
    print(e.filter)
    print((1.8-e.pam))#*0.0470)

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
    "scale": 0.0432,

    "softening": 2.,#0.1,
    "bias": {},
    "jitter": {},
    "defocus": {},#1e5#{}
    "despace": {},
    "mag": 3.3,
}

for exp in exposures_single:
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = np.log10(np.nansum(exp.data)/3)-3#np.zeros(npoly).at[0].set(1)*np.log10(np.nansum(exp.data)/nwavels/0.3)
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([8., 8.])#*1e2
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. #+ 180.
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = -0.233#2.4#800.#160.*20
    params["despace"][exp.fit.get_key(exp, "despace")] = 0.#2.4#800.#160.*20

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
plt.imshow(exposures_single[0].data)
plt.colorbar()

# %%
np.nanmax(exposures_single[0].data)

# %%
#plot_comparison(model_single, params, exposures_single)

# %%
exposures_single[0].pam

# %%
exposures_single[0].bad[35,60]

# %%
plt.imshow(exposures_single[0].data)
plt.colorbar()

# %%
cmap = matplotlib.colormaps['inferno']
cmap.set_bad('k',1)
plt.figure(figsize=(10,10))
plt.imshow(exposures_single[0].data**0.125, cmap=cmap)
plt.title(exposures_single[0].target)
plt.colorbar()


# %%
def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    return np.nansum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))

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

base_lbfgs = lambda vals: optax.lbfgs(vals)

optl = lambda lr, start, *schedule: base_lbfgs(scheduler(lr, start, *schedule))




def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 1e-2

things = {
    "positions": opt(g*15, 0),
    "spectrum": opt(g*20, 10),#, (20, 1.5)),
    #"cold_mask_shift": opt(g*100, 45),
    "cold_mask_shift": opt(g*80, 50),
    #"cold_mask_rot": opt(g*10, 100),
    "bias": opt(g*8, 20),
    #"defocus": opt(g*20, 40),
    "despace": opt(g*10, 30),
    "mag": opt(g*10, 80),
    "aberrations": opt(g*1, 60),
}


groups = list(things.keys())
paths = flatten(groups)
optimisers = [things[i] for i in groups]
groups = [list(x) if isinstance(x, tuple) else x for x in groups]

# %%
losses, models = optimise(params, model_single, exposures_single, things, 300, recalculate=True)

# %%
losses[-1]

# %%
plt.plot(np.asarray(losses[-20:])/(len(exposures_single)*wid**2))

# %%
#plot_params(models, groups, xw = 3)
#plot_comparison(model_single, models[-1], exposures_single)

# %%
models[-1].params





# %%
fsh = calc_fishers(models[-1].inject(model_single), exposures_single, ["despace"], fisher_fn, recalculate=True, save=False)

# %%
1/np.sqrt(list(fsh.values())[0].flatten()[0])

# %%
defocuses = [x for x in models[-1].params["despace"].values()]
errs = [np.sqrt(1/float(x.flatten()[0])) for x in fsh.values()]
mjds = [exp.mjd for exp in exposures_single]
print(mjds)
mins= [(x - mjds[0])*24*60 for x in mjds]


# %%
np.asarray([x for x in models[-1].params["cold_mask_shift"].values()])

# %%
np.asarray([x for x in models[-1].params["spectrum"].values()])

# %%
plt.errorbar(mins, defocuses, np.squeeze(np.asarray(errs)))
plt.xlabel("Time (minutes)")
plt.ylabel("Despace (µm)")

# %%
import pandas as pd
model = pd.read_csv("focus2.txt", sep=", ")[1:]


# %%
import interpax as ipx

# %%
mjds_nacho = np.array(mjds)
defocus_nacho = -np.array(defocuses)
err_nacho = np.array(errs)

mjds_st = model["Julian Date"].to_numpy()
msk = (mjds_st > np.min(mjds_nacho)) & (mjds_st < np.max(mjds_nacho))
defocus_st = model["Model"].to_numpy()

mjds_interpd = np.linspace(mjds_st.min(), mjds_st.max(), 10000)
defocus_interpd = ipx.interp1d(mjds_interpd, mjds_st, defocus_st, "linear")
msk_interpd = (mjds_interpd > np.min(mjds_nacho)) & (mjds_interpd < np.max(mjds_nacho))

numpy.savez(
    "time-series-compare.npz",
    mjds_nacho=mjds_nacho,
    defocus_nacho=defocus_nacho,
    err_nacho=err_nacho,
    mjds_st=mjds_st,
    msk=msk,
    defocus_st=defocus_st,
    mjds_interpd=mjds_interpd,
    defocus_interpd=defocus_interpd,
    msk_interpd=msk_interpd,
)

rang = np.max(defocus_nacho)-np.min(defocus_nacho)

top = np.max(defocus_interpd[msk_interpd])

fig, ax1 = plt.subplots(figsize=(10,10))
ax1.errorbar(mjds_nacho, defocus_nacho, yerr=err_nacho, fmt="o", label="Phase Retrieval")
ax2 = ax1.twinx()
ax2.scatter(mjds_st[msk], defocus_st[msk], color="r")
ax2.plot(mjds_interpd[msk_interpd], defocus_interpd[msk_interpd], color="r", label="STScI Model")
ax2.set_ylim(top-rang, top)
ax1.set_xlabel("MJD")
ax1.set_ylabel("Phase Retrieval Despace (μm)")
ax2.set_ylabel("STScI Focus Model (μm)")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.tight_layout()
plt.savefig("time-series-compare.png")