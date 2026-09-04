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
import optimistix as optx

# dLux imports
import dLux as dl
import dLux.utils as dlu

# Visualisation imports
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import chainconsumer as cc

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

from detectors import *
from apertures import *
from models import *
from stats import posterior
from fitting import *
from plotting import *
from spectra import *

import jax.tree_util as jtu
import interpax as ipx
from glob import glob

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

ddir = '../data/NICMOS-LAPL-DD2/LAPL_DATA_DD2/comtemp_flats-DD2/'
fnames = glob(ddir+"*_o_clc_calf.fits")
fnames.sort()

targinfo = []
for fname in fnames[:]:
    hdr = fits.getheader(fname)
    try:
        targinfo.append((fname, hdr["TARGNAME"], hdr["FILTER"], hdr["TARGCNTR"]))
    except:
        pass
        # print(hdr["TARGNAME"])

objs = [
    "HD-139664",
    "HD109085",
    "GJ784-PSF",
    "PSF-5-K0",
    "GSC8056-0482",
    "GSC8491-1194",
    "GSC8894-0426",
    "HIP10679",
    "HIP107947",
    "HIP109901",
    "HIP112312A",
    "HIP112312B",
    "HIP114530",
    "HIP118008",
    "HIP12413",
    "HIP12545",
    "HIP12787B",
    "HIP17695",
    "HIP18859",
    "HIP19335",
    "HIP1993",
    "HIP21547",
    "HIP21632",
    "HIP23309",
    "HIP26990",
    "HIP2729",
    "HIP36627",
    "HIP37766",
    "HIP39896B",
    "HIP41889",
    "HIP50156",
    "HIP56445",
    "HIP6485",
    "HIP77199",
    "HIP9892",
    "HIP9902",
    "SSS1102-34",
    "TWA12",
    "TWA23",
    "TYC5672-0216",
    "G238-44",
]

index = int(sys.argv[1])
target = objs[index]

files = [x[0] for x in targinfo if x[1]==target and x[2]=="F160W" and x[3]=="CENTERED"]

# %%
wid = 80
oversample = 4

nwavels = 20
npoly=4

n_modes = 20
n_zernikes = 50

resolved_wid = 1

optics = NICMOSCoronagraph(512, wid, oversample, n_modes=n_modes, n_zernikes=n_zernikes)

detector = NICMOSDetector(oversample, wid)

spectrum_basis = np.ones((nwavels, npoly))


vects = np.load("../data/iterative_spectrum_basis_F160W.npy")[:,:npoly]
assert vects.shape == (nwavels, npoly)
spectrum_basis = vects/np.sqrt(np.mean(vects**2, axis=0))

flatdir = '../data/NICMOS-LAPL-DD2/LAPL_HOLEFLATS_DD2/'



exposures_single = [
    exposure_from_file(f, SinglePointFit(spectrum_basis, "F160W"), crop=wid, flatcorr=flatdir) for f in files[0:1]
]



params = {
    "spectrum": {},
    "primary_opd": {},
    "primary_low": {},
    "primary_tilt": {},
    "cold_mask_opd": {},
    "cold_mask_tilt": {},
    "cold_mask_shift": {},
    "cold_mask_rot": {},
    "cold_mask_shear": {},
    "cold_mask_scale": {},
    "primary_rot": {},

    "bias": {},
    "occulter_radius": 0.7,
    "occulter_coeffs": np.zeros(2)+1,
    "fnumber": 45.7,
}


for idx, exp in enumerate(exposures_single):
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set((np.nansum(exp.data)/nwavels)*4)

    params["primary_tilt"][exp.fit.get_key(exp, "primary_tilt")] = np.array([-0.05, -0.75])*0.075
    params["cold_mask_tilt"][exp.fit.get_key(exp, "cold_mask_tilt")] = np.array([
        np.array(exp.hdr["TARSIAFX"],dtype=float) - (256-181), 
        np.array(exp.hdr["TARSIAFY"],dtype=float) - (256-44)
    ])*0.075


    params["primary_opd"][exp.fit.get_key(exp, "primary_opd")] = np.zeros((n_modes, n_modes))
    params["primary_low"][exp.fit.get_key(exp, "primary_low")] = np.zeros((n_zernikes))
    params["cold_mask_opd"][exp.fit.get_key(exp, "cold_mask_opd")] = np.array([120.])

    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.array([13.,10.]) #np.asarray([-13.,-7.])#
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = 0.#-90.
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -0.6##-90.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    

model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))

params = ModelParams(params)

# %%
plot_comparison(model_single, params, exposures_single, percentile=99, wf_size=512)

# %%
def sgd(lr, delay, momentum=0.5):
    return optax.sgd(zdx.optimisation.delay(lr, delay), momentum=momentum)

def adam(lr, delay):
    return optax.adam(zdx.optimisation.delay(lr, delay))


g = 5e-2

things = {
    "primary_opd": sgd(g*0.1, 0),

    "spectrum": sgd(g*3, 0),
    "primary_tilt": sgd(g*3, 0),
    "cold_mask_tilt": sgd(g*3, 0),
    "cold_mask_opd": sgd(g*1, 0),

    "bias": sgd(g*3, 0),
    "cold_mask_shift": sgd(g*3, 0),
    "cold_mask_rot": sgd(g*0.2, 0),
    "primary_rot": sgd(g*1, 0),

    "primary_low": sgd(g*1, 0),

    "cold_mask_scale": sgd(g*1, 0),
    "cold_mask_shear": sgd(g*1, 0),

    "occulter_radius": sgd(g*1., 0),
    # "occulter_coeffs": sgd(g*1, 0),


}

things_start = {
    "spectrum": sgd(g*3, 30.),
    "primary_tilt": sgd(g*10, 15.),
    "cold_mask_tilt": sgd(g*5, 0),
    "cold_mask_opd": sgd(g*3, 40),

    "bias": sgd(g*3, 50),
    "cold_mask_shift": sgd(g*3, 70),
    "cold_mask_rot": sgd(g*0.2, 85),
    "primary_rot": sgd(g*1, 85),

    "primary_low": sgd(g*0.5, 150),

    "cold_mask_scale": sgd(g*5, 100),
    "cold_mask_shear": sgd(g*10, 100),

    "occulter_radius": sgd(g*1., 120),
    # "occulter_coeffs": sgd(g*1, 120),
    # # "fnumber": sgd(g*2., 150),
}

groups = list(things.keys())

# %%
# orig_params = params.params
# opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
# losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 10)

# %%
# plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single)

# %%
orig_params = params.params
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things_start})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things_start, 300, nbatches=30)


# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"calibrators/intermediate-losses-{index}.png")

# %%
losses[-1]

# %%
plot_params(params_history, list(things_start.keys()), xw = 3, save=f"calibrators/intermediate-params-{index}")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False, wf_size=512, save=f"calibrators/intermediate-comparison-{index}")

orig_params = params.params | params_history[-1]
opt_params = set_array({k:orig_params[k] for k in orig_params if k in things})

# %%
losses, params_history = optimise_new(opt_params, model_single, exposures_single, things, 500, nbatches=50)

# %%
plt.figure(figsize=(10,10))
plt.plot(losses[:])
plt.savefig(f"calibrators/losses-{index}.png")


# %%
plot_params(params_history, list(things_start.keys()), xw = 3, save=f"calibrators/params-{index}")
plot_comparison(model_single, ModelParams(params_history[-1]), exposures_single, percentile=100, quadrature=False, wf_size=512, save=f"calibrators/comparison-{index}")

# %%
print(params_history[-1])



np.save(f"calibrators/params-{index}-{target}", params_history[-1])