import sys
sys.path.insert(0, '../')

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

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 72
plt.rcParams["font.size"] = 24

ddir = '../data/MAST_2024-07-11T09_26_05.575Z/'
fname_095 = ddir + 'HST/N43CA5020/n43ca5020_mos.fits'
fname_108 = ddir + 'HST/N43C03020/n43c03020_mos.fits'
fname_187 = ddir + 'HST/N43C03010/n43c03010_mos.fits'
fname_190 = ddir + 'HST/N43CA5010/n43ca5010_mos.fits'


wid = 64

exposure_095 = exposure_from_file(fname_095, SinglePointFit(), wid)
exposure_108 = exposure_from_file(fname_108, SinglePointFit(), wid)
exposure_187 = exposure_from_file(fname_187, SinglePointFit(), wid)
exposure_190 = exposure_from_file(fname_190, SinglePointFit(), wid)

exposures_s = [exposure_095]#, exposure_190]#,exposure_108, exposure_187]

exposure_095 = exposure_from_file(fname_095, BinaryFit(), wid)
exposure_108 = exposure_from_file(fname_108, BinaryFit(), wid)
exposure_187 = exposure_from_file(fname_187, BinaryFit(), wid)
exposure_190 = exposure_from_file(fname_190, BinaryFit(), wid)

exposures_b = [exposure_095]#, exposure_190]#, exposure_108, exposure_187]

"""

ADD GRID SEARCH HERE-ISH

"""

oversample = 8

optics = NICMOSOptics(512, wid, oversample)

detector = NICMOSDetector(oversample, wid)

params_s = {
    "fluxes": {},
    "positions": {},
    "aberrations": {},#np.zeros(8),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,
    "cold_mask_shift": {}, #np.asarray([-0.05, -0.05]),
    "cold_mask_rot": {},#np.asarray([np.pi/4]),
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
}

params_b = {
    "fluxes": {},
    "positions": {},
    "contrast": {},
    "separation": dlu.arcsec2rad(0.042),
    "position_angle": 1.8607855,
    "aberrations": {},#np.zeros(8),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,
    "cold_mask_shift": {}, #np.asarray([0.05, 0.05]),
    "cold_mask_rot": {},#np.asarray([np.pi/4]),
    "outer_radius": 1.2*0.955,
    "secondary_radius": 0.372*1.2,
    "spider_width": 0.077*1.2,
}

for exp in exposures_s:
    params_s["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params_s["fluxes"][exp.fit.get_key(exp, "fluxes")] = np.nansum(exp.data)
    params_s["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(8)
    params_s["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-0.05,-0.05])
    params_s["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = np.pi/4

for exp in exposures_b:
    params_b["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])
    params_b["fluxes"][exp.fit.get_key(exp, "fluxes")] = np.nansum(exp.data)/2
    params_b["contrast"][exp.fit.get_key(exp, "contrast")] = 0.3
    params_b["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(8)
    params_b["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([-0.05,-0.05])
    params_b["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = np.pi/4

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

point_model = set_array(NICMOSModel(exposures_s, params_s, optics, detector))
binary_model = set_array(NICMOSModel(exposures_b, params_b, optics, detector))
params_s = ModelParams(params_s)
params_b = ModelParams(params_b)



def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]

g = 3e-2

things_single = {
    "fluxes" : opt(g*20,10),
    "positions": opt(g*1, 0),
    "cold_mask_shift": opt(g*100, 100),
    "cold_mask_rot": opt(g*100, 100),
    "aberrations": opt(g*0.12,50),
    "outer_radius": opt(g*200, 130),
    "secondary_radius": opt(g*100,130),
    "spider_width": opt(g*200,130),
}

g = 2e-2

things_binary = {
    "fluxes" : opt(g*10,10),
    "positions": opt(g*1, 0),
    "separation": opt(g*5, 20),
    "contrast": opt(g*8, 20),
    "position_angle": opt(g*1, 20),
    "cold_mask_shift": opt(g*100,130),
    "cold_mask_rot": opt(g*100,100),
    "aberrations": opt(g*1,50),
    "outer_radius": opt(g*50, 100),
    "secondary_radius": opt(g*50,100),
    "spider_width": opt(g*10,100),
}

groups_s = list(things_single.keys())
paths_s = flatten(groups_s)
optimisers_s = [things_single[i] for i in groups_s]
groups_s = [list(x) if isinstance(x, tuple) else x for x in groups_s]

groups_b = list(things_binary.keys())
paths_b = flatten(groups_b)
optimisers_b = [things_binary[i] for i in groups_b]
groups_b = [list(x) if isinstance(x, tuple) else x for x in groups_b]

@zdx.filter_jit
@zdx.filter_value_and_grad(paths_s)
def loss_fn_s(params,exposures):
    model = params.inject(point_model)
    return np.nansum(np.asarray([posterior(model,exposure) for exposure in exposures]))

@zdx.filter_jit
@zdx.filter_value_and_grad(paths_b)
def loss_fn_b(params,exposures):
    model = params.inject(binary_model)
    return np.nansum(np.asarray([posterior(model,exposure) for exposure in exposures]))

rc = True
fishers_s = calc_fishers(point_model, exposures_s, paths_s, recalculate=rc)
lrs_s = calc_lrs(point_model, exposures_s, fishers_s, paths_s)

fishers_b = calc_fishers(binary_model, exposures_b, paths_b, recalculate=rc)
lrs_b = calc_lrs(binary_model, exposures_b, fishers_b, paths_b)

optim_s, opt_state_s = zdx.get_optimiser(
    params_s, groups_s, optimisers_s
)


losses_s, models_s = [], []
for i in tqdm(range(300)):
    loss, grads = loss_fn_s(params_s,exposures_s)
    grads = jtu.tree_map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs_s.params))
    updates, opt_state_s = optim_s.update(grads, opt_state_s)
    params_s = zdx.apply_updates(params_s, updates)

    models_s.append(params_s)
    losses_s.append(loss)

optim_b, opt_state_b = zdx.get_optimiser(
    params_b, groups_b, optimisers_b
)


losses_b, models_b = [], []
for i in tqdm(range(300)):
    loss, grads = loss_fn_b(params_b,exposures_b)
    grads = jtu.tree_map(lambda x, y: x * np.abs(y), grads, ModelParams(lrs_b.params))
    updates, opt_state_b = optim_b.update(grads, opt_state_b)
    params_b = zdx.apply_updates(params_b, updates)

    models_b.append(params_b)
    losses_b.append(loss)


print(losses_s[0], losses_s[-1])
print(losses_b[0], losses_b[-1])

fig, axs = plt.subplots(1,2, figsize=(18,8))
axs[0].plot(losses_s)
axs[1].plot(losses_s[-60:])
axs[0].plot(losses_b)
axs[1].plot(losses_b[-60:])
fig.tight_layout()

fig.savefig("loss_curves.png")

fig, axs = plt.subplots(3,3,figsize=(30*0.8,22*0.8))

point_model = params_s.inject(point_model)
binary_model = params_b.inject(binary_model)

cmap = matplotlib.colormaps['inferno']
cmap.set_bad('k',1)

ind = 0

coords = dlu.pixel_coords(512, 2.4)
cropped_frame = exposures_s[ind].data**0.125

exposure_s = exposures_s[ind]
exposure_b = exposures_b[ind]

point_frame = exposure_s.fit(point_model, exposure_s)**0.125
binary_frame = exposure_b.fit(binary_model, exposure_b)**0.125

single_resid = (exposure_s.data-exposure_s.fit(point_model, exposure_s))/exposure_s.err
binary_resid = (exposure_b.data-exposure_b.fit(binary_model, exposure_b))/exposure_b.err

vm = max(np.nanmax(cropped_frame),np.nanmax(point_frame), np.nanmax(binary_frame))


axs[0,0].imshow(cropped_frame,cmap=cmap, vmin=0, vmax=vm)
axs[0,1].imshow(point_frame,cmap=cmap, vmin=0, vmax=vm)
rlim = np.nanmax(np.abs(single_resid))
resid = axs[0,2].imshow(single_resid, vmin=-rlim, vmax=rlim, cmap='seismic')
plt.colorbar(resid,ax=axs[0,2])

axs[1,0].imshow(cropped_frame,cmap=cmap, vmin=0, vmax=vm)
axs[1,1].imshow(binary_frame,cmap=cmap, vmin=0, vmax=vm)

rlim = np.nanmax(np.abs(binary_resid))
resid = axs[1,2].imshow(binary_resid, vmin=-rlim, vmax=rlim, cmap='seismic')
plt.colorbar(resid,ax=axs[1,2])

#f095n = np.asarray(pd.read_csv("../data/HST_NICMOS1.F095N.dat", sep=' '))

e_filter = binary_model.filters[exposure_b.filter]

wavels = e_filter[:,0]
weights = e_filter[:,1]

positions = dlu.positions_from_sep(
        binary_model.get(exposure_b.map_param("positions")),
        binary_model.params["separation"],
        binary_model.params["position_angle"]
    )

binary_primary_source = dl.PointSource(
    spectrum=dl.Spectrum(wavels,weights),
    position = positions[1],
    flux = dlu.fluxes_from_contrast(
        binary_model.get(exposure_b.map_param("fluxes")),
        binary_model.get(exposure_b.map_param("contrast")),
    )[1]
)

binary_optics = exp.fit.update_optics(binary_model, exp)


binary_primary_system = dl.Telescope(
    binary_optics,
    #binary_model.optics,
    binary_primary_source,
    binary_model.detector
)

binary_primary_frame = binary_primary_system.model()**0.125

axs[2,0].imshow(cropped_frame,cmap=cmap, vmin=0, vmax=vm)
axs[2,1].imshow(binary_primary_frame,cmap=cmap, vmin=0, vmax=vm)

bp_resid = (exposure_b.data-binary_primary_system.model())/exposure_b.err
rlim = np.nanmax(np.abs(bp_resid))
resid = axs[2,2].imshow(bp_resid, vmin=-rlim, vmax=rlim, cmap='seismic')
plt.colorbar(resid,ax=axs[2,2])

x, y = dlu.rad2arcsec(positions[0])/0.042 + wid/2

print(x,y)

axs[2,2].axvline(x, color='k',linestyle='--')
axs[2,2].axhline(y, color='k',linestyle='--')

axs[0,0].set_title("Exposure")
axs[1,0].set_title("Exposure (again)")
axs[2,0].set_title("Exposure (yet again)")

axs[0,1].set_title("Single star fit")
axs[1,1].set_title("Binary star fit")
axs[2,1].set_title("Binary primary")

axs[0,2].set_title("Single star Residuals")
axs[1,2].set_title("Binary star Residuals")
axs[2,2].set_title("Binary primary Residuals")



for i in range(3):
    for j in range(3):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

fig.tight_layout()

fig.savefig("model_comparison.png")


xw = 3
yw = int(np.ceil(len(groups_s)/xw))

print(len(groups_s))


fig, axs = plt.subplots(xw,yw,figsize=(xw*10,yw*8))
for i, param in enumerate(groups_s):
    #print(param)
    sp = axs[i%xw, i//xw]
    if param in ["fluxes", "contrast", "positions", "aberrations", "cold_mask_shift", "cold_mask_rot"]:
        #print(np.asarray(list(models_s[0].get(param).values())))
        for p in np.asarray([np.asarray(list(x.get(param).values())) for x in models_s]).T:
            if len(p.shape)>1:
                for i in range(p.shape[0]):
                    sp.plot(p[i,:])
            else:
                sp.plot(p)
            sp.set_title(param)
    else:
        sp.set_title(param)
        sp.plot([x.get(param) for x in models_s])
    
fig.tight_layout()
fig.save("single.png")

xw = 4
yw = int(np.ceil(len(groups_b)/xw))

print(len(groups_b))

fig, axs = plt.subplots(xw,yw,figsize=(xw*10,yw*8))
for i, param in enumerate(groups_b):
    #print(param)
    sp = axs[i%xw, i//xw]
    if param in ["fluxes", "contrast", "positions", "aberrations", "cold_mask_shift", "cold_mask_rot"]:
        #print(np.asarray(list(models_b[0].get(param).values())))
        for p in np.asarray([np.asarray(list(x.get(param).values())) for x in models_b]).T:
            if len(p.shape)>1:
                for i in range(p.shape[0]):
                    sp.plot(p[i,:])
            else:
                sp.plot(p)
            sp.set_title(param)
    else:
        sp.set_title(param)
        sp.plot([x.get(param) for x in models_b])
    
fig.tight_layout()

plt.save("binary.png")