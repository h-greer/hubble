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
from spectra import *

import jax.tree_util as jtu
import interpax as ipx

def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)

# %%
extra_bad = None
#extra_bad = np.isnan(np.zeros((64, 64)).at[35,60].set(np.nan))

#extra_bad = np.isnan(np.zeros((wid,wid))).at[wid//2-3:wid//2+3,:].set(np.nan)


# %%
wid = 90
oversample = 4

nwavels = 20#13#6
npoly=15#2

n_zernikes = 30#30#12

optics = NICMOSSecondaryFresnelOptics(512, wid, oversample, mag=3.3, defocus=0., despace=0., n_zernikes = n_zernikes)

detector = NICMOSDetector(oversample, wid)

ddir = "../data/MAST_2024-09-22T03_37_01.724Z/HST/"

basis_file = np.load("spectrum_basis.npy")[:,:npoly]

spectrum_basis = ipx.interp1d(np.linspace(0,1,nwavels), np.linspace(0,1,basis_file.shape[0]), basis_file)
spectrum_basis = spectrum_basis/np.sqrt(np.mean(spectrum_basis**2, axis=0))




ddir = "../data/MAST_2024-09-26T22_53_13.719Z/HST/"

exposures_single = [
    exposure_from_file(ddir + "n8ry01tkq_cal.fits", SinglePointFit(spectrum_basis, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry01tmq_cal.fits", SinglePointFit(spectrum_basis, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry02tpq_cal.fits", SinglePointFit(spectrum_basis, "F110W"), crop=wid),
    exposure_from_file(ddir + "n8ry02tqq_cal.fits", SinglePointFit(spectrum_basis, "F110W"), crop=wid),

    exposure_from_file(ddir + "n8ry03vbq_cal.fits", SinglePointFit(spectrum_basis, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry03vcq_cal.fits", SinglePointFit(spectrum_basis, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry04vfq_cal.fits", SinglePointFit(spectrum_basis, "F160W"), crop=wid),
    exposure_from_file(ddir + "n8ry04vgq_cal.fits", SinglePointFit(spectrum_basis, "F160W"), crop=wid),
]


# %%
for e in exposures_single:
    print(e.mjd)#*86400)
    print(e.target)
    print(e.filter)
    print(e.exptime)

# %%
plt.plot(spectrum_basis[:, :5])

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

    "softening": 2.,#0.1,
    "bias": {},
    "jitter": {},

    "defocus": {},#1e5#{}
    "despace": {},
    "mag": 3.3,
}

positions = [[0.,0.,],[0.,0.,],[0.,0.,],[0.,0.,]]#[[0.43251792, 0.33013815],[ 0.49417186, -0.5629123 ]]


for idx, exp in enumerate(exposures_single):
    params["positions"][exp.fit.get_key(exp, "positions")] = np.asarray([0.,0.])#positions_dict[exp.fit.get_key(exp, "positions")]#np.asarray(positions[idx])#np.asarray([0.49162114, -0.5632928])#np.asarray([ 0.45184505, -0.8391668 ])#np.asarray([-0.2,0.4])
    params["spectrum"][exp.fit.get_key(exp, "spectrum")] = (np.zeros(npoly)).at[0].set(np.log10(np.nansum(exp.data)/nwavels))#np.asarray([-1.03646245, -0.29984712, -0.14137265, -0.04618831, -0.05788671, -0.02545625,
 #-0.03688181,  0.0231736,   0.02356589, -0.00177967]).at[0].set(np.log10(np.nansum(exp.data)/nw[idx]))#np.zeros(nspec[idx]).at[0].set(1)*np.log10(np.nansum(exp.data)/nw[idx])#np.ones(npoly)*np.log10(np.nansum(exp.data)/nwavels)#(np.zeros(npoly)).at[0].set(1)*np.log10(np.nansum(exp.data)/nwavels)
    params["aberrations"][exp.fit.get_key(exp, "aberrations")] = np.zeros(n_zernikes)#np.asarray([0., 24.884588  , -25.489779  , -17.15699   , -21.790146  ,
    #      -4.592212  ,  -4.832893  ,  19.196083  ,   0.37983412,
    #       7.0756216 ,   0.30277824,  -6.330534])#np.zeros(n_zernikes)
    params["cold_mask_shift"][exp.fit.get_key(exp, "cold_mask_shift")] = np.asarray([6.,6.])#np.asarray([9.599048, 6.196583])
    params["cold_mask_rot"][exp.fit.get_key(exp, "cold_mask_rot")] = -45.
    params["cold_mask_scale"][exp.fit.get_key(exp, "cold_mask_scale")] = np.asarray([1.,1.])
    params["cold_mask_shear"][exp.fit.get_key(exp, "cold_mask_shear")] = np.asarray([0.,0.])
    params["primary_rot"][exp.fit.get_key(exp, "primary_rot")] = -45. + 90. 
    params["primary_scale"][exp.fit.get_key(exp, "primary_scale")] = np.asarray([1.,1.])
    params["primary_shear"][exp.fit.get_key(exp, "primary_shear")] = np.asarray([0.,0.])
    params["defocus"][exp.fit.get_key(exp, "defocus")] = 0.#-0.233#2.4#800.#160.*20
    params["despace"][exp.fit.get_key(exp, "despace")] = 0.#2.4#800.#160.*20
    

    params["bias"][exp.fit.get_key(exp, "bias")] = 0.
    params["jitter"][exp.fit.get_key(exp, "jitter")] = 7/43*oversample


model_single = set_array(NICMOSModel(exposures_single, params, optics, detector))
#model_binary = set_array(NICMOSModel(exposures_binary, params, optics, detector))


params = ModelParams(params)

# %%
print(np.nansum(exp.data))

# %%
np.log10(7336/np.sum(10**spectrum_basis[:,0]))

# %%
print(params.params)

# %%
plot_comparison(model_single, params, exposures_single)

# %%
plt.imshow(exposures_single[0].err)
plt.colorbar()

# %%
plt.imshow(np.log10(exposures_single[0].data/exposures_single[0].err))
plt.colorbar()

# %%
print(exposures_single[0].exptime)

# %%
plt.imshow(exposures_single[0].data**0.125)
plt.xticks([])
plt.yticks([])

# %%
cmap = matplotlib.colormaps['inferno']
cmap.set_bad('k',1)
plt.figure(figsize=(10,10))
plt.imshow(exposures_single[0].data**0.125, cmap=cmap)
plt.title(exposures_single[0].target)
plt.colorbar()


# %%
def scheduler(lr, start, *args):
    shed_dict = {start: 1e10}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e10, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)

opt = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))

base_sgd2 = lambda vals: optax.noisy_sgd(vals)

opts = lambda lr, start, *schedule: base_sgd2(scheduler(lr, start, *schedule))


base_adam = lambda vals: optax.adam(vals)
opta = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))





def flatten(l):
    if isinstance(l, (tuple, list)):
         return [a for i in l for a in flatten(i)]
    else:
        return [l]



g = 5e-2

things = {
    "positions": opt(g*5, 0),
    "spectrum": opt(g*1, 10),#opt(g*2, 10),#opt(g*2, 10),#, (20, 1.5)),
    "cold_mask_shift": opt(g*1, 30),
    
    "bias": opt(g*3, 20),
    "aberrations": opt(g*0.05, 80),
    #"jitter": opt(g*1, 120),

    "despace": opt(g*0.8, 50),
    "mag": opt(g*10, 100),

    #"cold_mask_scale": opt(g*0.1, 100),
    #"cold_mask_shear": opt(g*0.1, 100),
    #"primary_scale": opt(g*1, 100),
    #"primary_shear": opt(g*1, 100),
}

things_start = {
    "positions": opt(g*5, 0),
}

groups = list(things.keys())

# %%
initial_losses, initial_models = optimise(params, model_single, exposures_single, things_start, 10, recalculate=True)

# %%
plot_comparison(model_single, initial_models[-1], exposures_single)

# %%
initial_models[-1].params

# %%
losses, models = optimise(initial_models[-1].inject(params), initial_models[-1].inject(model_single), exposures_single, things, 500, recalculate=True)
#losses, models = optimise(params, model_single, exposures_single, things, 150, recalculate=False)

# %%
plt.plot(np.asarray(losses[-50:])/(len(exposures_single)*wid**2))

# %%
print(losses[0], losses[-1])

# %%
models_pd = [jax.tree.map(lambda x,y: (x-y)/y, models[i], models[-1]) for i in range(len(models))]

# %%
plot_params(models, groups, xw = 3)
plot_comparison(model_single, models[-1], exposures_single)

# %%
print(models[-1].params)

# %%
models[-1].inject(model_single)

# %%
models[-1]

# %%
groups

# %%
#stop

# %%
#fsh = calc_fishers(models[-1].inject(model_single), exposures_single, groups, fisher_fn, recalculate=True, save=False)


# %%
def loss_fn(params, exposures, model):
    mdl = params.inject(model)
    res = np.sum(np.asarray([posterior(mdl,exposure) for exposure in exposures]))
    return np.where(res==0.0, np.inf, res)

@eqx.filter_jit
def fun(params, args):
    exposures, model = args
    return loss_fn(params, exposures, model)

def optimise_optimistix(params, model, exposures, things, niter):
    paths = list(things.keys())
    optimisers = [things[i] for i in paths]

    model_params = ModelParams({p: params.get(p) for p in things.keys()})

    solver = optx.BFGS(rtol=1e-6, atol=1e-6,verbose=frozenset({"step_size", "loss"}))
    sol = optx.minimise(fun, solver, model_params, (exposures, model), throw=False, max_steps=niter)
    
    return sol



# %%
sol = optimise_optimistix(models[-1], models[-1].inject(model_single), exposures_single, things, 500)
print(sol.value.params)
print(fun(sol.value, (exposures_single, model_single)), (losses[-1]))

# %%
final_params = sol.value
# final_params = models[-1]

# %%
final_params.params

# %%
#sol.stats

# %%
plot_comparison(final_params.inject((model_single)), final_params, exposures_single)

# %%
# calculate spectrum


plt.figure(figsize=(10,10))

wv, filt = calc_throughput("F110W", nwavels=nwavels)

spec = CombinedBasisSpectrum(wv, filt, final_params.get("spectrum.HZ4_F110W"), spectrum_basis)

wv2, filt2 = calc_throughput("F160W", nwavels=nwavels)

spec2 = CombinedBasisSpectrum(wv2, filt2, final_params.get("spectrum.HZ4_F160W"), spectrum_basis)


sp = spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6)
sp2 = spec2.spec_weights()/spec2.flux*spec2.proper_flux()/(wv2*1e6)

#plt.plot(wavels, params.get("spectrum.U10764_F110W"))
plt.plot(wv*1e6, sp)
plt.plot(wv2*1e6, sp2)

plt.xlabel("Wavelength (um)")
plt.savefig("wd-spec-basic.png")

# %%
fsh = calc_fishers(final_params.inject(model_single), exposures_single, ["spectrum"], fisher_fn, recalculate=True, save=False)
fsh

# %%
def populate_fisher_model(fishers, exposures, model_params):

    # Build the lr model structure
    params_dict = jax.tree.map(lambda x: np.zeros((x.size, x.size)), model_params).params

    # Loop over exposures
    for exp in exposures:

        # Loop over parameters
        for param in model_params.keys():

            # Check if the fishers have values for this exposure
            key = f"{exp.key}.{param}"
            if key not in fishers.keys():
                continue

            # Add the Fisher matrices
            if isinstance(params_dict[param], dict):
                params_dict[param][exp.get_key(param)] += fishers[key]
            else:
                params_dict[param] += fishers[key]

    fisher_params = model_params.set("params", params_dict)

    return fisher_params

# %%
fsh

# %%
fm = populate_fisher_model(fsh, exposures_single, models[-1])

# %%
fm.get("spectrum.HZ4_F110W")

# %%
cov_f110w = numpy.linalg.inv(fm.get("spectrum.HZ4_F110W"))#+fsh['n8yj02wyq.spectrum'])
spectrum_err = np.diag(np.sqrt(np.abs(cov_f110w)))

cov_f160w = numpy.linalg.inv(fm.get("spectrum.HZ4_F110W"))#+fsh['n8yj02wyq.spectrum'])


# %%


# %%
plt.imshow(np.linalg.inv(fm.get("spectrum.HZ4_F110W")), cmap='seismic', vmin=-np.max(np.abs(cov_f110w)), vmax=np.max(np.abs(cov_f110w)))
plt.colorbar()


# %%
vals, vects = np.linalg.eig(fm.get("spectrum.HZ4_F110W"))#+fsh['n8yj02wyq.spectrum'])

order = np.argsort(vals)[::-1]


plt.semilogy(np.sort(np.real(vals))[::-1])

# %%
#spec.filt_weights.sum()
#spec2.filt_weights.sum()

# %%
#(filt/(wv*1e6)).sum()/(filt2/(wv*1e6)).sum()

# %%


# %%
from scipy.optimize import curve_fit
from scipy.constants import h, c, k

def planck_wavelength(wav, T, scale, bg):
    """
    Planck's Law as a function of wavelength (m) and temperature (K).
    Returns intensity in arbitrary units with a scaling factor.
    """
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    intensity = scale * a / ( (wav**5) * (np.exp(b) - 1.0) ) + bg# * (wav/1e-6)
    return intensity

def rayleigh_jeans(wav, scale, bg):
    return scale/wav**4 + bg

# %%
wavels = np.concat((wv, wv2))
spectrum = np.concat((sp, sp2))

initial_guesses = [14500.0, 1e-12, 0.] 

# Perform the curve fit
popt, pcov = curve_fit(planck_wavelength, wavels, spectrum, p0=initial_guesses, maxfev=10000)
print(popt)


# %%
from jax.numpy import linalg as la
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# %%
popt

# %%
np.sqrt(pcov[0,0])

# %%
np.round(153.2, -2)

# %%
plt.figure(figsize=(10,7))

plt.plot(np.sort(wavels)*1e6, planck_wavelength(np.sort(wavels), *popt), label=fr"Blackbody Curve")# $T = {np.round(popt[0], -3):3.0f}$")
plt.plot(wv*1e6, sp, label = "F110W Spectrum")
plt.plot(wv2*1e6, sp2, label = "F160W Spectrum")

plt.axvline(1.28)
plt.axvline(1.09)
plt.axvline(0.955)
plt.axvline(1.09)


plt.xlabel("Wavelength (um)")
plt.legend()

# %%
plt.figure(figsize=(10,7))

plt.plot(wv*1e6, sp, color='orange')
plt.plot(wv2*1e6, sp2, color='orange')

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.HZ4_F110W"), nearestPD(cov_f110w))
    spec = CombinedBasisSpectrum(wv, filt, coeffs, spectrum_basis)#CombinedFourierSpectrum(wv, filt, coeffs)
    plt.plot(wv*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv*1e6), color='b', alpha=0.01, zorder=0)

for i in range(1000):
    coeffs = numpy.random.multivariate_normal(final_params.get("spectrum.HZ4_F160W"), nearestPD(cov_f160w))
    spec = CombinedBasisSpectrum(wv2, filt2, coeffs, spectrum_basis)#CombinedFourierSpectrum(wv, filt, coeffs)
    plt.plot(wv2*1e6, spec.spec_weights()/spec.flux*spec.proper_flux()/(wv2*1e6), color='b', alpha=0.01, zorder=0)

plt.plot(wavels*1e6, planck_wavelength(wavels, *popt), label=fr"Blackbody Curve")# $T = {np.round(popt[0], -3):3.0f}$")


plt.xlabel("Wavelength (um)")
#plt.legend()

plt.save("wd-spectrum.png")

