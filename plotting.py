import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from apertures import *
from detectors import *
from spectra import *
from models import *
from stats import *
from fisher import *

from matplotlib import pyplot as plt
import matplotlib


def plot_params(models, groups, xw = 4, save=False):
    yw = int(np.ceil(len(groups)/xw))

    print(len(groups))


    fig, axs = plt.subplots(xw,yw,figsize=(xw*10,yw*8))
    for i, param in enumerate(groups):
        sp = axs[i%xw, i//xw]
        if param in ["fluxes", "contrast", "positions", "aberrations", 
                    "cold_mask_shift", "cold_mask_rot", "cold_mask_scale", "cold_mask_shear",
                    "primary_rot","primary_scale", "primary_shear", "breathing", "slope", "spectrum", "primary_spectrum", "secondary_spectrum", "bias", "primary_distortion", "cold_mask_distortion"]:

            for p in np.asarray([np.asarray(list(x.get(param).values())).flatten() for x in models]).T:
                if len(p.shape)>1:
                    for i in range(p.shape[0]):
                        sp.plot(p[i,:])
                else:
                    sp.plot(p)
                sp.set_title(param)
        else:
            sp.set_title(param)
            sp.plot([x.get(param) for x in models])
        
    fig.tight_layout()
    if save:
        fig.savefig(f"{save}.png")


def plot_comparison(model, params, exposures, save=False):
    for f, exp in enumerate(exposures):

        fig, axs = plt.subplots(1,5, figsize=(50,8))

        fig.tight_layout()

        cmap = matplotlib.colormaps['inferno']
        cmap.set_bad('k',1)

        #vm = max(np.max(cropped_data),np.max(telescope.model()))



        model = params.inject(model)

        coords = dlu.pixel_coords(512, 2.4)
        cropped_frame = exp.data**0.125

        fit = exp.fit(model, exp)

        telescope_frame = fit**0.125

        vm = max(np.nanmax(cropped_frame),np.nanmax(telescope_frame))
        cd=axs[0].imshow(cropped_frame, vmin=0,vmax=vm,cmap=cmap)
        plt.colorbar(cd,ax=axs[0])
        tl=axs[1].imshow(telescope_frame, vmin=0, vmax=vm,cmap=cmap)
        plt.colorbar(tl,ax=axs[1])
        #axs[2].imshow(cropped_err)
        cmap = matplotlib.colormaps['seismic']
        cmap.set_bad('k',1)

        #start_aberrations = model.get(exp.fit.map_param(exp, "start_aberrations"))#*1e-9
        #end_aberrations = model.get(exp.fit.map_param(exp, "end_aberrations"))#*1e-9

        #aberrations_model = model.set(exp.map_param("aberrations"), (start_aberrations+end_aberrations)/2)

        optics = exp.fit.update_optics(model, exp)

        support = optics.transmission(coords,2.4/512)
        support_mask = support.at[support < .5].set(np.nan)

        opd = optics.AberratedAperture.eval_basis(coords)*1e9
        olim = np.nanmax(np.abs(opd*support_mask))
        apt =axs[2].imshow(support_mask*opd,cmap=cmap,vmin=-olim, vmax=olim)
        plt.colorbar(apt, ax=axs[2]).set_label("OPD (nm)")
        #axs[4].imshow(telescope.detector.pixel_response.pixel_response)
        resid = (exp.data - fit)/exp.err
        rlim = np.nanmax(np.abs(resid))
        resid=axs[3].imshow(resid, cmap='seismic',vmin=-rlim, vmax=rlim)
        plt.colorbar(resid,ax=axs[3])

        #axs[3].axvline((wid-1)/2 + params.get(exp.map_param("positions"))[0], color='k',linestyle='--')
        #axs[3].axhline((wid-1)/2 + params.get(exp.map_param("positions"))[1], color='k',linestyle='--')


        lpdf = posterior(model,exp,return_im=True)#*nanmap
        lpd = axs[4].imshow(lpdf)
        plt.colorbar(lpd, ax=axs[4])

        axs[0].set_title("Measured PSF")
        axs[1].set_title("Recovered PSF")
        axs[2].set_title("Recovered Aperture")
        axs[3].set_title("Residuals")
        axs[4].set_title("Log Likelihood Map")

        for i in range(4):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        if save:
            fig.savefig(f"{save}_{f}.png")

def plot_spectra(model, params, exposures):

    model = params.inject(model)

    fishers = calc_fishers(model, exposures, ["spectrum"], recalculate=True)

    for exp in exposures:


        spec = params.get("spectrum")[exp.fit.get_key(exp, "spectrum")]
        fisher = fishers[exp.filename]["spectrum"]

        spectrum_cov = np.linalg.inv(fisher)
        spectrum_err = np.diag(np.sqrt(np.abs(spectrum_cov)))

        plt.imshow(spectrum_cov, cmap='seismic', vmin=-np.max(np.abs(spectrum_cov)), vmax=np.max(np.abs(spectrum_cov)))
        plt.colorbar()

        plt.figure(figsize=(10,10))

        nwavels = 30#len(spec)

        wv, filt = calc_throughput("F110W", nwavels=nwavels)


        for i in range(200):
            coeffs = numpy.random.multivariate_normal(spec, spectrum_cov)
            plt.plot(wv, jax.nn.softplus(NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nwavels), coeffs).weights), color='b', alpha=0.1, zorder=0)

        plt.scatter(wv, jax.nn.softplus(NonNormalisedClippedPolySpectrum(np.linspace(-1, 1, nwavels), spec).weights), color="orange", zorder=1)
