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

from matplotlib import pyplot as plt
import matplotlib


def plot_params(models, groups, xw = 4, save=False):
    yw = int(np.ceil(len(groups)/xw))

    print(len(groups))


    fig, axs = plt.subplots(xw,yw,figsize=(xw*10,yw*8), squeeze=False)
    for i, param in enumerate(groups):
        sp = axs[i%xw, i//xw]
        # print(models[0].get(param))
        if param in ["primary_low", "primary_klip", "spectrum", "primary_opd", "cold_mask_opd","primary_tilt", "cold_mask_tilt", "cold_mask_shift", "cold_mask_shear", "cold_mask_scale", "cold_mask_rot", "primary_rot", "bias", "resolved", "primary_shear"]:

            for j in range(len(list(models[-1].get(param).values()))):
                vals = np.asarray([list(x.get(param).values())[j].flatten() for x in models]).T
                if len(vals.shape)>1:
                    for v in vals:
                        sp.plot(v)
                else:
                    sp.plot(vals)
                sp.set_title(param)
        else:
            sp.set_title(param)
            sp.plot([x.get(param) for x in models])
        
    fig.tight_layout()
    if save:
        fig.savefig(f"{save}.png")


def plot_comparison(model, params, exposures, quadrature=False, save=False, graticule=False, percentile=100, wf_size=512, klip=False):
    for f, exp in enumerate(exposures):

        fig, axs = plt.subplots(2,3, figsize=(30,20), layout='compressed')


        cmap = matplotlib.colormaps['inferno']
        cmap.set_bad('k',1)

        #vm = max(np.max(cropped_data),np.max(telescope.model()))



        model = params.inject(model)

        coords = dlu.pixel_coords(wf_size, model.optics.diameter)
        cropped_frame = exp.data**0.25

        fit = exp.fit(model, exp)

        wid = fit.shape[0]

        telescope_frame = fit**0.25

        vm = max(np.nanmax(cropped_frame),np.nanmax(telescope_frame))
        cd=axs[0, 0].imshow(cropped_frame, vmin=0,vmax=vm,cmap=cmap)
        plt.colorbar(cd,ax=axs[0,0])

        if graticule:
            axs[0, 0].axvline((wid-1)/2 + params.get(exp.map_param("positions"))[0], color='k',linestyle='--')
            axs[0, 0].axhline((wid-1)/2 + params.get(exp.map_param("positions"))[1], color='k',linestyle='--')

        tl=axs[1, 0].imshow(telescope_frame, vmin=0, vmax=vm,cmap=cmap)
        plt.colorbar(tl,ax=axs[1,0])

        if graticule:
            axs[1, 0].axvline((wid-1)/2 + params.get(exp.map_param("positions"))[0], color='k',linestyle='--')
            axs[1, 0].axhline((wid-1)/2 + params.get(exp.map_param("positions"))[1], color='k',linestyle='--')

        #axs[2].imshow(cropped_err)
        cmap = matplotlib.colormaps['bwr']
        cmap.set_bad('k',1)

        #start_aberrations = model.get(exp.fit.map_param(exp, "start_aberrations"))#*1e-9
        #end_aberrations = model.get(exp.fit.map_param(exp, "end_aberrations"))#*1e-9

        #aberrations_model = model.set(exp.map_param("aberrations"), (start_aberrations+end_aberrations)/2)

        optics = exp.fit.update_optics(model, exp)

        support = optics.primary.transmission(coords,2.4/wf_size)
        support_mask = support.at[support < .5].set(np.nan)

        # opd = optics.primary_opd.eval_basis(coords)*1e9
        if klip:
            opd = (optics.primary_opd.eval_basis() + optics.primary_low.eval_basis(coords)+optics.primary_klip.eval_basis())*1e9
        else:
            opd = (optics.primary_opd.eval_basis() + optics.primary_low.eval_basis(coords))*1e9
        olim = np.nanmax(np.abs(opd*support_mask))
        apt =axs[0,1].imshow(support_mask*opd,cmap=cmap,vmin=-olim, vmax=olim)
        plt.colorbar(apt, ax=axs[0,1]).set_label("OPD (nm)")


        support = optics.cold_mask.transmission(coords,2.4/wf_size)
        support_mask = support.at[support < .5].set(np.nan)

        opd = optics.cold_mask_opd.eval_basis(coords)*1e9 
        olim = np.nanmax(np.abs(opd*support_mask))
        apt =axs[1,1].imshow(support_mask*opd,cmap=cmap,vmin=-olim, vmax=olim)
        plt.colorbar(apt, ax=axs[1,1]).set_label("OPD (nm)")


        if quadrature:
            resid = (exp.data - fit)/(exp.err * 10**model.get(exp.fit.map_param(exp, "quadrature")))
        else:
            resid = (exp.data - fit)/exp.err

        print(np.nanstd(resid))
        if percentile < 100:
            rlim = np.nanpercentile(np.abs(resid), percentile)
        else:
            rlim = np.nanmax(np.abs(resid))
        residual=axs[0,2].imshow(resid, cmap='bwr',vmin=-rlim, vmax=rlim)
        plt.colorbar(residual,ax=axs[0,2])

        # if graticule:
        #     axs[4].axvline((wid-1)/2 + params.get(exp.map_param("positions"))[0], color='k',linestyle='--')
        #     axs[4].axhline((wid-1)/2 + params.get(exp.map_param("positions"))[1], color='k',linestyle='--')

        x = np.nanmax(np.abs(resid))
        xs = np.linspace(-x, x, 200)
        ys = jsp.stats.norm.pdf(xs, scale=np.nanstd(resid))#/np.sqrt(np.nanstd(resid))

        axs[1,2].set_title(fr"Noise normalised residual $\sigma ={np.nanstd(resid):.3}$")
        axs[1,2].hist(resid.flatten(), bins=50, density=True)
        axs[1,2].plot(xs, ys, c='k')
        #axs[4].set_xlabel("z-score")
        #axs[4].set_ylabel("Counts")

        #lpdf = posterior(model,exp,return_im=True)#*nanmap
        #lpd = axs[4].imshow(lpdf)
        #plt.colorbar(lpd, ax=axs[4])

        axs[0, 0].set_title("Observed Image")
        axs[1, 0].set_title("Recovered Image")
        axs[0, 1].set_title("Recovered Pupil")
        axs[1, 1].set_title("Recovered Cold Mask")
        axs[0, 2].set_title("Residual z-score")
        #axs[4].set_title("Log Likelihood Map")

        for i in range(4):
            axs[i//2, i%2].set_xticks([])
            axs[i//2, i%2].set_yticks([])
        
        # fig.tight_layout()

        if save:
            fig.savefig(f"{save}_{f}.png")
