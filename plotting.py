import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax
from jax.flatten_util import ravel_pytree
import numpy as onp

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


    fig, axs = plt.subplots(xw,yw,figsize=(xw*10,yw*8), squeeze=False)
    for i, param in enumerate(groups):
        sp = axs[i%xw, i//xw]
        # print(models[0].get(param))
        if isinstance(models[-1].get(param), dict):  # keyed (per exposure / target) parameters

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
    """2x3 diagnostic per exposure: observed and recovered image, pupil and cold mask OPD, residual
    z-score map and its histogram against a Gaussian."""
    model = params.inject(model)
    coords = dlu.pixel_coords(wf_size, model.optics.diameter)

    for f, exp in enumerate(exposures):
        fig, axs = plt.subplots(2, 3, figsize=(30, 20), layout='compressed')
        optics = exp.fit.update_optics(model, exp)
        fit = exp.fit(model, exp)
        wid = fit.shape[0]

        # Images on a quarter-power stretch with a shared colour scale
        vm = max(np.nanmax(exp.data**0.25), np.nanmax(fit**0.25))
        _panel(axs[0, 0], exp.data**0.25, "Observed Image", 'inferno', 0, vm)
        _panel(axs[1, 0], fit**0.25, "Recovered Image", 'inferno', 0, vm)
        if graticule:
            x, y = params.get(exp.map_param("positions"))[:2]
            for ax in axs[:, 0]:
                ax.axvline((wid-1)/2 + x, color='k', linestyle='--')
                ax.axhline((wid-1)/2 + y, color='k', linestyle='--')

        # OPDs in nm, blanked where the aperture does not transmit
        primary_opd = optics.primary_opd.eval_basis() + optics.primary_low.eval_basis(coords)
        if klip:
            primary_opd += optics.primary_klip.eval_basis()
        pupils = [
            (axs[0, 1], optics.primary, primary_opd, "Recovered Pupil"),
            (axs[1, 1], optics.cold_mask, optics.cold_mask_opd.eval_basis(coords), "Recovered Cold Mask"),
        ]
        for ax, aperture, opd, title in pupils:
            support = aperture.transmission(coords, 2.4/wf_size)
            support_mask = support.at[support < .5].set(np.nan)
            _signed_panel(ax, support_mask*opd*1e9, title, label="OPD (nm)")

        err = exp.err
        if quadrature:
            err = err * 10**model.get(exp.fit.map_param(exp, "quadrature"))
        resid = (exp.data - fit)/err
        _signed_panel(axs[0, 2], resid, "Residual z-score", percentile, bad='w')

        # Histogram of the residuals against a Gaussian of the same width
        x = np.nanmax(np.abs(resid))
        xs = np.linspace(-x, x, 200)
        axs[1, 2].set_title(fr"Noise normalised residual $\sigma ={np.nanstd(resid):.3}$")
        axs[1, 2].hist(resid.flatten(), bins=50, density=True)
        axs[1, 2].plot(xs, jsp.stats.norm.pdf(xs, scale=np.nanstd(resid)), c='k')

        if save:
            fig.savefig(f"{save}_{f}.png")

def _panel(ax, im, title, cmap, vmin=None, vmax=None, label=None, bad='k'):
    """imshow with a colourbar, NaNs in `bad` and no ticks."""
    cmap = matplotlib.colormaps[cmap]
    cmap.set_bad(bad, 1)
    cbar = plt.colorbar(ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax)
    if label:
        cbar.set_label(label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _signed_panel(ax, im, title, percentile=100, label=None, bad='k'):
    """bwr panel with colour limits symmetric about zero."""
    lim = np.nanpercentile(np.abs(im), percentile)
    _panel(ax, im, title, 'bwr', -lim, lim, label, bad)


def plot_comparison_detailed(model, params, exposures, quadrature=False, save=False, percentile=100, wf_size=512, klip=False):
    """3x3 diagnostic per exposure. Rows: observed / recovered image and occulter; pupil, cold mask
    and the combined pupil the beam sees; the recovered primary amplitude (percent, blank without a
    primary_amp layer) then residuals in z-score and DN/s."""
    model = params.inject(model)
    coords = dlu.pixel_coords(wf_size, model.optics.diameter)

    for f, exp in enumerate(exposures):
        fig, axs = plt.subplots(3, 3, figsize=(30, 30), layout='compressed')
        optics = exp.fit.update_optics(model, exp)
        fit = exp.fit(model, exp)

        # Images on a quarter-power stretch with a shared colour scale
        vm = max(np.nanmax(exp.data**0.25), np.nanmax(fit**0.25))
        _panel(axs[0, 0], exp.data**0.25, "Observed Image", 'inferno', 0, vm)
        _panel(axs[0, 1], fit**0.25, "Recovered Image", 'inferno', 0, vm)
        mft, occulter, _ = optics.occulter.layers.values()
        _panel(axs[0, 2], occulter.mask(mft.npixels, mft.pixel_scale), "Recovered Occulter", 'inferno', 0, 1)

        # OPDs in nm, blanked where the aperture does not transmit
        primary = optics.primary.transmission(coords, 2.4/wf_size)
        cold_mask = optics.cold_mask.transmission(coords, 2.4/wf_size)
        primary_opd = optics.primary_opd.eval_basis() + optics.primary_low.eval_basis(coords)
        if klip:
            primary_opd += optics.primary_klip.eval_basis()
        cold_mask_opd = optics.cold_mask_opd.eval_basis(coords)

        pupils = [
            (primary, primary_opd, "Recovered Pupil"),
            (cold_mask, cold_mask_opd, "Recovered Cold Mask"),
            (primary*cold_mask, primary_opd + cold_mask_opd, "Pupil x Cold Mask"),
        ]
        for ax, (support, opd, title) in zip(axs[1], pupils):
            _signed_panel(ax, np.where(support < .5, np.nan, opd*1e9), title, label="OPD (nm)")

        # Fractional amplitude variation on the primary, summed over whichever amplitude screens the optics has
        amp_layers = [optics.layers[k] for k in ("primary_amp", "primary_amp_klip") if k in optics.layers]
        if amp_layers:
            amp = 100*(np.exp(sum(layer.eval_basis() for layer in amp_layers)) - 1)
            _signed_panel(axs[2, 0], np.where(primary < .5, np.nan, amp), "Recovered Amplitude",
                          label="Amplitude (%)")
        else:
            axs[2, 0].axis('off')

        # Residuals: data and model are in DN/s
        err = exp.err
        if quadrature:
            err = err * 10**model.get(exp.fit.map_param(exp, "quadrature"))
        resid = exp.data - fit
        residuals = [(resid/err, "z-score"), (resid, "DN/s")]
        for ax, (im, unit) in zip(axs[2, 1:], residuals):
            _signed_panel(ax, im, fr"Residual ({unit}), $\sigma = {np.nanstd(im):.3}$", percentile, unit, bad='w')

        if save:
            fig.savefig(f"{save}_{f}.png")


# primary_opd[i, j] and primary_amp[i, j] coefficients to show. Along each axis index 0 is the DC
# term, odd i = 2k-1 is cos(k) and even i = 2k is sin(k), with k = (i+1)//2 cycles across the
# wavefront array. The sample spans k = 0..22 along each axis, the diagonal and mixed
# orientations, and both phases.
# (0, 0) is the piston, which update_optics zeroes.
OPD_SAMPLE = [(1, 0), (0, 2), (1, 1), (4, 3), (9, 0), (0, 12), (7, 7), (16, 5), (25, 25), (44, 43)]


def model_jacobian(model, params, exp, opd_sample=OPD_SAMPLE, batch_size=2):
    """Jacobian of the model image of `exp` with respect to every element of every param in
    `params` (only `opd_sample` for primary_opd and primary_amp). Returns (columns, jacobian): a
    list of (name, index) and a (n_columns, wid, wid) array in DN/s per unit, NaN at bad pixels."""
    model = params.inject(model)

    # This exposure's entry of each param (per-exposure dicts also hold other exposures' entries,
    # whose Jacobian is zero), flattened into one vector
    paths = {name: exp.fit.map_param(exp, name) for name in params.params}
    values = {name: np.asarray(model.get(path), dtype=float) for name, path in paths.items()}
    flat, unravel = ravel_pytree(values)

    def model_image(flat):
        new = unravel(flat)
        return exp.fit(model.set(list(paths.values()), [new[name] for name in paths]), exp)

    # One column per (name, index), each with a one-hot tangent
    columns = [(name, idx) for name in values
               for idx in (opd_sample if name in ("primary_opd", "primary_amp") else onp.ndindex(values[name].shape))]

    def one_hot(name, idx):
        tangent = jax.tree.map(np.zeros_like, values)
        tangent[name] = tangent[name].at[idx].set(1.)
        return ravel_pytree(tangent)[0]

    tangents = np.stack([one_hot(name, idx) for name, idx in columns])

    # Forward mode, with the tangents passed in as an argument: on this JAX version jitted JVPs with
    # constant tangents are wrong, and reverse mode is wrong for primary_rot and primary_shear.
    @eqx.filter_jit
    def jacobian_columns(flat, tangents):
        jvp = lambda t: jax.jvp(model_image, (flat,), (t,))[1]
        return jax.lax.map(jvp, tangents, batch_size=batch_size)

    jacobian = onp.array(jacobian_columns(flat, tangents))
    jacobian[:, onp.asarray(exp.bad)] = onp.nan
    return columns, jacobian


def plot_jacobian(columns, jacobian, ncols_max=5, save=False):
    """One figure per param, one panel per element, from the output of model_jacobian."""
    names = dict.fromkeys(name for name, _ in columns)  # unique, in order
    for name in names:
        ks = [k for k, (n, _) in enumerate(columns) if n == name]
        ncols = min(len(ks), ncols_max)
        nrows = -(-len(ks) // ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 10*nrows), layout='compressed', squeeze=False)

        for ax, k in zip(axs.ravel(), ks):
            idx = columns[k][1]
            title = f"{name}[{','.join(map(str, idx))}]" if idx else name
            _signed_panel(ax, jacobian[k], title, label="DN/s per unit")
        for ax in axs.ravel()[len(ks):]:
            ax.axis('off')

        if save:
            fig.savefig(f"{save}_{name}.png")
