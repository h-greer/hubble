import jax.numpy as np
import jax.scipy as jsp

def gauss_log_likelihood(psf, data):
    """Per-pixel -log L of the model `psf`, zero at masked pixels. A non-finite model pixel, masked or not, gives +inf
    there, so a blown-up model costs +inf through any nansum instead of being dropped (a NaN pixel would vanish from
    the sum while its gradient stays NaN)."""
    img, err, bad = data
    img = np.where(bad, 0., img)
    err = np.where(bad, 1., err)
    finite = np.isfinite(psf)
    im = -np.where(bad, 0., jsp.stats.norm.logpdf(np.where(finite, psf, 0.), img, err))
    return np.where(finite, im, np.inf)

def posterior(model, exposure, per_pix=False, return_im=False):
    return exposure.fit.loglike(model, exposure, per_pix=per_pix, return_im=return_im)
