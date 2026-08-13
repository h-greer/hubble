import jax.numpy as np
import jax.scipy as jsp

"""
stats utilities
"""

def gauss_log_likelihood(psf, data):
    """
    Gaussian log-likelihood of a psf given observed data as (img, err, bad)
    """
    img, err, bad = data
    img = np.where(bad, 0., img)
    err = np.where(bad, 1., err)
    return -np.where(bad, 0., jsp.stats.norm.logpdf(psf, img, err))

def posterior(model, exposure, per_pix=False, return_im=False):
    return exposure.fit.loglike(model, exposure, per_pix=per_pix, return_im=return_im)
