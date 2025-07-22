import jax.numpy as np
import jax.scipy as jsp

def loss_fn(model,exposure):
    img, err, bad = exposure.data, exposure.err, exposure.bad
    psf = exposure.fit(model,exposure)
    img = np.where(bad, psf, img)
    err = np.where(bad, 1e20, err)
    return -np.where(bad, 0., jsp.stats.norm.logpdf(psf, img, err))

def posterior(model, exposure, per_pix=False, return_im=False):
    posterior_im = loss_fn(model, exposure)

    if return_im:
        return posterior_im
    
    if per_pix:
        return np.nanmean(posterior_im)
    return np.nansum(posterior_im)

def orthogonalise(x, cov):
    eig_vals, eig_vecs = np.linalg.eig(cov)
    eig_vals, eig_vecs = eig_vals.real, eig_vecs.real.T
    ortho_cov = np.dot(eig_vecs, np.dot(cov, np.linalg.inv(eig_vecs)))
    ortho_x = np.dot(eig_vecs, x)
    return ortho_x, ortho_cov, eig_vecs, eig_vals