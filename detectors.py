import jax
import jax.numpy as np

import dLux as dl
import dLux.utils as dlu
import interpax as ipx
from functools import cache

"""
models for extra weird things that the NICMOS detector does that aren't in base dLux
should be fitted to data first
"""

def interp(image, knot_coords, sample_coords, method="linear", fill=0.0, **kwargs):
    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    return ipx.interp2d(ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill, **kwargs).reshape(
        sample_coords[0].shape
    )


@cache
def cubic2_deriv_matrix(n):
    """Constant (n, n) D with D @ f = interpax's cubic2 df/dx on the pixel_coords(n, 2) grid; cached so jit/grad never
    redo (or differentiate through) its tridiagonal solves."""
    with jax.ensure_compile_time_eval():
        return ipx.approx_df(dlu.pixel_coords(n, 2)[1][:, 0], np.eye(n), "cubic2", axis=0)


class Resample(dl.layers.detector_layers.DetectorLayer):
    """Resamples the oversampled PSF through a rotation, an anisotropy and a polynomial distortion
    (dl.DistortedCoords; its linear terms duplicate the rotation, anisotropy and plate scale and are kept at zero by
    ModelFit). Coordinates span [-1, 1] across the (padded, see NICMOSDetector) image, so an order-k term's coefficient is its shift at the edge
    in units of the half-width (0.01 = 0.4 detector px for an 80 px crop). The map takes detector coordinates to
    model-image coordinates; the image is sampled there and scaled by |det J| of the map, which conserves flux."""
    rotation: float
    anisotropy: np.ndarray
    distortion: dl.DistortedCoords

    def __init__(self, rotation=0.0, anisotropy=1.00, order=3):
        self.rotation = np.array(rotation, float)
        self.anisotropy = np.array(anisotropy, float)
        self.distortion = dl.DistortedCoords(order)   # built here: gen_powers cannot be traced under jit

    def transform(self, coords):
        """Detector coords (2, ...) -> model-image coords: rotation, then anisotropy, then distortion."""
        rot_coords = dlu.rotate_coords(coords, dlu.deg2rad(self.rotation))
        return self.distortion(rot_coords * np.array([1.0, self.anisotropy])[:, None, None])

    def apply(self, PSF):
        coords = dlu.pixel_coords(PSF.data.shape[0], 2)
        point = lambda p: self.transform(p[:, None, None])[:, 0, 0]
        jac = jax.vmap(jax.jacfwd(point))(coords.reshape(2, -1).T)
        det = np.abs(np.linalg.det(jac)).reshape(coords.shape[1:])
        # TODO: Test different interpolation methods
        f = PSF.data
        D = cubic2_deriv_matrix(f.shape[0]).astype(f.dtype)
        fx = D @ f
        return PSF.set("data", det*interp(f, coords, self.transform(coords), "cubic2", fx=fx, fy=f @ D.T, fxy=fx @ D.T))

    def __call__(self, PSF):
        return self.apply(PSF)


class NICMOSDetector(dl.LayeredDetector):
    """pad : the optics model a PSF `pad` detector pixels wider on each side than the data (psf_npixels = wid + 2 pad),
    so Resample never samples beyond the modelled image (it zero-fills there); Resize crops back to wid at the end."""
    def __init__(self: dl.LayeredDetector, oversample, wid, pad=0):
        layers = [
            #("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid*oversample,wid*oversample)))),
            # ("jitter", dl.layers.ApplyJitter(sigma=7/43*oversample)),
            ("resample", Resample(anisotropy=1.)),
            ("downsample", dl.layers.Downsample(oversample)),
            ("bias", dl.layers.AddConstant(value=0.0)),
        ]
        if pad:
            layers.append(("crop", dl.layers.Resize(wid)))
        super().__init__(layers)