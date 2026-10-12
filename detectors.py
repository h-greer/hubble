import jax
import jax.numpy as np

import dLux as dl
import dLux.utils as dlu
import interpax as ipx

"""
models for extra weird things that the NICMOS detector does that aren't in base dLux
should be fitted to data first
"""

def interp(image, knot_coords, sample_coords, method="linear", fill=0.0):
    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    return ipx.interp2d(ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill).reshape(
        sample_coords[0].shape
    )


class Resample(dl.layers.detector_layers.DetectorLayer):
    """Resamples the oversampled PSF through a rotation, an anisotropy and a polynomial distortion
    (dl.DistortedCoords; its linear terms duplicate the rotation, anisotropy and plate scale and are kept at zero by
    ModelFit). Coordinates span [-1, 1] across the image, so an order-k term's coefficient is its shift at the edge
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
        return PSF.set("data", det*interp(PSF.data, coords, self.transform(coords), "cubic2"))

    def __call__(self, PSF):
        return self.apply(PSF)


class NICMOSDetector(dl.LayeredDetector):
    def __init__(self: dl.LayeredDetector, oversample, wid):
        super().__init__(
            [
                #("pixel_response",dl.layers.ApplyPixelResponse(np.ones((wid*oversample,wid*oversample)))),
                # ("jitter", dl.layers.ApplyJitter(sigma=7/43*oversample)),
                ("resample", Resample(anisotropy=1.)),
                ("downsample", dl.layers.Downsample(oversample)),
                ("bias", dl.layers.AddConstant(value=0.0)),
            ]
        )