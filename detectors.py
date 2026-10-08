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
    rotation: float
    anisotropy: np.ndarray

    def __init__(self, rotation=0.0, anisotropy=1.00):
        self.rotation = np.array(rotation, float)
        self.anisotropy = np.array(anisotropy, float)

    def apply(self, PSF):
        angle = dlu.deg2rad(self.rotation)
        coords = dlu.pixel_coords(PSF.data.shape[0], 2)
        rot_coords = dlu.rotate_coords(coords, angle)
        sample_coords = rot_coords * np.array([1.0, self.anisotropy])[:, None, None]
        # TODO: Test different interpolation methods
        return PSF.set("data", interp(PSF.data, coords, sample_coords, "cubic2"))
    
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