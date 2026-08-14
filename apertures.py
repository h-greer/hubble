from dLux.optical_systems import OpticalSystem
import jax.numpy as np
from jax import Array, vmap

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx

from abcdLux.lct import *
from abcdLux.abcd import *


class HSTMainAperture(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "mirror" : dl.CircularAperture(
                radius = 1.2,
                softening=self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.022*1.2,#0.038*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330*1.2,
                occulting = True,
                softening = self.softening
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921*1.2, 0),
                ),
                softening = self.softening
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7606*1.2),
                ),
                softening=self.softening
            )
        }



class NICMOSColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.955,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.077*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372*1.2,
                occulting = True,
                softening = self.softening
            ),
        }

class NIC2ColdMask(dl.CompoundAperture):
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.9768,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.072*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.357*1.2,
                occulting = True,
                softening = self.softening
            ),

            "pad_1" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.9021*1.2, 0),
                    rotation=np.deg2rad(0)
                ),
                softening = self.softening
            ),
            "pad_2" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7655*1.2),
                    rotation=np.deg2rad(-121.15)
                ),
                softening = self.softening
            ),
            "pad_3" : dl.RectangularAperture(
                width = 0.1650*1.2,
                height = 0.1410*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7706*1.2),
                    rotation=np.deg2rad(121.52)
                ),
                softening = self.softening
            )
        }

def fourier_circle(x0, y0, r0, cc, ss, xx, yy):
    '''
    Create a circle with a radius that varies as a function of angle, defined by a Fourier series with coefficients cc and ss.
    The circle is centered at (x0, y0) and the radius is defined as r0 + cc * cos(theta) + ss * sin(theta), 
    where theta is the angle from the center of the circle to each point in the grid defined by xx and yy.

    Parameters
    ----------
    x0 : float
        x-coordinate of the center of the circle
    y0 : float
        y-coordinate of the center of the circle
    r0 : float
        Base radius of the circle
    cc : float
        Coefficient for the cosine term in the Fourier series
    ss : float
        Coefficient for the sine term in the Fourier series
    xx : jnp.ndarray
        2D array of x-coordinates for the grid
    yy : jnp.ndarray
        2D array of y-coordinates for the grid

    Returns
    -------
    latent : jnp.ndarray
        2D array representing the circle with varying radius, where each point is defined 
        by the distance from the center and the angle to that point.
    '''

    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    theta = np.arctan2(yy - y0, xx - x0)

    radius = r0 + np.sum(vmap(lambda c, s, n: c * np.cos(n*theta) + s * np.sin(n*theta))(cc, ss, np.arange(len(cc))+1), axis=0)**2

    # radius = r0 + (cc * np.cos(theta) + ss * np.sin(theta))**2

    latent = -(rr ** 2)/(radius*2)**2 + 0.5

    return latent

def climb_circle(x0, y0, r0, cc, ss, xx, yy):
    '''
    Apply CLIMB to a Fourier-defined aperture.

    Assumes the latent is oversampled by a factor of 3.
    '''

    latent = fourier_circle(x0, y0, r0, cc, ss, xx, yy)

    latent = dlu.soft_binarise(latent,oversample=3)

    return latent

class CLIMBOcculter(dl.OpticalLayer):
    r: Array
    cc: Array
    ss: Array

    normalise: bool

    def __init__(self, r, cc, ss, normalise: bool = False):
        assert cc.shape == ss.shape
        self.r = r
        self.cc = cc
        self.ss = ss
        self.normalise = normalise
    
    def __call__(self, wavefront):

        xx, yy = dlu.pixel_coords(wavefront.npixels*3, pixel_scale=wavefront.pixel_scale/3)

        circ = climb_circle(0., 0., self.r, self.cc, self.ss, xx, yy)

        wf = wavefront * circ

        if self.normalise:
            return wf.normalise()
        return wf

class SoummerFastObstruction(dl.OpticalLayer):
    layers: dict
    normalise: bool
    def __init__(self, layers, normalise: bool = False):
        self.layers = dlu.list2dictionary(layers, True)
        self.normalise = normalise
    
    def __call__(self, wavefront):
        wf_prop = wavefront
        for prop in self.layers.values():
            wf_prop = prop.apply(wf_prop)
        
        wf = wavefront.flip(axis=(0,1)) - wf_prop
        if self.normalise:
            return wf.normalise()
        return wf


class NICMOSCoronagraph(dl.LayeredOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, n_modes=12, n_zernikes=1.):
        diameter = 2.4
        layers = [
            ("primary",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=2)),

            ("primary_tilt", dl.Tilt(angles=(0.,0.))),

            ("primary_opd", dl.FourierBasis(wf_npixels, n_modes=n_modes)),

            ("primary_low", dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2),
                    noll_inds=np.arange(4,4+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )),

            # ("prop1", dl.MFT(128, pixel_scale=dlu.arcsec2rad(0.01))),
            # ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3), np.zeros(1), np.zeros(1))),
            # ("occulter", dl.CircularAperture(dlu.arcsec2rad(0.3)*24*2.4)),

            # ("prop1", dl.MFT(128, focal_length=24*2.4, pixel_scale=3e-6)),
            # ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3)*24*2.4, np.zeros(1), np.zeros(1))),



            ("occulter", SoummerFastObstruction([
                ("prop1", dl.MFT(128, focal_length=24*2.4, pixel_scale=3e-6)),
                # ("occulter", dl.CircularAperture(dlu.arcsec2rad(0.3)*24*2.4)),
                ("occulter", CLIMBOcculter(dlu.arcsec2rad(0.3)*24*2.4, np.zeros(1), np.zeros(1))),
                ("prop1", dl.MFT(wf_npixels, focal_length=24*2.4, pixel_scale=2.4/wf_npixels)),
            ])),


            ("cold_mask",   NIC2ColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),

            ("cold_mask_opd", dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform(translation=np.asarray((-0.05, -0.05)))),
                    noll_inds=np.arange(4,5),
                    coefficients = np.zeros(1),
                )),

            ("cold_mask_tilt", dl.Tilt(angles=(0.,0.))),
            
            ("prop1", dl.MFT(psf_npixels*oversample, focal_length=45*2.4, pixel_scale=40e-6/oversample)),
        ]

        super().__init__(wf_npixels, diameter, layers)



class NICMOSOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, psf_oversample=1, n_zernikes = 26):
        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,4+n_zernikes),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.zeros(n_zernikes)#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431/psf_oversample,
            oversample
        )


class NICMOSFresnelOptics(dl.AngularOpticalSystem):
    defocus: np.ndarray
    fnumber: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, fnumber, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.fnumber = fnumber

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf *= layer

        u_in = wf.phasor

        fl = self.fnumber*self.diameter
        abcd = compose_abcd([abcd_lens(fl), abcd_free_space(fl + self.defocus)])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop_basic(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(N_out, N_out*dx_out, wavelength).set(
            ["amplitude", "phase"], [np.abs(u_out), np.angle(u_out)]
        )

        if return_wf:
            return wf
        return wf.psf

def abcd_magnification(m):
    return np.array([[m, 0.], [0., 1/m]])

class NICMOSSecondaryFresnelOptics(dl.AngularOpticalSystem):
    defocus: np.ndarray
    despace: np.ndarray
    mag: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, despace, mag, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.despace = despace
        self.mag = mag

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf *= layer

        u_in = wf.phasor

        abcd = compose_abcd([
            abcd_lens(5.52085),
            abcd_free_space(4.907028205 + self.despace),
            abcd_lens(-0.6790325),
            abcd_free_space(6.3919974 + self.despace + self.defocus),
            abcd_magnification(self.mag),
        ])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop_basic(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(N_out, N_out*dx_out, wavelength).set(
            ["amplitude", "phase"], [np.abs(u_out), np.angle(u_out)]
        )

        if return_wf:
            return wf
        return wf.psf


class NICMOSDistortedOptics(dl.AngularOpticalSystem):
    def __init__(self, wf_npixels, psf_npixels, oversample, distortion_orders=5, n_zernikes = 26):

        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=DistortedCoords(order=distortion_orders),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=DistortedCoords(order=distortion_orders), softening=2)),
                    #("bar",dl.Spider(width=2.4,angles=[90],))
                ],normalise=True, transformation=dl.CoordTransform(rotation=np.pi/4)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,4+n_zernikes),#,12,13,14,15,16,17,18,19,20,21,22]),
                    coefficients = np.zeros(n_zernikes),#np.asarray([0,18,19.4,-1.4,-3,3.3,1.7,-12.2])*1e-9,#,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*1e-9
                ),
            ],
            psf_npixels,
            0.0431,
            oversample
        )
    #def apply(self, wavefront):


