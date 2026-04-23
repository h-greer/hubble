import jax.numpy as np
import jax.scipy as jsp

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
#from astrocut import fits_cut
from astropy.nddata import Cutout2D
import numpy
import pandas as pd

def get_filter(file):
    flt = np.asarray(pd.read_csv(file, sep=' '))#[::20,:]

    wv = flt[:,0]
    bp = flt[:,1]

    ebp = bp#/(wv/1e4)

    nebp = ebp/np.sum(ebp)*(np.max(wv)-np.min(wv))*0.01
    final = flt.at[:,1].set(nebp)
    return final


filter_files = {
    'F170M': get_filter("../data/HST_NICMOS1.F170M.dat")[120:-180],
    'F095N': get_filter("../data/HST_NICMOS1.F095N.dat"),
    'F145M': get_filter("../data/HST_NICMOS1.F145M.dat"),
    'F165M': get_filter("../data/HST_NICMOS1.F165M.dat"),
    'F190N': get_filter("../data/HST_NICMOS1.F190N.dat")[40:220],
    'F164N': get_filter("../data/HST_NICMOS1.F164N.dat")[60:220],
    'F166N': get_filter("../data/HST_NICMOS1.F166N.dat")[60:220],
    'F108N': get_filter("../data/HST_NICMOS1.F108N.dat")[40:140],
    'F187N': get_filter("../data/HST_NICMOS1.F187N.dat")[50:210],
    'F090M': get_filter("../data/HST_NICMOS1.F090M.dat"),
    'F110W': get_filter("../data/HST_NICMOS1.F110W.dat"),#[80:-150],
    'F110M': get_filter("../data/HST_NICMOS1.F110M.dat"),
    'F160W': get_filter("../data/HST_NICMOS1.F160W.dat")[120:-200],
    'POL0S': get_filter("../data/HST_NICMOS1.POL0S.dat"),
    'POL240S': get_filter("../data/HST_NICMOS1.POL240S.dat"),
    'POL120S': get_filter("../data/HST_NICMOS1.POL120S.dat"),
}

def filter_integrate(wl_array, throughput_array, nwavels, norm=False):
    

    # filter_path = os.path.join()
    #file_path = pkg.resource_filename(__name__, f"/data/filters/{filt}.dat")
    #wl_array, throughput_array = np.array(onp.loadtxt(file_path, unpack=True))

    edges = np.linspace(wl_array.min(), wl_array.max(), nwavels + 1)
    wavels = np.linspace(wl_array.min(), wl_array.max(), 2 * nwavels + 1)[1::2]

    areas = []
    for i in range(nwavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = np.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array)/np.sum(cond1 & cond2))

    areas = np.array(areas)
    weights = areas if not norm else areas / np.sum(areas)
    return wavels, weights


def calc_throughput(filt, nwavels=9):

    filtr = filter_files[filt]

    wl_array = filtr[:,0]
    throughput_array = filtr[:,1]

    wavels, weights = filter_integrate(wl_array, throughput_array, nwavels)
    

    wavels *= 1e-10
    return np.array([wavels, weights])