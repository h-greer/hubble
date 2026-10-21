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


# The complete NIC2 filter set, from SVO (HST/NICMOS2.<filter>). Unlike the NIC1 curves used previously,
# none have long low-throughput wings (all exceed 1e-3 of peak within ~50 rows of each end), so none are trimmed
NIC2_FILTERS = ['F110W', 'F160W', 'F165M', 'F171M', 'F180M', 'F187N', 'F187W', 'F190N', 'F204M', 'F205W',
                'F207M', 'F212N', 'F215N', 'F216N', 'F222M', 'F237M', 'POL0L', 'POL120L', 'POL240L']
filter_files = {f: get_filter(f"../data/HST_NICMOS2.{f}.dat") for f in NIC2_FILTERS}

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