# -*- Encoding: utf-8 -*-

import numpy as np
from croaks import NTuple
import os
import matplotlib.pyplot as plt
from saunerie.instruments import InstrumentModel
from scipy.sparse import csr_matrix, coo_matrix
from saunerie.interpolation import Func1D
from saunerie import saltpath, salt2, jlasim
from saunerie.stellarlibs import FilterSet, ALIGN_WITH_INTEG_FLUX
from pycosmo import cosmo
import scipy
from saunerie.spectrum import Spectrum
from saunerie.instruments import FilterWheel
from glob import glob
import lc_simulation
from saunerie import jlasim

"""
Simulates a SN Ia dataset or take light curves + SALT2 output
file, then compress the information along the phase axis to obtain
information on the amplitude for the calibration impact study
"""

"""
parser = argparse.ArgumentParser(description='Simulates of import a SN dataset and compress the information along the phase axis to create an output for the calibration impact study')
parser.add_argument('-s', '--survey', type=str, help="Expliciting the survey, if 'jla' you have to give a list of light curves via -f and a SALT2 output file via -a")
parser.add_argument('-o', '--outputname', type=str, default='lc_amplitudes_data.nxt', help='Name of output NTuple')
parser.add_argument('-f', '--lcfiles', default=None, nargs='+', help='Files where to find the light curves (i.e for JLA)')
parser.add_argument('-a', '--saltoutput', default=None, type=str, help='SALT2 output file to associate with the light curves that were used')
args = parser.parse_args()
"""

output_name = 'data_ntuples/jla_mags.nxt'

ALL_ZPs, ALL_LAMBDAs = {}, {}

def band_from(full_band):
    return full_band.split('::')[-1]

def get_lambda_eff(full_band, lc):
    if full_band in ALL_LAMBDAs.keys():
        return ALL_LAMBDAs[full_band]
    filtre_trans = lc.instrument.get_transmission(band_from(full_band))
    x_min, x_max = filtre_trans.x_min, filtre_trans.x_max
    step = 1.
    lambdas = np.arange(x_min, x_max+step, step)
    l = Func1D(lambda x: x)
    num = (l**2*filtre_trans).integrate(x_min, x_max)
    den = (l*filtre_trans).integrate(x_min, x_max)
    ALL_LAMBDAs.update({full_band : num/den})
    return num/den

def get_zp(full_band, lc):
    if full_band in ALL_ZPs.keys():
        return ALL_ZPs[full_band]
    it = 1
    lambdas = np.arange(2000, 20000+it, it)
    filtre = lc.instrument.get_transmission(band_from(full_band))
    zp = -2.5*np.log10((lambdas*filtre(lambdas)).sum()*it)
    ALL_ZPs.update({full_band : zp})
    return zp

def fill_line(lc, n, band):
    z, dL, DayMax, X1, c, ra, dec, name = lc.sn
    _band, A, snr, zp, dAdX0, dAdX1, dAdC, dAdDM = lc.compressed_data[lc.compressed_data['band']==band][0]
    l_eff = get_lambda_eff(band, lc)
    return band, A, snr, dAdX0, dAdX1, dAdC, dAdDM, z, c, lc.lcmodel.X0(lc.sn), X1, dL, name, n, l_eff, zp
    

isjla = True

if isjla:
    global model_components
    model_components = salt2.ModelComponents('salt2.npz')
    jlasurv = jlasim.JlaSurvey('selected_sn_complete.list')
    lcs_files = glob('/data/betoule/jla1/preproc_GMG5BWI/data/lc*.list')
    lcs = jlasurv.generate_lightcurves(model_components, lcs_files, fit=True)
else:
    lcs_deep = lc_simulation.create_survey('deep_ideal', n=4)
    lcs_wide = lc_simulation.create_survey('wide')
    lcs = lcs_wide + lcs_deep

dt = [('band', 'S20'), 
      ('A', float), ('snr', float), 
      ('dAdX0', float), ('dAdX1', float), 
      ('dAdC', float), ('dAdDM', float), 
      ('z', float), ('c', float), ('X0', float), 
      ('X1', float), ('dL', float), ('name', 'S15'), ('#SN', int),
      ('l_eff', float), ('zp', float)]

data = np.zeros(len(lcs)*8, dtype=np.dtype(dt))
k = 0
i = 0
for lc in lcs:
    bands = np.unique(lc.lc['band'])
    sn_ok = False
    for j, band in enumerate(bands):
        ligne = fill_line(lc, i, band)
        if np.isnan(ligne[3]):
            continue
        sn_ok = True
        data[k+j] = ligne
    k += len(bands)
    if sn_ok:
        i += 1
    else:
        continue

data_ok = data[data['band'] != '']

#data_ok.view(NTuple).tofile(output_name)
