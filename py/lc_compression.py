# -*- Encoding: utf-8 -*-

import numpy as np
from croaks import NTuple
import os
import matplotlib.pyplot as plt
from saunerie.instruments import InstrumentModel
from saunerie.interpolation import Func1D
from saunerie import salt2, jlasim
from glob import glob
import lc_simulation
from saunerie import jlasim
import argparse

"""
Simulates a SN Ia dataset or take light curves + SALT2 output
file, then compress the information along the phase axis to obtain
information on the amplitude for the calibration impact study
"""

"""
parser = argparse.ArgumentParser(description='Simulates of import a SN dataset and compress the information along the phase axis to create an output for the calibration impact study')
parser.add_argument('-s', '--survey', type=str, help="Expliciting the survey, if 'jla' you have to give a list of light curves via -f and a SALT2 output file via -a")
parser.add_argument('-o', '--outputname', type=str, default='lc_amplitudes_data.nxt', help='Name of output NTuple')
parser.add_argument('-f', '--lcfiles', default=None, nargs='+', help='Files where to find the light curves (e.g. for JLA)')
parser.add_argument('-a', '--saltoutput', default=None, type=str, help='SALT2 output file to associate with the light curves that were used')
args = parser.parse_args()
"""

output_name = 'data_ntuples/lsst20k.nxt'

ALL_ZPs, ALL_LAMBDAs = {}, {}

def band_from(full_band):
    return full_band.split('::')[-1]

def get_lambda_eff(full_band, instrument):
    if full_band in ALL_LAMBDAs.keys():
        return ALL_LAMBDAs[full_band]
    filtre_trans = instrument.get_transmission(band_from(full_band))
    x_min, x_max = filtre_trans.x_min, filtre_trans.x_max
    step = 1.
    lambdas = np.arange(x_min, x_max+step, step)
    l = Func1D(lambda x: x)
    num = (l**2*filtre_trans).integrate(x_min, x_max)
    den = (l*filtre_trans).integrate(x_min, x_max)
    ALL_LAMBDAs.update({full_band : num/den})
    return num/den

def get_zp(full_band, instrument):
    if full_band in ALL_ZPs.keys():
        return ALL_ZPs[full_band]
    it = 1
    lambdas = np.arange(2000, 20000+it, it)
    filtre =  instrument.get_transmission(band_from(full_band))
    zp = -2.5*np.log10((lambdas*filtre(lambdas)).sum()*it)
    ALL_ZPs.update({full_band : zp})
    return zp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulates of import a SN dataset and compress the information along the phase axis to create an output for the calibration impact study')
    parser.add_argument('-s', '--survey', type=str, help="Expliciting the survey, if 'jla' you have to give a list of light curves via -f and a SALT2 output file via -a")
    parser.add_argument('-o', '--outputname', type=str, default='no_record', help='Name of output NTuple')
    parser.add_argument('-f', '--lcfiles', default=None, nargs='+', help='Files where to find the light curves (e.g. for JLA)')
    parser.add_argument('-a', '--saltoutput', default=None, type=str, help='SALT2 output file to associate with the light curves that were used')
    parser.add_argument('-i', '--instrument', type=str, default='LSSTPG', help='If only one instrument model used, you can specify its name here')
    args = parser.parse_args()

    if args.survey == 'jla':
        isjla = True
    elif args.survey == 'lsst':
        isjla = False
    else:
        raise ValueError('Survey type not recognized')

    if isjla:
        global model_components
        model_components = salt2.ModelComponents('salt2.npz')
        jlasurv = jlasim.JlaSurvey('selected_sn_complete.list')
        lcs_files = glob('/data/betoule/jla1/preproc_GMG5BWI/data/lc*.list')
        lcs = jlasurv.generate_lightcurves(model_components, lcs_files, fit=True)
    else:
        lcs_deep = lc_simulation.create_survey('deep_ideal', n=4, zrange=(0.1, 0.9))
        lcs_wide = lc_simulation.create_survey('wide', zrange=(0.05, 0.35))
        lcs = lcs_wide + lcs_deep
    names = ['i', '#SN', 'band', 'l_eff', 'z', 'flux', 'err', 'dL', 'zp', 'X0', 'X1', 'c', 't0', 'dfdX0', 'dfdX1', 'dfdc', 'dfdt0', 'mjd']
    types = [int,   int,  'S20'] + [float]*15
    dt = zip(names, types)

    all_sn = []
    if not isjla:
        instrument = InstrumentModel(args.instrument)
    for i, lc in enumerate(lcs):
        if isjla:
            instrument = lc.instrument
        N = len(lc.lc)
        data = np.zeros(N, dtype=dt)
        data['band']  = lc.lc['band']
        data['i']     = np.arange(N)
        data['#SN']   = i
        data['z']     = lc.sn['z']
        data['flux']  = lc.lc['flux']
        data['err']   = lc.lc['err']
        data['dL']    = lc.sn['dL']
        data['zp']    = [get_zp(band, instrument) for band in lc.lc['band']]
        data['l_eff'] = [get_lambda_eff(band, instrument) for band in lc.lc['band']]
        data['X0']    = lc.lcmodel.X0(lc.sn)
        data['X1']    = lc.sn['X1']
        data['c']     = lc.sn['Color']
        data['t0']    = lc.sn['DayMax']
        data['dfdX0'] = lc.J[:, 0] 
        data['dfdX1'] = lc.J[:, 1]
        data['dfdc']  = lc.J[:, 2]
        data['dfdt0'] = lc.J[:, 3]
        data['mjd']   = lc.lc['mjd']
        all_sn += [data]
    data    = np.hstack(all_sn)
    data_ok = data[data['band'] != '']
    data_ok['l_eff'] = data_ok['l_eff']/(1+data_ok['z'])
    if args.outputname != 'no_record':
        data_ok.view(NTuple).tofile(args.outputname)
