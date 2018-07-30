# -*- Encoding: utf-8 -*-

import numpy as np
from croaks import NTuple
import os
from saunerie import bspline
import example
import matplotlib.pyplot as plt
from saunerie.instruments import InstrumentModel
from scipy.sparse import csr_matrix, coo_matrix
from saunerie.interpolation import Func1D
import saunerie.constants as constants
from saunerie import saltpath, salt2
from saunerie.stellarlibs import FilterSet, ALIGN_WITH_INTEG_FLUX
import scipy.sparse as sparse
from scipy.interpolate import interp1d
from scikits.sparse.cholmod import cholesky
from pycosmo import cosmo

###
# Extraction du spectre SALT2
###
m = InstrumentModel("LSSTPG")
cfg = saltpath.read_card_file(saltpath.fitmodel_filename)
salt2_model_path = saltpath.SALTPATH + os.sep + cfg['SALT2']
M0_filename = salt2_model_path + os.sep + 'salt2_template_0.dat'
nt = NTuple.fromtxt(M0_filename)
idx = nt['f0'] == 0
gx = np.linspace(nt['f1'][idx].min()-1e-10, nt['f1'][idx].max()+1e-10, 100)
base = bspline.BSpline(gx, order=4)
p = base.linear_fit(nt['f1'][idx], nt['f2'][idx])

lcs0, log0, model = example.main(0)
lcs1, log1, model = example.main(1)

color_law_params = np.array([  1.86053680e-13,  -3.60052385e-09,   2.60815642e-05, -8.46865354e-02,   1.04582345e+02])

dt = [('z', float), ('c', float), ('X1', float), ('dL', float),
      ('band', 'S1'), ('#SN', int),
      ('A', float), 
      ('l_eff', float),
      ('zp', float), 
      ('snr', float)]


def find_amplitude(snova, band, k='LSSTPG::'):
    band = k + band
    amplitude = snova.lcmodel(snova.sn, [snova.sn.DayMax], [band])[0]
    return amplitude

def find_snr(snova, band, k='LSSTPG::'):
    band = k + band
    snr = snova.amplitude_snr(band)
    return snr

def rescale_mag(fi, dL, zp, z):
    A_hc = 50341170
    M = -2.5*np.log10(fi) - 5*( np.log10(dL) ) - 30 + 2.5 * np.log10(ALIGN_WITH_INTEG_FLUX) - zp + 2.5*np.log10(A_hc) - 2.5*np.log10(1+z)
    return M


def get_lambda_f(band):
    global m
    filtre_trans = m.EffectiveFilterByBand(band)
    x_min = filtre_trans.x_min
    x_max = filtre_trans.x_max
    step = 1.
    lambdas = np.arange(x_min, x_max+step, step)
    l = Func1D(lambda x: x)
    num = (l**2*filtre_trans).integrate(x_min, x_max)
    den = (l*filtre_trans).integrate(x_min, x_max)
    return num/den

### Pas un vrai Zp, : -2.5 log(\int T(\lambda) \lambda d\lambda))
def get_zp(band):
    it = 0.1
    lambdas = np.arange(2000, 12000+it, it)
    filtre = m.EffectiveFilterByBand(band)
    return -2.5*np.log10((lambdas*filtre(lambdas)).sum()*it)

##### Calcul du NTuple des flux interpolés à DayMax

lambda_effs = np.zeros(1, dtype=[(band, object) for band in 'ugrizy'])
zps = np.zeros(1, dtype=[(band, object) for band in 'ugrizy'])
for band in 'ugrizy':
    lambda_effs[band] = get_lambda_f(band)
    zps[band] = get_zp(band)

nfil = len('ugrizy')

def compute_data(lcs):
    data = np.zeros(nfil*len(lcs), dtype=dt)
    for i, lc in enumerate(lcs):
        for j, band in enumerate('ugrizy'):
            band = band[-1]
            data[nfil*i+j] = (lc.sn['z'], lc.sn['Color'], lc.sn['X1'], lc.sn['dL'], band, i,
                           find_amplitude(lc, band), lambda_effs[band][0], zps[band][0], find_snr(lc, band))
        if (i+1)%100 == 0:
            print 'Data computed for %d supernovae' % (i+1)
    data['l_eff'] = 1./(1.+data['z'])*data['l_eff']
    return data

############################################################
################################################################


def plot_sn(**kwargs):
    plt.plot(data['l_eff'], rescale_mag(data['A'], data['dL'], data['zp'], data['z']), **kwargs)

A_hc = 50341170

#### Exemple de modèle #####
mjds = lcs[42].lc['mjd']
bands = lcs[42].lc['band']
momodel = model.model
filterset = salt2.load_filters(np.unique(bands))
zed = lcs[42].sn['z']
X0 = model.X0_norm

bu = salt2.SALT2(mjds, bands, momodel, filterset, zed)
f = Func1D(lambda x : bu.model.basis.eval(x, np.zeros(len(x))) * bu.M0.T.flatten())
fp = Func1D(lambda (x, p) : bu.model.basis.eval(x, p) * bu.M0.T.flatten())
###########################
    
###############
#### FIT ######
###############
data_ok = data[data['snr'] != 0]

leff = data_ok['l_eff']
wls = np.arange(3063., 5028.)
Atot = data_ok['A']
ztot = data_ok['z']
dLtot = data_ok['dL']
zptot = data_ok['zp']
sigmatot = 1./data_ok['snr']
salt2_norm = np.sum(-2.5*np.log10(f(wls)*X0))
values = np.append(-2.5*np.log10(Atot) - 2.5*np.log10(ALIGN_WITH_INTEG_FLUX) - 30 - zptot - 2.5*np.log10(1+ztot) + 2.5*np.log10(A_hc), [salt2_norm])

n = 20
lambda_grid = np.linspace(np.min(leff)-10, np.max(leff)+10, n)
base = bspline.BSpline(lambda_grid, order=3)

#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b
###############
