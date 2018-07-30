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
import scipy

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


### Préparation de la loi de couleur
#color_law_params = np.array([  1.86053680e-13,  -3.60052385e-09,   2.60815642e-05, -8.46865354e-02,   1.04582345e+02])
#color_law_params = np.array([  4.40357739e-14,  -1.03606013e-09,   9.09426282e-06, -3.58966206e-02,   5.34152535e+01])
color_law_params = np.array([  4.83000620e-14,  -1.13062696e-09,   9.85181326e-06, -3.84767882e-02,   5.65383601e+01])

beta = 3.
###

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

if os.path.exists('data.nxt'):
    data_ok = NTuple.fromfile('data.nxt').view(np.ndarray)
else:
    #lcs, log, model = example.main()
    lcs_deep, log_deep, model = example.create_survey('deep')
    lcs_wide, log_wide, model_wide = example.create_survey('wide')
    lcs = lcs_wide + lcs_deep

    if len(lcs) > 10000:
        lcs1 = np.random.choice(lcs, 10000)
    else:
        lcs1 = lcs

    data = np.zeros(4*len(lcs1), dtype=dt)

    lambda_effs = np.zeros(1, dtype=[(band, object) for band in 'grizy'])
    zps = np.zeros(1, dtype=[(band, object) for band in 'grizy'])
    for band in 'grizy':
        lambda_effs[band] = get_lambda_f(band)
        zps[band] = get_zp(band)

    for i, lc in enumerate(lcs1):
        if 'LSSTPG::g' in np.unique(lc.lc['band']):
            bands = 'griz'
        else:
            bands = 'rizy'
        for j, band in enumerate(bands):
            data[4*i+j] = (lc.sn['z'], lc.sn['Color'], lc.sn['X1'], lc.sn['dL'], band, i,
                         find_amplitude(lc, band), lambda_effs[band][0], zps[band][0], find_snr(lc, band))
        if (i+1)%100 == 0:
            print 'Data computed for %d supernovae' % (i+1)

    data['l_eff'] = 1./(1.+data['z'])*data['l_eff']

    data_ok = data[data['snr'] != 0]
    data_ok.view(NTuple).tofile('data.nxt')
############################################################
################################################################


def plot_sn(**kwargs):
    plt.plot(data['l_eff'], rescale_mag(data['A'], data['dL'], data['zp'], data['z']), **kwargs)

A_hc = 50341170


###########################

###############
#### FIT ######
###############

leff = data_ok['l_eff']
wls = np.arange(3063., 5028.)
Atot = data_ok['A']
ztot = data_ok['z']
dLtot = data_ok['dL']
zptot = data_ok['zp']
sigmatot = 1./data_ok['snr']

n = 20
lambda_grid = np.linspace(np.min(leff)-10, np.max(leff)+10, n)
base = bspline.BSpline(lambda_grid, order=3)

#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b + c*Cl(\lambda_b) + \beta*c
###############


#### Fit cosmo
from pycosmo import priors

def cosmo_part(cosmo, zs):
    x0 = cosmo.pars()
    J = np.zeros((len(zs), len(x0)))
    eps = np.sqrt(np.sqrt(np.finfo(float).eps))
    f0 = cosmo.mu(zs) #- 10
    for i in range(len(x0)):
        h1 = eps * np.abs(x0[i])
        if h1 == 0:
            h1 = eps
        pars = [p for p in x0]
        pars[i] += h1
        indexes = np.zeros(len(pars)).astype(bool)
        indexes[i] = True
        cosmo.update_pars(pars[i], indexes)
        J[:, i] = (cosmo.mu(zs) - f0)/h1
    return J

def prior_hessian(cosmo, Priors=[]):
    print 'Calculating Fisher matrix contribution from the Planck prior ...'
    #cosmo.update_pars(b, freepar)
    JJ = np.vstack([p.wjac(cosmo, np.array([1, 1, 1, 1, 1, 1])) for p in Priors])
    fish = np.dot(JJ.T, JJ)
    fish[1, 1] = fish[1, 1] #+ 1e7
    print '... Done!'
    return fish

band_number = {'g':0, 'r':1, 'i':2, 'z':3, 'y':4}

def get_zp_mat(d):
    print 'Calculating the zero-point part of the Jacobian ...'
    zp_mat = None
    for i in np.unique(d['#SN']):
        idx = d['#SN'] == i
        snd = d[idx]
        for band in snd['band']:
            ligne = np.zeros(5)
            ligne[band_number[band]] = 1
            if zp_mat is None:
                zp_mat = ligne
            else:
                zp_mat = np.vstack((zp_mat, ligne))
    print '... Done'
    return zp_mat

def get_noise_mat(d, val):
    print 'Extracting measurements covariance matrix ...'
    noise_mat = []
    for i in np.unique(d['#SN']):
        idx = d['#SN'] == i
        snd = d[idx]; taille = len(snd)
        a = np.matrix(val*np.ones(taille))
        noise_mat += [a.T*a]
    print '... Done!'
    return sparse.block_diag(noise_mat)

def get_color_mat(data):
    print 'Calculating the color part of the Jacobian ...'
    Jc = []
    for d in data:
        c_arr = np.zeros(len(np.unique(data['#SN'])))
        c_arr[d['#SN']] = beta + np.polyval(color_law_params, d['l_eff'])
        Jc += [np.hstack((c_arr, d['c'], d['c']*np.array([d['l_eff']**4, d['l_eff']**3, d['l_eff']**2, d['l_eff'], 1])))]
    Jc = np.vstack(Jc)
    print '... Done!'
    return Jc

def color_priors(J, indice, n_SN):
    print 'Adding the convenient lines to constrain the model ...'
    lb = 4343.78
    lv = 5462.1
    lbw = np.linspace(lb, lv, 300)[:150]
    lvw = np.linspace(lb, lv, 300)[150:]
    val = np.zeros((4, J.shape[1]))
    #val[2, indice:indice+n_SN] = 1
    val[0, indice+n_SN+1:indice_mb] = np.array([lb**4, lb**3, lb**2, lb, 1])
    val[1, indice+n_SN+1:indice_mb] = np.array([lv**4, lv**3, lv**2, lv, 1])
    val[2, :n+1] = base.eval(lbw).toarray().sum(axis=0) + base.eval(lvw).toarray().sum(axis=0)
    val[3, :n+1] = base.eval(lbw).toarray().sum(axis=0) - base.eval(lvw).toarray().sum(axis=0)
    #val[3, :n+1] = base.eval(np.array([lb])).toarray()[0]
    #val[4, 6000] = 1
    #val[3, :n+1] = base.eval(np.array([lb])).toarray()[0] - base.eval(np.array([lv])).toarray()[0]
    print '... Done!'
    return val

def get_mb_mat(data):
    print 'Calculating the mb part of the Jacobian ...'
    Jm = []
    for d in data:
        m_arr = np.zeros(len(np.unique(data['#SN'])))
        m_arr[d['#SN']] = 1
        Jm += [m_arr]
    Jm = np.vstack(Jm)
    print '... Done!'
    return Jm    

n_SN = len(np.unique(data_ok['#SN']))
mod = cosmo.Cosmow0wa()
ncosmo = len(mod.pars())      
indice = n+7
indice_mb = indice + n_SN + 6
planck = priors.PLANCK
boss = priors.BOSS
zp_uncertainties = np.logspace(-4, 0, 10)
new_jac = base.eval(np.array(data_ok['l_eff']))
new_jac = np.hstack((new_jac.toarray(), cosmo_part(mod, ztot), get_color_mat(data_ok), get_mb_mat(data_ok)))
constraints = color_priors(new_jac, indice, n_SN)
lignes = constraints.shape[0]
new_jac = sparse.coo_matrix(np.vstack((new_jac, constraints)))
K = get_zp_mat(data_ok)
K = sparse.coo_matrix(np.vstack((K, np.zeros((lignes, K.shape[1])))))

### Adaptation de la matrice de covariance Cm
Cm = sparse.coo_matrix(np.diag(sigmatot**2)) + get_noise_mat(data_ok, 0.1)
Cm = np.hstack((Cm.toarray(), np.zeros((Cm.shape[0], lignes))))
Cm = np.vstack((Cm, np.zeros((lignes , Cm.shape[1]))))
for i in range(lignes):
    Cm[-(i+1), -(i+1)] = 1e-7
Cm = sparse.coo_matrix(Cm)
###

print 'Factorizing the measurements covariance matrix ...'
fac = cholesky(Cm.tocsc())
print '... Done'
#Wm = sparse.linalg.inv(Cm)
cosmo_sig = None
A = fac(new_jac.tocsc())
B = fac(K.tocsc())
print 40*'#'+'\nEntering the FoM vs Ws calculation part\n'+40*'#'
for i, zp_uncertainty in enumerate(zp_uncertainties):
    Ws = sparse.coo_matrix(1./(zp_uncertainty)**2*np.identity(5))
    Fisher = np.append((A.T*A).toarray(), (A.T*B).toarray(), axis=1)
    #Fisher = np.append((new_jac.T*Wm*new_jac).toarray(), (new_jac.T*Wm*K).toarray(), axis=1)
    #Fisher = np.append(Fisher, np.append((K.T*Wm*new_jac).toarray(), (K.T*Wm*K).toarray()+Ws.toarray(), axis=1), axis=0)
    Fisher = np.append(Fisher, np.append((B.T*A).toarray(), (B.T*B).toarray()+Ws.toarray(), axis=1), axis=0)
    planck_prior_fisher = np.zeros_like(Fisher)
    planck_prior_fisher[n+1:n+1+ncosmo, n+1:n+1+ncosmo] = prior_hessian(mod, [planck, boss])
    Fisher = sparse.coo_matrix(Fisher + planck_prior_fisher)
    print '--> Inverting Fisher matrix'
    cov2 = sparse.linalg.inv(Fisher)
    print '--> Done'
    cov_cosmo = cov2[n+1:n+1+ncosmo, n+1:n+1+ncosmo]
    cov_w0wa = cov2[n+3:n+5, n+3:n+5]
    params_sig = np.sqrt(np.diagonal(cov2))
    if cosmo_sig is None:
        cosmo_sig = params_sig[n+1:n+1+ncosmo]
        zp_sig = params_sig[-5:]
        FoM = 1. / np.sqrt(np.linalg.det(cov_w0wa))
        sigmas_tot = params_sig
    else:
        cosmo_sig = np.vstack((cosmo_sig, params_sig[n+1:n+1+ncosmo]))
        zp_sig = np.vstack((zp_sig, params_sig[-5:]))
        FoM = np.vstack((FoM, 1. / np.sqrt(np.linalg.det(cov_w0wa))))
        sigmas_tot = np.vstack((sigmas_tot, params_sig))
    plot_sig = params_sig[n+1:]
    print '%d iterations done over %d' % (i, len(zp_uncertainties))
