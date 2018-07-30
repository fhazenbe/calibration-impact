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
from saunerie.fitparameters import FitParameters
from saunerie.stellarlibs import FilterSet, ALIGN_WITH_INTEG_FLUX
import scipy.sparse as sparse
from scipy.interpolate import interp1d
from scikits.sparse.cholmod import cholesky
from pycosmo import cosmo
import scipy
import time

###
# Extraction du spectre SALT2
###
print 'Starting --- '+time.ctime()
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
color_law_params  = np.array([ 0.01076587, -0.0856708 ,  0.38838263, -1.3387273 , -0.02079356])
beta = 3.
###

dt = [('z', float), ('c', float), ('X0', float), 
      ('X1', float), ('dL', float),
      ('band', 'S1'), ('#SN', int),
      ('A', float), 
      ('l_eff', float),
      ('zp', float), 
      ('snr', float)]


def find_amplitude(snova, band, k='LSSTPG::'):
    band = k + band
    amplitude = snova.lcmodel(snova.sn, [snova.sn.DayMax], [band])[0]
    return amplitude


#####
##### A regarder plus tard :
####
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

fichier = 'data.nxt'
taille_max = 5000
if os.path.exists(fichier):
    print 'Already have a data file --- '+time.ctime()
    data_ok = NTuple.fromfile(fichier).view(np.ndarray)
else:
    #lcs, log, model = example.main()
    print 'Simulating survey --- '+time.ctime()
    lcs_deep, log_deep, model = example.create_survey('deep_ideal', n=4)
    lcs_wide, log_wide, model_wide = example.create_survey('wide')
    lcs = lcs_wide + lcs_deep
    if len(lcs) > taille_max:
        lcs = np.random.choice(lcs, taille_max)
    #data = np.zeros(4*len(lcs), dtype=dt)
    data = np.zeros(6*len(lcs), dtype=dt)

    lambda_effs = np.zeros(1, dtype=[(band, object) for band in 'grizy'])
    zps = np.zeros(1, dtype=[(band, object) for band in 'grizy'])
    for band in 'grizy':
        lambda_effs[band] = get_lambda_f(band)
        zps[band] = get_zp(band)

    k = 0
    print 'Creating data ntuple (needs SALT2 redo) --- '+time.ctime()
    for i, lc in enumerate(lcs):
        bands = np.unique(lc.lc['band'])
        # if 'LSSTPG::g' in np.unique(lc.lc['band']):
        #     bands = 'griz'
        # else:
        #     bands = 'rizy'
        for j, band1 in enumerate(bands):
            band = band1[-1]
            data[k+j] = (lc.sn['z'], lc.sn['Color'], lc.lcmodel.X0(lc.sn), lc.sn['X1'], lc.sn['dL'], band, i,
                         find_amplitude(lc, band), lambda_effs[band][0], zps[band][0], find_snr(lc, band))
        k += len(bands)
        if (i+1)%100 == 0:
            print 'Data computed for %d supernovae' % (i+1)

    data['l_eff'] = 1./(1.+data['z'])*data['l_eff']
    data = data[data['band'] != '']

    data_ok = data[data['snr'] != 0]
    data_ok.view(NTuple).tofile(fichier)
############################################################
################################################################


def plot_sn(**kwargs):
    plt.plot(data_ok['l_eff'], rescale_mag(data_ok['A'], data_ok['dL'], data_ok['zp'], data_ok['z']), **kwargs)

A_hc = 50341170.


###########################

###############
#### FIT ######
###############
mod = cosmo.Cosmow0wa()
leff = np.array(data_ok['l_eff'])
n_SN = len(np.unique(data_ok['#SN']))
X0 = 76516964.662612781

n = 30
lambda_grid = np.linspace(np.min(leff)-10, np.max(leff)+10, n)
base = bspline.BSpline(lambda_grid, order=3)
idxfit = nt['f0'] ==0
idxfit &= nt['f1'] <= leff.max()+1
idxfit &= nt['f1'] >= leff.min()-1
wls = np.array(nt['f1'][idxfit])
flx = np.array(nt['f2'][idxfit])
spectrum_fit = base.linear_fit(wls, -2.5*np.log10(flx))

par_list = [('color_law', len(color_law_params)), 'Omega_m', 'Omega_k', 'w', 'wa', 'H0', 'Omega_b_h2', 'beta', 'zpg', 'zpr', 'zpi', 'zpz', 'zpy', ('theta_salt', n+1),  ('mB', n_SN), ('c', n_SN), 'dlg', 'dlr', 'dli', 'dlz', 'dly']
#params = FitParameters([('mB', n_SN), ('c', n_SN), ('color_law', len(color_law_params)), 'Omega_m', 'Omega_k', 'w', 'wa', 'H0', 'Omega_b_h2', 'beta', ('theta_salt', n+1), 'zpg', 'zpr', 'zpi', 'zpz', 'zpy'])
params = FitParameters(par_list)
fixed_pars = ['dlg', 'dlr', 'dli', 'dlz', 'dly']

par_names = []
for ppp in par_list:
    if type(ppp) == tuple:
        a = ppp[0]
    else:
        a = ppp
    if a in fixed_pars:
        continue
    else:
        par_names += [a]

from saunerie.indextools import make_index
snindex = make_index(data_ok['#SN'])
snindex = [sn[0] for sn in snindex]
params['mB'] = -2.5*np.log10(X0)
params['c'] = data_ok[snindex]['c']
params['color_law'] = color_law_params
params['Omega_m'] = mod.Omega_m
params['Omega_k'] = mod.Omega_k
params['w'], params['wa'] = mod.w, mod.wa
params['H0'], params['Omega_b_h2'] = mod.H0, mod.Omega_b_h2
params['beta'] = 3.
params['theta_salt'] = spectrum_fit
 
for par in fixed_pars:
    params.fix(par)

#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b + c*Cl(\lambda_b) + \beta*c
###############
def transfo(lam):
    ret = (lam-lb)/(lv-lb)
    return ret

lb = 4343.78
lv = 5462.1

#### Fit cosmo
from pycosmo import priors

class Model(object):
    def __init__(self, d, p, cosmo, spline_base, intrinsic_dispersion=0.15):
        self.data = d
        self.n_sn = len(np.unique(d['#SN']))
        self.spline_base = spline_base
        self.params = p
        self.model_cosmo = cosmo
        self.free_cosmo_params = []
        self.cosmo_free_idx = np.zeros(len(self.model_cosmo.param_names))
        for i, par in enumerate(self.model_cosmo.param_names):
            if self.params[par].indexof() != -1:
                self.free_cosmo_params += [par]
                self.cosmo_free_idx[i] = 1
        self.intrinsic_dispersion = intrinsic_dispersion
        self.color_law_degree = len(p['color_law'].free)-1
        self.impacted = False
        self.Jr = None

    def new_cosmo_model(self, p):
        model = cosmo.Cosmow0wa()
        model.Omega_m = p['Omega_m'].free
        model.Omega_k = p['Omega_k'].free
        model.w = p['w'].free
        model.wa = p['wa'].free
        model.H0 = p['H0'].free
        model.Omega_b_h2 = p['Omega_b_h2'].free
        return model
        
    def __call__(self, p):
        Zps = np.zeros(len(self.data))
        momo = self.new_cosmo_model(p)
        for band in 'grizy':
            Zps[self.data['band']==band] = p['zp'+band].free + zps[band]
        
        return p['mB'].free[self.data['#SN']] + p['c'].free[self.data['#SN']]*(np.polyval(p['color_law'].free, transfo(self.data['l_eff']))+p['beta'].free) + momo.mu(self.data['z']) + np.dot(self.spline_base.eval(np.array(self.data['l_eff'])).toarray(), p['theta_salt'].free) + Zps
        
    def compute_derivatives(self, param, index=None, epsi=1e-4, graph=False):
        p1 = self.params
        p2 = p1.copy()
        h = np.zeros_like(p1.free)
        if index is None:
            p2[param] = p1[param].free + epsi
            h[self.params[param].indexof()] = epsi
        else:
            p2[param][index] = p1[param].free[index] + epsi
            h[self.params[param].indexof(index)] = epsi
        Jp = np.dot(MODEL.J[:len(self.data), :], h)
        if graph:
            plt.plot((self(p2)-self(p1)-Jp), 'o')
        return self(p2), self(p1), Jp
    
    def compare_derivatives(self, param, index=None, epsi=1e-8, graph=True):
        if index is None:
            ret = np.hstack((self.J[:len(self.data), self.params[param].indexof()].reshape((len(self.data), 1)), 
                             self.compute_derivatives(param, index, epsi).reshape((len(self.data), 1))))
        else:
            ret = np.hstack((self.J[:len(self.data), self.params[param].indexof(index)].reshape((len(self.data), 1)), 
                             self.compute_derivatives(param, index, epsi).reshape((len(self.data), 1))))
        if graph:
            plt.plot(self.data['z'], ret[:, 0]-ret[:, 1], 'o', label='model-numerical derivatives')
            #plt.plot(ret[:, 1], 'o', label='numerical derivatives')
            plt.legend()
        return ret      

    def construct_jacobian(self):
        print 'Constructing the Jacobian ...'
        d, p, cosmo, base = self.data, self.params, self.model_cosmo, self.spline_base
        zs = d['z'].reshape((len(d), 1))
        J = np.zeros((len(d), len(p.free)))
        x0 = cosmo.pars()
        eps = np.sqrt(np.sqrt(np.finfo(float).eps))
        #eps = np.finfo(float).eps
        f0 = cosmo.mu(zs)
        c = d['c']
        for par in self.free_cosmo_params:
            h1 = 1e-4
            p2 = self.params.copy()
            p2[par] = self.params[par].free + h1
            J[:, p[par].indexof()] = (self.new_cosmo_model(p2).mu(zs) - f0)/h1
        if 'theta_salt' not in fixed_pars:
            J[:, p['theta_salt'].indexof()] = base.eval(np.array(d['l_eff'])).toarray()
        for j in np.unique(d['#SN']):
            idx = d['#SN'] == j
            snd = d[idx]; taille = len(snd)
            if 'c' not in fixed_pars: 
                J[idx, p['c'].indexof(j)] = beta + np.polyval(color_law_params, transfo(snd['l_eff']))
            if 'mB' not in fixed_pars:
                J[idx, p['mB'].indexof(j)] = np.ones(len(snd))
        for band in 'grizy':
            idxband = d['band']==band
            if 'zp'+band not in fixed_pars:
                J[idxband, p['zp'+band].indexof()] = 1.
            if 'dl'+band not in fixed_pars:
                J[idxband, p['dl'+band].indexof()] = (1./(1+d['z'][idxband])*(base.deriv(np.array(d['l_eff'][idxband])) * params['theta_salt'].free + 1./(lv-lb)*np.polyval(np.polyder(color_law_params), transfo(d['l_eff'][idxband])))).T
        if 'beta' not in fixed_pars:
            J[:, p['beta'].indexof()] = d['c'].reshape((len(d), 1))
        kl = self.color_law_degree
        if 'color_law' not in fixed_pars:
            J[:, p['color_law'].indexof()] = np.array([c*transfo(d['l_eff'])**(kl-k) for k in range(kl+1)]).T
        self.J = J
        self.Jr = J
        Cm = (sparse.coo_matrix(np.diag(1./d['snr']**2))).toarray()
        self.C = Cm
        self.Cr = Cm
        print '... DONE !'
        return J, Cm

    def empty_block(self, taille):
        return np.zeros((taille, self.J.shape[1]))

    def update_cov(self, cov):
        if not np.iterable(cov):
            cov = np.array([[cov]])
        n, m = self.C.shape
        _n, _m = cov.shape
        Cm = np.zeros((n+_n, m+_m))
        Cm[:n, :m] = self.C
        Cm[n:, m:] = cov
        self.C = Cm

    def update_J(self, params, entity):
        ret = self.empty_block(entity.shape[0])
        if type(params) == str:
            ret[:, self.params[params].indexof()] = entity
        elif type(params) == list:
            ret[:, np.hstack([self.params[param].indexof() for param in params])] = entity
        else:
            raise TypeError('Params must be a str of list type')
        self.J = np.vstack((self.J, ret))

    def update_model(self, params, entity, cov):
        if type(params) is str and all(self.params[params].indexof() != -1):
            self.update_J(params, entity)
            self.update_cov(cov)
        elif type(params) is list:
            _params = []
            good_idx = []
            for ii, param in enumerate(params):
                if all(self.params[param].indexof() != -1):
                    _params += [param]
                    good_idx += [ii]
                else:
                    print '%s is fixed' % param
                good_idx = np.array(good_idx)
            if len(_params) != 0:
                self.update_J(_params, entity)
                if len(good_idx) != 0:
                    self.update_cov(cov)
                else:
                    print 'All parameters are fixed in this group'
        elif type(params) is str and all(self.params[params].indexof() == -1):
            print '%s is fixed' % params
        else:
            raise TypeError('Parameters entry type not understood, use *str* or *list* of *str*')

    def add_priors(self, Priors, el=1e-4):
        kl = self.color_law_degree
        d, p, cosmo, base = self.data, self.params, self.model_cosmo, self.spline_base
        lB = 0.
        lV = 1.
        lbw = np.linspace(lb, lv, 300)[:150]
        lvw = np.linspace(lb, lv, 300)[150:]
        ### Color priors
        print 'Adding color priors'
        self.update_model('color_law', np.array([lB**(kl-k) for k in range(kl+1)]).reshape((1, kl+1)), el)
        self.update_model('color_law', np.array([lV**(kl-k) for k in range(kl+1)]).reshape((1, kl+1)), el)
        #self.update_model('c', np.ones(self.n_sn).reshape((1, self.n_sn)), np.array([[1e-2/self.n_sn]]))
        ### Spectrum priors
        print 'Adding spectrum priors'
        self.update_model('theta_salt', 
                          (base.eval(lbw).toarray().sum(axis=0) + base.eval(lvw).toarray().sum(axis=0)).reshape((1, base.n_knots+1)), 
                          el)
        self.update_model('theta_salt', (base.eval(lbw).toarray().sum(axis=0) - base.eval(lvw).toarray().sum(axis=0)).reshape((1, base.n_knots+1)), el)
        ### mb priors
        self.update_model('mB', np.identity(self.n_sn), 0.15**2*np.identity(self.n_sn))
        ### Experiment priors
        print 'Adding experiment priors'
        for pp in Priors:
            self.update_model(self.free_cosmo_params, pp.jac(self.model_cosmo, self.cosmo_free_idx), pp.C)

    def calib_impact(self, sigma_zp, sigma_dl):
        if self.impacted:
            self.J = NTuple.view(self.Jp)
            self.C = NTuple.view(self.Cp)
        else:
            self.Jp = NTuple.view(self.J)
            self.Cp = NTuple.view(self.C)
        ### delta-zp priors
        print 'Adding delta-zp priors'
        self.update_model(['zp'+band for band in 'grizy'], np.identity(5), sigma_zp**2*np.identity(5))
        self.update_model(['dl'+band for band in 'grizy'], np.identity(5), sigma_dl**2*np.identity(5))
        self.impacted = True
        
    def reset(self):
        if self.J is None:
            raise ValueError('Jacobian has not been initialized yet...')
        elif self.Jr is None:
            print 'Jacobian is not yet modified'
        else:
            self.J = self.Jr
            self.C = self.Cr

def color_plot(x, y, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(x, y, **kwargs)
    cz = 0
    cc = 0
    colors = ['cyan', 'g', 'y', 'purple']
    for par in par_names:
        if par[:2] == 'zp':
            if cz == 0:
                ax.axvspan(x[params[par].indexof(0)]-0.5, x[params[par].indexof(0)]+4.5, alpha=0.5, color='red', label='$\delta zp$')
                cz += 1
            else:
                continue
        elif len(params[par].indexof()) == 1:
            if par == 'Omega_m':
                plt.text(x[params[par].indexof(0)], y[params[par].indexof(0)], '$ \Omega_m $', fontsize=17)
            elif par == 'Omega_k':
                plt.text(x[params[par].indexof(0)], y[params[par].indexof(0)], '$ \Omega_k $', fontsize=17)
            elif par == 'beta':
                plt.text(x[params[par].indexof(0)], y[params[par].indexof(0)], '$ \\beta $', fontsize=17)
            elif par == 'Omega_b_h2':
                plt.text(x[params[par].indexof(0)], y[params[par].indexof(0)], '$ \Omega_b h^2 $', fontsize=17)
            else:
                plt.text(x[params[par].indexof(0)], y[params[par].indexof(0)], '$ '+par+' $', fontsize=17)
        else:
            ax.axvspan(x[params[par].indexof(0)]-0.5, x[params[par].indexof(-1)]+0.5, alpha=0.5, color=colors[cc], label=par)
            cc += 1
    plt.legend(loc=0)
            
def inv_mat(M):
    fac = cholesky(M)
    uni = sparse.coo_matrix(np.identity(M.shape[0]))
    return fac(uni)
    

#ncosmo = len(mod.pars())
planck = priors.PLANCK
boss = priors.BOSS
sdssr = priors.SDSSR
MODEL = Model(data_ok, params, mod, base)
MODEL.construct_jacobian()
MODEL.add_priors([planck, boss], el=1e-10)
zp_uncertainties = np.logspace(-4, 0, 15)
l_uncertainties = np.logspace(-2, 2, 15)
FoM = None

for i, a in enumerate(zp_uncertainties):
    print ('Computing FoM for zp_uncertainty : %d over %d --- '+time.ctime()) % (i+1, len(zp_uncertainties))
    #MODEL.calib_impact(zp_uncertainties[0], l_uncertainties[i])
    MODEL.calib_impact(zp_uncertainties[i], 1e-3)
    J = sparse.coo_matrix(MODEL.J)
    C = sparse.coo_matrix(MODEL.C)
    fac = cholesky(C.tocsc())
    A = fac(J.tocsc())
    Fisher = J.T*A
    cov = inv_mat(Fisher.tocsc())
    #sigmas = np.sqrt(np.diagonal(cov.toarray()))
    covw0wa = cov.toarray()[params['w'].indexof(0):params['wa'].indexof(0)+1, params['w'].indexof(0):params['wa'].indexof(0)+1]
    if FoM is None:
        FoM = 1. / np.sqrt(np.linalg.det(covw0wa))
        sigmas = cov.diagonal()
    else:
        FoM = np.vstack((FoM, 1. / np.sqrt(np.linalg.det(covw0wa))))
        sigmas = np.vstack((sigmas, cov.diagonal()))

