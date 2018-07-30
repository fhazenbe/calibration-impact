# -*- Encoding: utf-8 -*-
import numpy as np
from croaks import NTuple
import os
import scipy
from saunerie import bspline
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from saunerie.interpolation import Func1D
import saunerie.constants as constants
from saunerie.fitparameters import FitParameters
from saunerie.stellarlibs import FilterSet, ALIGN_WITH_INTEG_FLUX
import scipy.sparse as sparse
from scipy.interpolate import interp1d
from scikits.sparse.cholmod import cholesky
from pycosmo import cosmo
import time
from saunerie.linearmodels import SparseSystem
import pyfits
from saunerie.spectrum import Spectrum


A_hc = 50341170.
#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b + c*Cl(\lambda_b) + \beta*c
###############

### Préparation de la loi de couleur Cl(\lambda_b)
color_law_params  = np.array([ 0.01076587, -0.0856708 ,  0.38838263, -1.3387273 , -0.02079356])
beta = 3.
###


lb = 4343.78
lv = 5462.1

### Color smeering
color_dispersion_source = NTuple.fromtxt('/home/fhazenbe/software/snfit_data/snfit_data/salt2-4/salt2_color_dispersion.dat')
color_disp_func = scipy.interpolate.interp1d(color_dispersion_source['l'], color_dispersion_source['s'], bounds_error=False, fill_value=0.02)


def transfo(l):
    """
    Wavelength transformation as \lambda_B = 0 and \lambda_V = 1
    """
    ret = (l-lb)/(lv-lb)
    return ret

def dlogS(data, key):
    return -2.5*1/data['A']*np.log10(np.abs(data[key]))


#### Fit cosmo
from pycosmo import priors
class Model(object):
    def __init__(self, d, p, cosmo, spline_base, intrinsic_dispersion=0.10):
        global model_type
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
        self.int_disp = intrinsic_dispersion
        self.color_law_degree = len(p['color_law'].free)-1
        self.impacted = False
        self.lines, self.cols, self.J_dat = np.zeros(0), np.zeros(0), np.zeros(0)
        self.cal_C = pyfits.getdata('/home/betoule/cosmo_jla1/results/snlssdss_v9/CalCov.fits')
        #self.cal_C = self.cal_C[:37, :37]
        if self.cal_C.shape[0] != 37:
            for ite in np.arange(62, 69):
                self.cal_C[ite, ite] += 1e-4
            self.cal_C[37:, :] *=10
            self.cal_C[:, 37:] *=10
        self.jambon = []

    def new_cosmo_model(self, p):
        if model_type == "wwa":
            model = cosmo.Cosmow0wa()
            model.Omega_k = p['Omega_k'].free
            model.wa = p['wa'].free
        else:
            model = cosmo.CosmoLambda()
            model.Omega_lambda = p['Omega_lambda'].free
        model.Omega_m = p['Omega_m'].free
        model.w = p['w'].free
        model.H0 = p['H0'].free
        model.Omega_b_h2 = p['Omega_b_h2'].free
        return model

    def update_lines(self, _lines, _cols, dat, jac=True):
        l, c = np.meshgrid(_lines, _cols)
        lines, cols = l.flatten(), c.flatten()
        idx0 = dat != 0
        if np.iterable(dat):
            lines, cols = lines[idx0], cols[idx0]
            dat = dat[idx0]
        if len(lines) != len(cols):
            raise ValueError('Lines and columns have different lengths')
        if not np.iterable(dat):
            dat = dat * np.ones(len(lines))
        if len(dat) != len(lines):
            raise ValueError('Indexes and data have different lengths')
        if jac:
            self.lines = np.hstack((self.lines, lines))
            self.cols = np.hstack((self.cols, cols))
            self.J_dat = np.hstack((self.J_dat, dat))
        else:
            self.Clines = np.hstack((self.Clines, lines))
            self.Ccols = np.hstack((self.Ccols, cols))
            self.C_dat = np.hstack((self.C_dat, dat))

    def update_lines_2(self, lines, cols, dat, jac=True):
        idx0 = dat != 0
        if np.iterable(dat):
            lines, cols = lines[idx0], cols[idx0]
        if len(lines) != len(cols):
            raise ValueError('Lines and columns have different lengths')
        if not np.iterable(dat):
            dat = dat * np.ones(len(lines))
        if len(dat) != len(lines):
            raise ValueError('Indexes and data have different lengths')
        if jac:
            self.lines = np.hstack((self.lines, lines))
            self.cols = np.hstack((self.cols, cols))
            self.J_dat = np.hstack((self.J_dat, dat))
        else:
            self.Clines = np.hstack((self.Clines, lines))
            self.Ccols = np.hstack((self.Ccols, cols))
            self.C_dat = np.hstack((self.C_dat, dat))  

    def construct_jacobian(self):
        print 'Constructing the Jacobian ...'
        d, p, cosmo, base = self.data, self.params, self.model_cosmo, self.spline_base
        zs = d['z'].reshape((len(d), 1))
        x0 = cosmo.pars()
        eps = np.sqrt(np.sqrt(np.finfo(float).eps))
        #eps = np.finfo(float).eps
        f0 = cosmo.mu(zs)
        c = d['c']
        all_lines = np.arange(len(d))
        for par in self.free_cosmo_params:
            h1 = 1e-4
            p2 = self.params.copy()
            p2[par] = self.params[par].free + h1
            self.update_lines(all_lines, p[par].indexof(), ((self.new_cosmo_model(p2).mu(zs) - f0)/h1).flatten())
        if 'theta_salt' not in fixed_pars:
            bev = base.eval(np.array(d['l_eff']))
            self.lines = np.hstack((self.lines, bev.row)); self.cols = np.hstack((self.cols, bev.col+ p['theta_salt'].indexof(0)))
            self.J_dat = np.hstack((self.J_dat, bev.data))
            #self.update_lines(bev.row, bev.col + p['theta_salt'].indexof(0), bev.data)
        if 'c' not in fixed_pars: 
            self.update_lines_2(np.arange(len(d)), p['c'].indexof(d['#SN']), beta + np.polyval(color_law_params, transfo(d['l_eff'])))
        if 'mB' not in fixed_pars:
            self.update_lines_2(np.arange(len(d)), p['mB'].indexof(d['#SN']), np.ones(len(d)))
        if 't0' not in fixed_pars:
            self.update_lines_2(np.arange(len(d)), p['t0'].indexof(d['#SN']), dlogS(d, 'dAdDM'))
        if 'X1' not in fixed_pars:
            self.update_lines_2(np.arange(len(d)), p['X1'].indexof(d['#SN']), dlogS(d, 'dAdX1'))
        for band1 in np.unique(d['band']):
            idxband = d['band']==band1
            ### 
            ### ATTENTION CA CHANGE ICI EN FONCTION DE L UNITE DE ZP ET DL
            ###
            band = band1.replace('::', '_')
            if 'ZP_'+band not in fixed_pars:
                self.update_lines(all_lines[idxband], p['ZP_'+band].indexof(), 1.)
            if 'DL_'+band not in fixed_pars:
                derder = base.deriv(np.array(d['l_eff'][idxband])) * params['theta_salt'].full
                if np.isnan(derder).sum() != 0:
                    self.jambon += list(d['band'][idxband][np.isnan(derder)])
                self.update_lines(all_lines[idxband], p['DL_'+band].indexof(), 
                                  (1./(1+d['z'][idxband])*(derder + d['c'][idxband]*1./(lv-lb)*np.polyval(np.polyder(color_law_params), transfo(d['l_eff'][idxband])))).flatten())
        if 'beta' not in fixed_pars:
            self.update_lines(all_lines, p['beta'].indexof(), c)
        kl = self.color_law_degree
        if 'color_law' not in fixed_pars:
            self.update_lines(all_lines, p['color_law'].indexof(), np.array([c*transfo(d['l_eff'])**(kl-k) for k in range(kl+1)]).flatten())
        self.Clines = np.arange(len(d))
        self.Ccols = np.arange(len(d))
        self.C_dat = 1./d['snr']**2+color_disp_func(d['l_eff'])**2
        print '... DONE !'
        self.W = sparse.coo_matrix(((1./(1./np.array(d['snr']**2)+color_disp_func(d['l_eff'])**2)), (all_lines, all_lines)))


    def update_cov(self, cov, w=None, diag=False):
        n, m = np.max(self.Clines)+1, np.max(self.Ccols)+1
        if diag:
            _n, _m = len(cov), len(cov)
            self.update_lines_2(n+np.arange(_n), m+np.arange(_m), cov.flatten(), jac=False)
            self.W = sparse.block_diag((self.W, sparse.coo_matrix((1./cov, (np.arange(len(cov)), np.arange(len(cov)))))))
        else:
            if not np.iterable(cov):
                cov = np.array([[cov]])
            _n, _m = cov.shape
            self.update_lines(n+np.arange(_n), m+np.arange(_m), np.array(cov).flatten(), jac=False)
            if w is None:
                self.W = sparse.block_diag((self.W, np.linalg.inv(cov)))
            else:
                self.W = sparse.block_diag((self.W, sparse.coo_matrix(w)))

    def update_J(self, params, entity, diag=False):
        n, m = np.max(self.lines)+1, np.max(self.cols)+1
        if type(params) == str:
            if diag:
                _n, _m = len(entity), len(entity)
                self.update_lines_2(n+np.arange(_n), self.params[params].indexof(), entity)
            else:
                _n, _m = entity.shape
                self.update_lines(n+np.arange(_n), self.params[params].indexof(), np.array(entity).flatten())
        elif type(params) == list:
            _n, _m = entity.shape
            self.update_lines(n+np.arange(_n), np.hstack([self.params[param].indexof() for param in params]), np.array(entity).flatten())
        else:
            raise TypeError('Params must be a str of list type')
        print '%d lines were added to J ' % (np.max(self.lines)-n+1)

    def update_model(self, params, entity, cov, w=None, diag=False):
        if type(params) is str and all(self.params[params].indexof() != -1):
            self.update_J(params, entity, diag)
            self.update_cov(cov, w, diag)
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
                self.update_J(_params, entity, diag)
                if len(good_idx) != 0:
                    self.update_cov(cov, w, diag)
            else:
                print 'All parameters are fixed in this group'
        elif type(params) is str and all(self.params[params].indexof() == -1):
            print '%s is fixed' % params
        else:
            raise TypeError('Parameters entry type not understood, use *str* or *list* of *str*')

    def add_priors(self, Priors, el=1e-8):
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
        self.update_model('theta_salt', 
                          (base.eval(lbw).toarray().sum(axis=0) - base.eval(lvw).toarray().sum(axis=0)).reshape((1, base.n_knots+1)), 
                          el)

        ### mb priors
        print 'Adding intrinsic dispersion'
        self.old_idx_W, self.old_idx_C = len(self.W.data), len(self.C_dat)
        self.update_model('mB', np.ones(self.n_sn), np.full(self.n_sn, self.int_disp**2), diag=True)
        self.end_idx_W, self.end_idx_C = len(self.W.data), len(self.C_dat)
        
        ### Experiment priors
        print 'Adding experiment priors'
        for pp in Priors:
            k = np.max(self.lines) + 1
            wj = sparse.coo_matrix(pp.jac(self.model_cosmo, self.cosmo_free_idx))
            self.lines = np.hstack((self.lines, wj.row+k))
            self.cols = np.hstack((self.cols, wj.col+p[self.free_cosmo_params[0]].indexof(0)))
            self.J_dat = np.hstack((self.J_dat, wj.data))
            #self.update_model(self.free_cosmo_params, pp.jac(self.model_cosmo, self.cosmo_free_idx), pp.C)
            self.update_cov(pp.C, w=pp.W)

    def calib_impact(self):
        if self.impacted:
            self.reset()
        else:
            self.J_reco = self.lines, self.cols, self.J_dat
            self.C_reco = self.Clines, self.Ccols, self.C_dat
            self.W_reco = self.W
        ### delta-zp priors
        print 'Adding calibration priors'
        C = self.cal_C
        J_calib = np.identity(self.cal_C.shape[0])
        self.update_model(zp_list+dl_list, J_calib, C)
        
        self.impacted = True
        self.generate_J()
        self.generate_C()
        return C

    def generate_J(self):
        idx_J = self.J_dat != 0
        self.J = sparse.coo_matrix((self.J_dat[idx_J], (self.lines[idx_J], self.cols[idx_J])))
        return self.J

    def generate_C(self):
        idx_C = self.C_dat != 0
        self.C = sparse.coo_matrix((self.C_dat[idx_C], (self.Clines[idx_C], self.Ccols[idx_C])))
        return self.C

    def reset(self):
        if self.impacted:
            self.lines, self.cols, self.J_dat = self.J_reco
            self.Clines, self.Ccols, self.C_dat = self.C_reco
            self.W = self.W_reco
        else:
            return 0



###############
#### FIT ######
###############

data_ok = NTuple.fromfile('data_ntuples/jla_mags.nxt')

data_ok['l_eff'] = data_ok['l_eff'] / (1 + data_ok['z'])

# data_ok = data_ok[~np.isnan(data_ok['dAdX0'])]

data_ok['dAdX0'] = data_ok['dAdX0'] / data_ok['dL']

# model_type = "wwa"
model_type = "lambdaCDM"

if model_type == "wwa":
    mod = cosmo.Cosmow0wa()
else:
    mod = cosmo.CosmoLambda()

data_ok = np.sort(data_ok, order='z')
### Calcul des dérivées du spectre de HD165459 (pour les delta_lambda)
standard = pyfits.getdata('p330e_stisnic_008.fits')
spec = Spectrum(standard['WAVELENGTH'], standard['FLUX'])
spec_2 = Spectrum(standard['WAVELENGTH']-1, standard['FLUX'])

#####################################################################



leff = np.array(data_ok['l_eff'])
n_SN = len(np.unique(data_ok['#SN']))
X0 = 76516964.662612781

n = 20
lambda_grid = np.linspace(np.min(leff)-10, np.max(leff)+10, n)
base_spectre = bspline.BSpline(lambda_grid, order=3)

ext = pyfits.open('/home/betoule/cosmo_jla1/results/snlssdss_v9/smoothed_derivatives.fits')
params_names = ext[2].data['CalPar']

zp_list = list(params_names[:37])
dl_list = list(params_names[37:])

if model_type == "wwa":
    par_list = [('color_law', len(color_law_params)), 'Omega_m', 'Omega_k', 'w', 'wa', 'H0', 'Omega_b_h2', 'beta'] + [('theta_salt', n+1),  ('mB', n_SN), ('c', n_SN), ('X1', n_SN), ('t0', n_SN)] + zp_list + dl_list
else:
    par_list = [('color_law', len(color_law_params)), 'Omega_m', 'Omega_lambda', 'w', 'H0', 'Omega_b_h2', 'beta'] + [('theta_salt', n+1),  ('mB', n_SN), ('c', n_SN), ('X1', n_SN), ('t0', n_SN)] + zp_list + dl_list
params = FitParameters(par_list)

#fixed_pars = dl_list #+ zp_list #+ ['theta_salt']
#fixed_pars = ['theta_salt']
fixed_pars = []

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
if model_type == 'wwa':
    params['w'], params['wa'] = mod.w, mod.wa
    params['Omega_k'] = mod.Omega_k
else:
    params['Omega_lambda'] = mod.Omega_lambda
    params['w'] = mod.w
params['H0'], params['Omega_b_h2'] = mod.H0, mod.Omega_b_h2
params['beta'] = 3.


zpb = np.zeros(len(data_ok))
for band1 in np.unique(data_ok['band']):
    band = band1.replace('::', '_')
    zpb[data_ok['band']==band] = params['ZP_'+band].free
flx = -2.5*np.log10(data_ok['A']) - (params['mB'].free[data_ok['#SN']] + mod.mu(data_ok['z']) - 10 + zpb + 2.5*np.log10(1+data_ok['z']) + data_ok['zp'] - 2.5*np.log10(A_hc) + params['c'].free[data_ok['#SN']]*(np.polyval(params['color_law'].free, transfo(data_ok['l_eff']))+params['beta'].free))
spectrum_fit = base_spectre.linear_fit(np.array(data_ok['l_eff']), np.array(flx))
params['theta_salt'] = spectrum_fit
 
for par in fixed_pars:
    params.fix(par)

def extract_blocks(A, s):
    N = A.shape[0]
    assert A.shape[1] == N
    n = len(s)
    i = np.arange(N)
    J = coo_matrix((np.ones(n), 
                    (i[np.in1d(i,s)], np.arange(n))), shape=(N,N))
    K = coo_matrix((np.ones(N-n), 
                    (i[~np.in1d(i,s)], np.arange(N-n))), shape=(N,N))

    l = [J.T*A*J, J.T*A*K, K.T*A*J, K.T*A*K]
    m = N-n
    shapes = [(n,n), (n,m), (m,n), (m,m)]
    r = []
    for p,U in enumerate(l):
        U = U.tocoo()
        r.append(coo_matrix((U.data, (U.row, U.col)), shape=shapes[p]))
        
    return r

def block_cov_matrix(W, s):
    """
    extract block inverse from W. 
    the indices of the block elements are specified in the array s
    """
    A, B, C, D = extract_blocks(W, s)
    f = cholesky(D)
    w = A - B * f(C)
    return np.linalg.inv(w.todense())

def deriv(f, x):
    h = 1e-1
    return (f(x+h) - f(x))/h

#data_ok['snr'] = data_ok['snr'] * 2
#ncosmo = len(mod.pars())
planck = priors.PLANCK
boss = priors.BOSS
sdssr = priors.SDSSR
MODEL = Model(data_ok, params, mod, base_spectre)
MODEL.construct_jacobian()
MODEL.add_priors([planck], el=1e-8)
FoM = None
if fixed_pars != dl_list + zp_list:
    C_calib = MODEL.calib_impact()
#MODEL.calib_impact(zp_uncertainties[i], 1e-3)
J, C = MODEL.generate_J(), MODEL.generate_C()
W = MODEL.W
Fisher = J.T*W*J
print 'Starting block decomposition of Fisher matrix --- '+time.ctime()

if model_type == "wwa":
    covw0wa = block_cov_matrix(Fisher, [params['w'].indexof(0), params['wa'].indexof(0)])
    FoM = 1. / np.sqrt(np.linalg.det(covw0wa))
    print 'Analysis done with a w0wa cosmology model, we find a FoM of %.1f' % FoM

sigma_w = np.sqrt(block_cov_matrix(Fisher, [params['w'].indexof(0)]))
print 'OK! --- '+time.ctime()

print 'We find an uncertainty on w of %.2f %%' % (sigma_w*100)


# markers = {'MEGACAMPSF':'+', 'SDSS':'*'} 
# for band in np.unique(data_ok['band']):
#     idx = data_ok['band'] == band
#     try:
#         plt.plot(data_ok['l_eff'][idx], flx[idx], markers[band.split('::')[0]])
#     except KeyError:
#         plt.plot(data_ok['l_eff'][idx], flx[idx], 'o')

# for band in np.unique(data_ok['band']):
#     idx = data_ok['band'] == band
#     try:
#         plt.plot(data_ok['A'][idx], data_ok['dAdDM'][idx], markers[band.split('::')[0]])
#     except KeyError:
#         plt.plot(data_ok['A'][idx], data_ok['dAdDM'][idx], 'o')

# for band in np.unique(data_ok['band']):
#     idx = data_ok['band'] == band
#     try:
#         plt.plot(data_ok['z'][idx], data_ok['c'][idx], markers[band.split('::')[0]])
#     except KeyError:
#         plt.plot(data_ok['z'][idx], data_ok['c'][idx], 'o')
