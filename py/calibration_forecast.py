# -*- Encoding: utf-8 -*-

import numpy as np
from croaks import NTuple
import os
from saunerie import bspline
import matplotlib.pyplot as plt
from saunerie.instruments import InstrumentModel
from saunerie.interpolation import Func1D
from saunerie import saltpath, salt2
from saunerie.fitparameters import FitParameters
from saunerie.stellarlibs import FilterSet, ALIGN_WITH_INTEG_FLUX
import scipy.sparse as sparse
from scipy.interpolate import interp1d
from scikits.sparse.cholmod import cholesky
from pycosmo import cosmo
import time
try:
    import pyfits
except ImportError:
    from astropy.io import fits as pyfits
from saunerie.spectrum import Spectrum

###
# Extraction du spectre SALT2
###
print 'Starting --- '+time.ctime()
m = InstrumentModel("LSSTPG")

### Préparation de la loi de couleur
color_law_params  = np.array([ 0.01076587, -0.0856708 ,  0.38838263, -1.3387273 , -0.02079356])
beta = 3.
lb = 4343.78 # Central wavelength of B-band
lv = 5462.1  # Central wavelength of V-band
color_dispersion_source = NTuple.fromtxt('/home/fhazenbe/software/snfit_data/snfit_data/salt2-4/salt2_color_dispersion.dat') # Color Dispersion law from Guy et al.
color_disp_func = interp1d(color_dispersion_source['l'], color_dispersion_source['s'])


### Constantes
A_hc = 50341170.
X0 = 76516964.662612781
###

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

def get_sne_measurments(fichier, lcs=None, bands='grizy', taille_max=100000):
    dt = [('z', float), ('c', float), ('X0', float), 
          ('X1', float), ('dL', float),
          ('band', 'S1'), ('#SN', int),
          ('A', float), 
          ('l_eff', float),
          ('zp', float), 
          ('snr', float)]
    if os.path.exists(fichier):
        print 'Already have a data file --- '+time.ctime()
        data_ok = NTuple.fromfile(fichier).view(np.ndarray)
    elif lcs is not None:
        if len(lcs) > taille_max:
            lcs = np.random.choice(lcs, taille_max)
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
    else:
        raise ValueError('Could not produce a data NTuple without an existing datafile or a list of light curves')
    return np.sort(data_ok, order='z')
############################################################
################################################################


def plot_sn(**kwargs):
    plt.plot(data_ok['l_eff'], rescale_mag(data_ok['A'], data_ok['dL'], data_ok['zp'], data_ok['z']), **kwargs)

def get_standard_wavelength_derivatives(std_spec_file='p330e_stisnic_008.fits', bands='grizy'):
    standard = pyfits.getdata('p330e_stisnic_008.fits')
    spec = Spectrum(standard['WAVELENGTH'], standard['FLUX'])
    spec_2 = Spectrum(standard['WAVELENGTH']-1, standard['FLUX'])
    der_dl = []
    for band in bands:
        filtre = m.EffectiveFilterByBand(band)
        der_dl += [-2.5*np.log10(spec_2.IntegFlux(filtre)/spec.IntegFlux(filtre))]
    return np.array(der_dl)

# fixed_pars = ['theta_salt', 'color_law']
# fixed_pars = []

def transfo(lam):
    ret = (lam-lb)/(lv-lb)
    return ret


#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b + c*Cl(\lambda_b) + \beta*c
###############

#### Fit cosmo
from pycosmo import priors

class Model(object):
    def __init__(self, d, fixed_pars, cosmo=cosmo.Cosmow0wa(), n_spline=30, intrinsic_dispersion=0.08):
        self.int_disp = intrinsic_dispersion
        self.data = d
        self.bands = np.unique(self.data['band'])
        self.n_sn = len(np.unique(d['#SN']))
        self.model_cosmo = cosmo
        self.params, self.spline_base = self.init_params(fixed_pars, n_spline)
        self.free_cosmo_params = []
        self.fixed_pars = fixed_pars
        self.cosmo_free_idx = np.zeros(len(self.model_cosmo.param_names))
        for i, par in enumerate(self.model_cosmo.param_names):
            if self.params[par].indexof() != -1:
                self.free_cosmo_params += [par]
                self.cosmo_free_idx[i] = 1
        self.color_law_degree = len(self.params['color_law'].free)-1
        self.impacted = False
        self.lines, self.cols, self.J_dat = np.zeros(0), np.zeros(0), np.zeros(0)
        self.pedestal = color_disp_func(d['l_eff'])
        try:
            self.der_dl = get_standard_wavelength_derivatives(bands=[b.split('::')[-1] for b in self.bands])
        except:
            self.der_dl = np.zeros(len(self.bands))
        #self.pedestal = 0.02

    def init_params(self, fixed_pars, n_spline):
        data = self.data
        mod = self.model_cosmo
        n = n_spline
        leff = np.array(data['l_eff'])
        n_SN = self.n_sn
        par_list = [('color_law', len(color_law_params)), 'Omega_m', 'Omega_k', 'w', 'wa', 'H0', 'Omega_b_h2', 'beta', ('theta_salt', n+1),  ('mB', n_SN), ('c', n_SN)] + ['zp'+band for band in self.bands] + ['dl'+band for band in self.bands]
        lambda_grid = np.linspace(np.min(leff)-10, np.max(leff)+10, n)
        base = bspline.BSpline(lambda_grid, order=3)
        params = FitParameters(par_list)
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
        snindex = make_index(data['#SN'])
        snindex = [sn[0] for sn in snindex]
        params['mB'] = -2.5*np.log10(X0)
        params['c'] = data[snindex]['c']
        params['color_law'] = color_law_params
        params['Omega_m'] = mod.Omega_m
        params['Omega_k'] = mod.Omega_k
        params['w'], params['wa'] = mod.w, mod.wa
        params['H0'], params['Omega_b_h2'] = mod.H0, mod.Omega_b_h2
        params['beta'] = beta
        zpb = np.zeros(len(data))
        for band in self.bands:
            zpb[data['band']==band] = params['zp'+band].free
        flx = -2.5*np.log10(data['A']) - (params['mB'].free[data['#SN']] + mod.mu(data['z']) + zpb + 20 + 2.5*np.log10(1+data['z']) + data['zp'] - 2.5*np.log10(A_hc) + params['c'].free[data['#SN']]*(np.polyval(params['color_law'].free, transfo(data['l_eff']))+params['beta'].free))
        spectrum_fit = base.linear_fit(np.array(data['l_eff']), np.array(flx))
        params['theta_salt'] = spectrum_fit
        for par in fixed_pars:
            params.fix(par)
        return params, base


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
        zps = np.zeros(len(self.data))
        momo = self.new_cosmo_model(p)
        data = self.data
        for band in self.bands:
            zps[self.data['band']==band] = p['zp'+band].free
        
        # return p['mB'].free[self.data['#SN']] + p['c'].free[self.data['#SN']]*(np.polyval(p['color_law'].free, transfo(self.data['l_eff']))+p['beta'].free) + momo.mu(self.data['z']) + np.dot(self.spline_base.eval(np.array(self.data['l_eff'])).toarray(), p['theta_salt'].free) + zps + 20 + 2.5*np.log10(1+data['z']) + data['zp'] - 2.5*np.log10(A_hc)
        return p['mB'].free[self.data['#SN']] + momo.mu(self.data['z']) + np.dot(self.spline_base.eval(np.array(self.data['l_eff'])).toarray(), p['theta_salt'].free) + zps + 20 + 2.5*np.log10(1+data['z']) + data['zp'] - 2.5*np.log10(A_hc) + p['c'].free[self.data['#SN']]*(np.polyval(p['color_law'].free, transfo(self.data['l_eff']))+p['beta'].free)
    
    def spec(self, p):
        zps = np.zeros(len(self.data))
        momo = self.new_cosmo_model(p)
        return -2.5*np.log10(data['A']) - self(p) + np.dot(self.spline_base.eval(np.array(self.data['l_eff'])).toarray(), p['theta_salt'].free)

    def update_lines_optimized(self, lines, cols, dat, jac=True):
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
        Jp = np.dot(self.J[:len(self.data), :], h)
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
        pedestal = self.pedestal
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
        if 'theta_salt' not in self.fixed_pars:
            bev = base.eval(np.array(d['l_eff']))
            self.lines = np.hstack((self.lines, bev.row)); self.cols = np.hstack((self.cols, bev.col+ p['theta_salt'].indexof(0)))
            self.J_dat = np.hstack((self.J_dat, bev.data))
            #self.update_lines(bev.row, bev.col + p['theta_salt'].indexof(0), bev.data)
        if 'c' not in self.fixed_pars: 
            self.update_lines_optimized(np.arange(len(d)), p['c'].indexof(d['#SN']), beta + np.polyval(color_law_params, transfo(d['l_eff'])))
        if 'mB' not in self.fixed_pars:
            self.update_lines_optimized(np.arange(len(d)), p['mB'].indexof(d['#SN']), np.ones(len(d)))
        for band in self.bands:
            idxband = d['band']==band
            if 'zp'+band not in self.fixed_pars:
                self.update_lines(all_lines[idxband], p['zp'+band].indexof(), 1.)
            if 'dl'+band not in self.fixed_pars:
                #if 'theta_salt' not in self.fixed_pars:
                self.update_lines(all_lines[idxband], p['dl'+band].indexof(), 
                                  (1./(1+d['z'][idxband])*(base.deriv(np.array(d['l_eff'][idxband])) * self.params['theta_salt'].full + d['c'][idxband]*1./(lv-lb)*np.polyval(np.polyder(color_law_params), transfo(d['l_eff'][idxband])))).flatten())
        if 'beta' not in self.fixed_pars:
            self.update_lines(all_lines, p['beta'].indexof(), c)
        kl = self.color_law_degree
        if 'color_law' not in self.fixed_pars:
            self.update_lines(all_lines, p['color_law'].indexof(), np.array([c*transfo(d['l_eff'])**(kl-k) for k in range(kl+1)]).flatten())
        self.Clines = all_lines
        self.Ccols = all_lines
        self.C_dat = 1./d['snr']**2 + pedestal**2
        print '... DONE !'
        self.W = sparse.coo_matrix((1./(1./np.array(d['snr']**2)+pedestal**2), (all_lines, all_lines)))

    def change_pedestal(self, pedestal):
        self.pedestal = pedestal
        d = self.data
        Ns = np.arange(len(d))
        self.reset()
        self.C_dat[:len(d)] = 1./d['snr']**2 + pedestal**2
        #self.C_dat[self.old_idx_C:self.end_idx_C] = self.int_disp**2
        self.C_reco = self.Clines, self.Ccols, self.C_dat
        self.W.data[:len(d)] = 1./(1./np.array(d['snr']**2)+pedestal**2)
        self.W.data[self.old_idx_W:self.end_idx_W] = 1./(self.int_disp**2)
        W = sparse.coo_matrix((self.W.data, (self.W.row, self.W.col)))
        self.W_reco = W
        self.impacted = True
        self.reset()
        print '... DONE !'

    def update_cov(self, cov, w=None, diag=False):
        n, m = np.max(self.Clines)+1, np.max(self.Ccols)+1
        if diag:
            _n, _m = len(cov), len(cov)
            self.update_lines_optimized(n+np.arange(_n), m+np.arange(_m), cov.flatten(), jac=False)
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
                self.update_lines_optimized(n+np.arange(_n), self.params[params].indexof(), entity)
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

    def calib_impact(self, sigma_zp, sigma_dl, covfile=None):
        if self.impacted:
            self.reset()
        else:
            self.J_reco = self.lines, self.cols, self.J_dat
            self.C_reco = self.Clines, self.Ccols, self.C_dat
            self.W_reco = self.W
        ### delta-zp priors
        print 'Adding delta-zp priors'
        if 'dlr' in self.fixed_pars:
            self.update_model(['zp'+band for band in self.bands], np.identity(len(self.bands)), sigma_zp**2*np.identity(len(self.bands)))
        else:
            C_dzp = np.zeros((2*len(self.bands), 2*len(self.bands)))
            for i in range(len(self.bands)): C_dzp[i, i] = sigma_zp**2
            # C_dzp[0, 0] = 1e-16 # NEED TO FIX AT LEAST 1 PARAM?
            A = np.vstack((np.diag(self.der_dl), np.identity(len(self.bands))))
            C_dl = sigma_dl**2 * np.identity(len(self.bands))
            if covfile is None:
                C = C_dzp + np.dot(np.dot(A, C_dl), A.T)
            else:
                C = pyfits.get_data(covfile)
            J_calib = np.identity(2*len(self.bands))
            self.update_model(['zp'+band for band in self.bands]+['dl'+band for band in self.bands], J_calib, C)
        
        self.impacted = True
        if 'dl'+self.bands[0] not in self.fixed_pars:
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
        self.impacted = False
            

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

def extract_blocks(A, s):
    N = A.shape[0]
    assert A.shape[1] == N
    n = len(s)
    i = np.arange(N)
    J = sparse.coo_matrix((np.ones(n), 
                    (i[np.in1d(i,s)], np.arange(n))), shape=(N,N))
    K = sparse.coo_matrix((np.ones(N-n), 
                    (i[~np.in1d(i,s)], np.arange(N-n))), shape=(N,N))

    l = [J.T*A*J, J.T*A*K, K.T*A*J, K.T*A*K]
    m = N-n
    shapes = [(n,n), (n,m), (m,n), (m,m)]
    r = []
    for p,U in enumerate(l):
        U = U.tocoo()
        r.append(sparse.coo_matrix((U.data, (U.row, U.col)), shape=shapes[p]))
        
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

def eval_FoM(DataModel, sigma_zp, sigma_dl, cov_file=None):
    C_calib = DataModel.calib_impact(sigma_zp, sigma_dl, cov_file)
    J, C = DataModel.generate_J(), DataModel.generate_C()
    W = DataModel.W
    Fisher = J.T*W*J
    print 'Starting block decomposition of Fisher matrix --- '+time.ctime()
    covw0wa = block_cov_matrix(Fisher, [DataModel.params['w'].indexof(0), DataModel.params['wa'].indexof(0)])
    print 'OK! --- '+time.ctime()
    return 1. / np.sqrt(np.linalg.det(covw0wa))


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(
        description='data file (one point per meas.)')
    parser.add_argument(
        'datafile',
        help='filenames')
    parser.add_argument(
        '-z', '--sigma_zeropoint', default=0.001, type=float,
        help='Uncertainty on the zeropoint [mag]')
    parser.add_argument(
        '-l', '--sigma_lambda', default=1, type=float,
        help='uncertainty on the mean wavelength [Angstrom]')
    parser.add_argument(
        '-s', '--intrinsic_dispersion', default=0.1, type=float,
        help='Intrinsic dispersion of the SNe Ia absolute magnitude [mag]')
    parser.add_argument(
        '-f', '--fixed_pars', nargs='+', default='',
        help='Parameters that will be fixed')
    parser.add_argument(
        '-c', '--cov_mat', default='',
        help='Covariance matrix for all bands of the combined survey. The columns and the lines have to be ordered alphabetically by band')

    args = parser.parse_args()
    planck = priors.PLANCK
    boss = priors.BOSS
    sdssr = priors.SDSSR
    data_ok = get_sne_measurments(args.datafile)
    MODEL = Model(data_ok, list(args.fixed_pars), intrinsic_dispersion=args.intrinsic_dispersion)

    MODEL.construct_jacobian()
    MODEL.add_priors([planck], el=1e-10)
    if args.cov_mat == '':
        covmat = None
    else:
        covmat = args.cov_mat
    # zp_uncertainties = np.logspace(-5, 0, 20)
    # l_uncertainties = np.logspace(-2, 3, 20)
    # zt, lt = np.meshgrid(zp_uncertainties, l_uncertainties)
    # zt, lt = zt.flatten(), lt.flatten()
    FoM = eval_FoM(MODEL, args.sigma_zeropoint, args.sigma_lambda, covmat)
    print 'Calculated FoM : %.1f' % FoM

    """
    iterator = zt
    for i, a in enumerate(zt):
        print ('Computing FoM for zp_uncertainty : %d over %d --- '+time.ctime()) % (i+1, len(iterator))
        if FoM is None:
            FoM = [eval_FoM(MODEL, a, lt[i])]
        else:
            FoM = np.vstack((FoM, [eval_FoM(MODEL, a, lt[i])]))

    ret = np.zeros(len(zt), dtype=[('sigma_zp', float), ('sigma_dl', float), ('FoM', float)])
    ret['sigma_zp'] = zt
    ret['sigma_dl'] = lt
    ret['FoM'] = FoM[:, 0]

    gr = np.array([1, -1, 0, 0, 0])
    ri = np.array([0, 1, -1, 0, 0])
    iz = np.array([0, 0, 1, -1, 0])
    zy = np.array([0, 0, 0, 1, -1])
    all_vec = [gr, ri, iz, zy]
    color_names = ['gr', 'ri', 'iz', 'zy']
    colors_colors = ['b', 'g', 'y', 'r']

    ### graphic_part
    
    main_foms = [80, 240, 460, 680]
    #main_foms = [80, 240, 320, 680]
    #main_foms = [20, 60]
    foms = np.arange(0, 701, 20)
    fmts = dict(zip(foms, list(np.array(foms).astype(str))))
    lws = [4 if a in main_foms else 0.5 for a in foms]
    hatches = ['/' if i == len(foms)-1 else None for i in range(len(foms))]
    """
