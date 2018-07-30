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
from scikits.sparse.cholmod import cholesky_AAt
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
# f = Func1D(lambda x : base.eval(x)*p)                       # Fonction du spectre
# f = Func1D(lambda x : bu.model.basis.eval(x, np.zeros(len(x))) * bu.M0.T.flatten())
# df = Func1D(lambda (x, h) : (f(x+h)-f(x))/h)                # Derivee par rapport a lambda
# d2f = Func1D(lambda (x, h) : (df((x+h, h))-df((x, h)))/h)   # Derivee seconde par rapport a lambda
###
#
###

lcs, log, model = example.main()

dt = [('z', float), ('c', float), ('X1', float), ('dL', float), 
      ('Ag', float), ('Ar', float), ('Ai', float), ('Az', float), ('Ay', float), 
      ('lg', float), ('lr', float), ('li', float), ('lz', float), ('ly', float),
      ('zpg', float), ('zpr', float), ('zpi', float), ('zpz', float), ('zpy', float), 
      ('snr_g', float), ('snr_r', float), ('snr_i', float), ('snr_z', float), ('snr_y', float)]


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


def get_lambda_f(band, z):
    global m
    filtre_trans = m.EffectiveFilterByBand(band)
    x_min = filtre_trans.x_min
    x_max = filtre_trans.x_max
    step = 1.
    lambdas = np.arange(x_min, x_max+step, step)
    l = Func1D(lambda x: x)
    num = (l**2*filtre_trans).integrate(x_min, x_max)
    den = (l*filtre_trans).integrate(x_min, x_max)
    return 1/(1.+z) * num/den

##### Calcul du NTuple des flux interpolés à DayMax
if len(lcs) > 4000:
    lcs1 = np.random.choice(lcs, 4000)
else:
    lcs1 = lcs

data = np.zeros(len(lcs1), dtype=dt)

for i, lc in enumerate(lcs1):
    bands = np.unique(lc.lc['band'])
    airmass = np.mean(lc.lc['airmass'])
    data[i] = (lc.sn['z'], lc.sn['Color'], lc.sn['X1'], lc.sn['dL'], 
               find_amplitude(lc, 'g'), find_amplitude(lc, 'r'), find_amplitude(lc, 'i'), 
               find_amplitude(lc, 'z'), find_amplitude(lc, 'y'), 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 
               find_snr(lc, 'g'), find_snr(lc, 'r'), find_snr(lc, 'i'), 
               find_snr(lc, 'z'), find_snr(lc, 'y'))
    if (i+1)%100 == 0:
        print 'Data computed for %d supernovae' % (i+1)
for band in 'grizy':
    data['l'+band] = get_lambda_f(band, data['z'])
############################################################

### Pas un vrai Zp, : -2.5 log(\int T(\lambda) \lambda d\lambda))
def get_zp(band):
    it = 0.1
    lambdas = np.arange(2000, 12000+it, it)
    filtre = m.EffectiveFilterByBand(band)
    return -2.5*np.log10((lambdas*filtre(lambdas)).sum()*it)

for a in 'grizy':
    data['zp'+a] = get_zp(a)
################################################################


def plot_sn(band, **kwargs):
    plt.plot(data['l'+band], rescale_mag(data['A'+band], data['dL'], data['zp'+band], data['z']), **kwargs)

filters = np.zeros(1, dtype=[(band, object) for band in 'grizy'])
for band in 'grizy':
    filters[band] = m.EffectiveFilterByBand(band)
A_hc = 50341170

def flux_vrai(band, z, dL, it=0.1):
    filtre = filters[band][0]
    lambdas = np.arange(2000, 12000+it, it)
    ig = np.sum(filtre(lambdas)*f(lambdas/(1+z))*lambdas)*it
    return 1e-12 * X0/dL**2 * 1./(1+z) * ig * A_hc * ALIGN_WITH_INTEG_FLUX

def compare_flux(sn_number, band, graph=False):
    sn = data[sn_number]
    f_t = sn['A'+band]
    ff = flux_vrai(band, sn['z'], sn['dL'])
    if graph:
        print 'Le flux mesuré est de %.4f e-/s \nLe flux calculé est de %.4f e-/s' % (f_t, ff)
    return ff
mjds = lcs[42].lc['mjd']
bands = lcs[42].lc['band']
momodel = model.model
filterset = salt2.load_filters(np.unique(bands))
zed = lcs[42].sn['z']
X0 = model.X0_norm

bu = salt2.SALT2(mjds, bands, momodel, filterset, zed)
f = Func1D(lambda x : bu.model.basis.eval(x, np.zeros(len(x))) * bu.M0.T.flatten())
fp = Func1D(lambda (x, p) : bu.model.basis.eval(x, p) * bu.M0.T.flatten())

def test_band(band, zs):
    it = 0.1
    wls = np.arange(2000, 12000+it, it)
    A = []
    B = []
    leff = get_lambda_f(band, 0)
    rapideT = filters[band][0](wls)
    rapide_int = np.sum(rapideT*wls)*it
    B = f(leff/(1+zs))
    for i, z in enumerate(zs):
        A += [np.sum(rapideT*f(wls/(1+z))*wls)*it/rapide_int]
        #B += [f(np.array([leff/(1+z)]))[0]]
        if i % 100 == 0:
            print 'Test done for %d SNe' % i
    plt.plot(leff/(1+zs), A, 'go', label='Spectre in')
    plt.plot(leff/(1+zs), B, 'ro', label='Spectre out')
    plt.legend()
    return A, B, leff
    
###############
#### FIT ######
###############
leff = np.hstack([np.hstack([d['l'+band] for band in 'grizy']) for d in data])
#wls = np.arange(np.min(data['lg']), np.max(data['ly']), 1)
wls = np.arange(3063., 5028.)
Atot = np.hstack([np.hstack([d['A'+band] for band in 'grizy']) for d in data])
ztot = np.hstack([np.hstack([d['z'] for band in 'grizy']) for d in data])
dLtot = np.hstack([np.hstack([d['dL'] for band in 'grizy']) for d in data])
zptot = np.hstack([np.hstack([d['zp'+band] for band in 'grizy']) for d in data])
sigmatot = 1./np.hstack([np.hstack([d['snr_'+band] for band in 'grizy']) for d in data])
salt2_norm = np.sum(-2.5*np.log10(f(wls)*X0))
values = np.append(-2.5*np.log10(Atot) - 2.5*np.log10(ALIGN_WITH_INTEG_FLUX) - 30 - zptot - 2.5*np.log10(1+ztot) + 2.5*np.log10(A_hc), [salt2_norm])
sigmas = 0.1*values
sigmas[-1] = 0.1

n = 20
lambda_grid = np.linspace(np.min(data['lg'])-10, np.max(data['ly'])+10, n)
base = bspline.BSpline(lambda_grid, order=3)
#### MODEL ####
### m_{b} = P(\lambda_b / (1+z)) - 2.5log(X_0) + 30 + \mu 
###         - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z) - 2.5log(A_hc) + \Delta ZP_b
###############
J = base.eval(leff)

### Mu

# mat_mu = np.zeros((J.shape[0], len(leff)/5.))
# i, j = 0, 0
# while i < J.shape[0]:
#     mat_mu[i:i+5, j] = 1
#     i += 5
#     j += 1
N = 4
mat_mu = np.zeros((J.shape[0], N))
for i in range(N):
    mat_mu[:, i] = np.log10(ztot)**i

###
J = np.append(J.toarray(), mat_mu, axis=1)
J2 = np.sum(base.eval(wls).toarray(), axis=0)
J2 = sparse.coo_matrix(np.append(J2, np.zeros(mat_mu.shape[1]), axis=1)).toarray()
J = sparse.coo_matrix(np.append(J, J2, axis=0))
#W1_2 = sparse.coo_matrix(np.diag(np.ones(len(values))))
m_red = np.identity(5)
#m_red[2, 2] = 0
m_red = np.tile(m_red, (len(leff)/5, 1))
m_red = np.vstack((m_red, np.zeros((1, 5))))

J = sparse.coo_matrix(np.append(J.toarray(), m_red, axis=1))
### Ajout des "mesures" de delta ZP
mat_zp = np.zeros((5, J.shape[1]))
mat_zp[0, -5] = 1
mat_zp[1, -4] = 1
mat_zp[2, -3] = 1
mat_zp[3, -2] = 1
mat_zp[4, -1] = 1
J = sparse.coo_matrix(np.append(J.toarray(), mat_zp, axis=0))

sigmas = np.append(sigmas, 1*np.ones(5))
sigmas[-3] = 0.00001
values = np.append(values, np.zeros(5))
W1_2 = np.diag(1/sigmas)
W = sparse.coo_matrix(W1_2**2)
W1_2 = sparse.coo_matrix(W1_2)
VJ = (W1_2 * J).tocsr()
factor = cholesky_AAt(VJ.T)
ga = factor(J.T * W*values)
spec = ga[:n+1]
dist = ga[n+1:-5]
dzps = ga[-5:]
fisher = (VJ.T*VJ).toarray()
cov = np.linalg.inv(fisher)
computed_sigmas = np.diag(np.ones(cov.shape[0]))*cov
computed_sigmas = np.sqrt(computed_sigmas[computed_sigmas!=0].flatten())


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
    JJ = np.vstack([p.wjac(cosmo, np.ones(len(cosmo.pars()))) for p in Priors])
    return 2 * np.dot(JJ.T, JJ)

planck = priors.PLANCK
zp_uncertainties = np.logspace(-4, 0, 20)
#zp_uncertainty = 1e-3
cosmo_sig = None
mag_sigma = 0.01
mod = cosmo.Cosmow0wa()
leff1 = leff #np.linspace(np.min(leff), np.max(leff))
for zp_uncertainty in zp_uncertainties:
    new_jac = base.eval(leff1)
    new_jac = sparse.coo_matrix(np.append(new_jac.toarray(), cosmo_part(mod, ztot), axis=1))
    K = sparse.coo_matrix(np.tile(np.identity(5), len(leff1)/5)).T
    Ws = sparse.coo_matrix(1./(zp_uncertainty)**2*np.identity(5))
    #val = -2.5*np.log10(Atot) - 2.5*np.log10(ALIGN_WITH_INTEG_FLUX) - 30 - zptot - 2.5*np.log10(1+ztot) + 2.5*np.log10(A_hc)
    Wm = sparse.coo_matrix(np.diag(sigmatot))
    #Wm = sparse.coo_matrix(np.diag(np.ones(len(val))))
    Fisher = np.append((new_jac.T*Wm*new_jac).toarray(), (new_jac.T*Wm*K).toarray(), axis=1)
    Fisher = np.append(Fisher, np.append((K.T*Wm*new_jac).toarray(), (K.T*Wm*K).toarray()+Ws.toarray(), axis=1), axis=0)
    planck_prior_fisher = np.zeros_like(Fisher)
    planck_prior_fisher[n+1:-5, n+1:-5] = prior_hessian(mod, [planck])
    Fisher = Fisher + planck_prior_fisher
    cov2 = np.linalg.inv(Fisher)
    cov_cosmo = cov2[n+1:-5, n+1:-5]
    cov_w0wa = cov2[n+3:-7, n+3:-7]
    params_sig = np.sqrt(np.diagonal(cov2))
    if cosmo_sig is None:
        cosmo_sig = params_sig[n+1:-5]
        zp_sig = params_sig[-5:]
        FoM = 1. / np.sqrt(np.linalg.det(cov_w0wa))
    else:
        cosmo_sig = np.vstack((cosmo_sig, params_sig[n+1:-5]))
        zp_sig = np.vstack((zp_sig, params_sig[-5:]))
        FoM = np.vstack((FoM, 1. / np.sqrt(np.linalg.det(cov_w0wa))))
    plot_sig = params_sig[n+1:]
