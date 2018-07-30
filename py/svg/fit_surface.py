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


lcs, log, model = example.main()

for (i, lc) in enumerate(lcs):
    if i == 0:
        ret = np.array(np.lib.recfunctions.append_fields(lc.lc, ['phase', 'dL', 'z'], 
                                                         [lc.lc['mjd']-lc.sn['DayMax'], lc.sn['dL'], lc.sn['z']]))
    else:
        paquet = np.array(np.lib.recfunctions.append_fields(lc.lc, ['phase', 'dL', 'z'], 
                                                            [lc.lc['mjd']-lc.sn['DayMax'], lc.sn['dL'], lc.sn['z']]))
        ret = np.append(ret, paquet)
    if i % 100 == 0:
        print 'Data computed for %d supernovae' % i

lambda_grid = np.linspace(2000, 8000, 300)
phase_grid = np.linspace(-35, 70, 8)
base = bspline.BSpline2D(lambda_grid, phase_grid, xorder=4, yoerde=4)

#### MODEL ####
### m_{b, p} = P(\lambda_b / (1+z), p) - 2.5log(X_0) + 30 + \mu - 2.5log( \int T_b(\lambda) \lambda d\lambda) + 2.5log(1+z)
###############

lambdas = 

J = base.eval(
