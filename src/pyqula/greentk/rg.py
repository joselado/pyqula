import numpy as np
from .. import algebra
from numba import jit

use_fortran = False
use_numba = False

def green_renormalization(intra,inter,**kwargs):
    if use_numba:
      return green_renormalization_jit(intra,inter,**kwargs)
    else: 
      return green_renormalization_python(intra,inter,**kwargs)




def green_renormalization_python(intra,inter,energy=0.0,nite=None,
                            info=False,delta=0.001,
                            use_fortran = use_fortran,**kwargs):
    """ Calculates bulk and surface Green function by a renormalization
    algorithm, as described in I. Phys. F: Met. Phys. 15 (1985) 851-858 """
    intra = algebra.todense(intra)
    inter = algebra.todense(inter)
    error = np.abs(delta)*1e-6 # overwrite error
    e = np.matrix(np.identity(intra.shape[0])) * (energy + 1j*delta)
    ite = 0
    alpha = inter.copy()
    beta = algebra.dagger(inter).copy()
    epsilon = intra.copy()
    epsilon_s = intra.copy()
    while True: # implementation of Eq 11
      einv = algebra.inv(e - epsilon) # inverse
      epsilon_s = epsilon_s + alpha @ einv @ beta
      epsilon = epsilon + alpha @ einv @ beta + beta @ einv @ alpha
      alpha = alpha @ einv @ alpha  # new alpha
      beta = beta @ einv @ beta  # new beta
      ite += 1
      # stop conditions
      if not nite is None:
        if ite > nite:  break
      else:
        if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error: break
    if info:
      print("Converged in ",ite,"iterations")
    g_surf = algebra.inv(e - epsilon_s) # surface green function
    g_bulk = algebra.inv(e - epsilon)  # bulk green function
    return g_bulk,g_surf

def green_renormalization_jit(intra,inter,energy=0.0,delta=1e-4,**kwargs):
    intra = algebra.todense(intra)*(1.0+0j)
    inter = algebra.todense(inter)*(1.0+0j)
    g0 = intra*0.0j
    g1 = intra*0.0j
    nite = int(10/delta)
    error = delta*1e-3
    energyz = energy + 1j*delta
    e = np.array(np.identity(intra.shape[0]),dtype=np.complex) * energyz
    return green_renormalization_jit_core(g0,g1,intra,inter,e,nite,
                                                error)


#@jit()
#@jit(nopython=True)
### this seems to not work with numba ###
def green_renormalization_jit_core(g0,g1,intra,inter,e,nite,error):
    ite = 0
    alpha = inter*1.0
    beta = np.conjugate(inter).T*1.0
    epsilon = intra*1.0
    epsilon_s = intra*1.0
    while True: # implementation of Eq 11
      einv = np.linalg.inv(e - epsilon) # inverse
      epsilon_s = epsilon_s + alpha @ einv @ beta
      epsilon = epsilon + alpha * einv @ beta + beta @ einv @ alpha
      alpha = alpha @ einv @ alpha  # new alpha
      beta = beta @ einv @ beta  # new beta
      ite += 1
      # stop conditions
      if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error: break
    g_surf = np.linalg.inv(e - epsilon_s) # surface green function
    g_bulk = np.linalg.inv(e - epsilon)  # bulk green function 
    g0,g1 = g_bulk*1,g_surf*1
    return g0,g1
                     
