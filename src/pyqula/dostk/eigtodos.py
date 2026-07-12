import numpy as np
from numba import jit,njit,prange


def calculate_dos(es,xs,d,w=None,parallel=True):
    """COmpute DOS, es are the eigenenergies, xs, the frequency grid"""
    if w is None: w = np.zeros(len(es)) + 1.0 # initialize
    else: w = w.real # make it real just in case
    es = np.array(es).real
    xs = np.array(xs).real
    d = np.array(d).real
    if parallel:  ys = calculate_dos_jit(es,xs,d,w) # compute
    else: ys = calculate_dos_jit_serial(es,xs,d,w) # compute
    return ys
  
@jit(nopython=True)
def calculate_dos_jit_serial(es,xs,d,w):
      ys = xs*0.
      for i in range(len(es)): # loop over energies
          e = es[i]
          iw = w[i]
          de = np.abs(xs - e) # E - Ei
          de = d/(d*d + de*de) # 1/(delta^2 + (E-Ei)^2)
          ys += de*iw # times the weight
      return ys

# new version, probably faster (but should be checked)
@njit(parallel=True, fastmath=True, cache=True)
def calculate_dos_jit(es, xs, d, w):
    n_x = xs.size
    n_e = es.size
    d2 = d * d
    ys = np.zeros(n_x, dtype=np.float64)
    for j in prange(n_x):          # parallel over output grid points
        x = xs[j]
        s = 0.0
        for i in range(n_e):
            de = x - es[i]  
            s += w[i] * (d / (d2 + de * de))
        ys[j] = s
    return ys
