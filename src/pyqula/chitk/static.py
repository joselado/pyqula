
# routines to compute static charge-charge correlators
from numba import jit
from .. import algebra
import numpy as np


def chargechi(h,i=0,j=0):
    """Compute the charge correlator for a Hamiltonian"""
    if h.has_eh: 
        print("Hamiltonians with eh not implemented")
        raise # not implemented
    if not h.has_spin:
        return single_chargechi(h,i=i,j=j)
    else:
        up = single_chargechi(h,i=2*i,j=2*j) 
        dn = single_chargechi(h,i=2*i+1,j=2*j+1) 
        return up + dn # return both contributions


def single_chargechi(h,i=0,j=0,temp=1e-7):
    """Compute charge response function for a single orbital"""
    if h.dimensionality!=0: raise
    hk = h.get_hk_gen() # get generator
    m = hk(0) # get Hamiltonian
    esh,ws = algebra.eigh(m) # diagonalize
    ws = np.transpose(ws) # transpose wavefunctions
    if i<0: raise # sanity check
    if j<0: raise # sanity check
    return elementchi(ws,esh,ws,esh,i,j,temp)



@jit(nopython=True)
def elementchi(ws1,es1,ws2,es2,ii,jj,T):
    """Compute the response function"""
    out = 0. # initialize
    n = len(ws1) # number of wavefunctions
    for i in range(n): # first loop over states
      oi = es1[i]<0.0 # first occupation
      for j in range(n): # second loop over states
          oj = es2[j]<0.0 # second occupation
          fac = ws1[i][ii]*ws2[j][ii] # add the factor
          fac *= np.conjugate(ws1[i][jj]*ws2[j][jj]) # add the factor
          # probably this should be written better
          fac *= abs(oi - oj)/2. # occupation factor
          out = out + fac
    return out



