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
        ud = single_chargechi(h,i=2*i+1,j=2*j) 
        du = single_chargechi(h,i=2*i,j=2*j+1) 
        return up + dn + ud + du # return both contributions



def szchi(h,i=0,j=0):
    """Compute the charge correlator for a Hamiltonian"""
    if h.has_eh:
        print("Hamiltonians with eh not implemented")
        raise # not implemented
    if not h.has_spin:
        return single_chargechi(h,i=i,j=j)
    else:
        up = single_chargechi(h,i=2*i,j=2*j)
        dn = single_chargechi(h,i=2*i+1,j=2*j+1)
        ud = single_chargechi(h,i=2*i,j=2*j+1) 
        du = single_chargechi(h,i=2*i+1,j=2*j) 
        return (up + dn - ud -du)/4. # return both contributions


def sxchi(H,**kwargs):
    """Compute magnetic response in the x axis"""
    from .. import rotate_spin
    vector = [0.,1.,0.] # rotation vector
    H = H.copy() # make a copy
    phi = 1.0/2.
    rotate_spin.hamiltonian_spin_rotation(H,vector=vector,angle=phi)
    return szchi(H,**kwargs)




def sychi(H,**kwargs):
    """Compute magnetic response in the y axis"""
    from .. import rotate_spin
    vector = [1.,0.,0.] # rotation vector
    H = H.copy() # make a copy
    phi = 1.0/2.
    rotate_spin.hamiltonian_spin_rotation(H,vector=vector,angle=phi)
    return szchi(H,**kwargs)





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
    out = 0j # initialize
    n = len(ws1) # number of wavefunctions
    for i in range(n): # first loop over states
      oi = (-np.tanh(es1[i]/T) + 1.0)/2. # first occupation
      for j in range(n): # second loop over states
          oj = (-np.tanh(es1[j]/T) + 1.0)/2. # first occupation
          fac = np.conjugate(ws1[i][ii])*ws2[j][ii] # add the factor
          fac *= ws1[i][jj]*np.conjugate(ws2[j][jj]) # add the factor
          # probably this should be written better
          fac *= oi*(1.- oj) # occupation factor
#          fac *= abs(oi - oj)/2. # occupation factor
          out = out + fac
    return out



