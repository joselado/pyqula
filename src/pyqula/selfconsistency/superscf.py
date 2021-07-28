import numpy as np
from numba import jit



def get_mf_anomalous(v,dm):
    """Compute the anomalous part of the mean-field,
    input in the anomalous density matrix in Nambu basis"""
    zero = dm[(0,0,0)]*0. # zero
    mf = dict()
    for d in v: mf[d] = zero.copy()  # initialize
    for d in v: # loop over directions
        d2 = tuple(-np.array(d))
        m = anomalous_term_ij(2*v[d],dm[d2]) # get matrix
        mf[d] = mf[d] + m # add normal term
    return mf



def anomalous_term_ij(v,dm):
    """Return the anomalous term of the mean-field,
    assuming a nmabu basis"""
    # we will assume that v contains up,down in alternating order
    n = dm.shape[0] # number of spinless sites
    out = np.zeros((n,n),dtype=np.complex)
    return anomalous_term_ij_jit(v,dm,out)



# this anomalous term enforces even superconductivity

@jit(nopython=True)
def anomalous_term_ij_jit(v,dm,out):
    ns = len(dm)//2 # number of spinless sites
    for i in range(ns): # loop over spinless sites
        for j in range(ns): # loop over spinless sites
          out[2*i,2*j] = v[2*i,2*j+1]*dm[2*j,2*i]  # down,up
          out[2*i,2*j+1] = v[2*i,2*j]*dm[2*j+1,2*i]  # up,up
          out[2*i+1,2*j+1] = v[2*i+1,2*j]*dm[2*j+1,2*i+1]  # up,down
          out[2*i+1,2*j] = v[2*i+1,2*j+1]*dm[2*j,2*i+1]  # down,down
    return out



def enforce_eh_symmetry_anomalous(d01):
    """Enforce electron-hole symmetry in the two sectors"""
    d01 = enforce_eh_symmetry_anomalous_sector(d01)
    d10 = enforce_eh_from_sector(d01)
    return d01,d10



def enforce_eh_from_sector(d):
    """Given one sector of the Hamiltonian, return the other one"""
    out = dict() # dictionary
    for key in d:
        m = d[key] # one key
        o01 = m*0.0
        key2 = tuple(-np.array(key)) # the opposite
        out[key2] = enforce_eh_from_sector_jit(m,o01)
    return out # return


@jit(nopython=True)
def enforce_eh_from_sector_jit(d,o):
    """Given the ee sector, return the hh sector"""
    return np.conjugate(d.T) # hermitian conjugate
#    ns = len(d)//2 # number of sites
#    for i in range(ns): # loop
#        for j in range(ns): # loop
#            o[2*i+1,2*j+1] = np.conjugate(d[2*j+1,2*i+1])      # ud
#            o[2*i,2*j] = np.conjugate(d[2*j,2*i])      # du
#            o[2*i,2*j+1] = np.conjugate(d[2*j+1,2*i])  # uu
#            o[2*i+1,2*j] = np.conjugate(d[2*j,2*i+1])  # dd
#    return o


def enforce_eh_symmetry_anomalous_sector(d01):
    """Enforce electron-hole symmetry in one of the sectors"""
    out01 = dict()
    for key in d01: # loop over keys
        key2 = tuple(-np.array(key)) # minus one
        o01 = d01[key]*0.0
        o01 = enforce_eh_symmetry_anomalous_jit(d01[key],d01[key2],o01)
        out01[key] = o01
    return out01



@jit(nopython=True)
def enforce_eh_symmetry_anomalous_jit(d01,d10,o01):
    """Enforce electron-hole symmetry"""
    ns = len(d01)//2 # number of spinless sites
    for i in range(ns): # loop
        for j in range(ns): # loop
            # enforce the up|up sector
            o01[2*i,2*j+1] = d01[2*i,2*j+1] - d10[2*j,2*i+1]
            # enforce the down|down sector1
            o01[2*i+1,2*j] = d01[2*i+1,2*j] - d10[2*j+1,2*i]
            # enforce the up|down sector1 (beware of the minus sign)
            o01[2*i,2*j] = d01[2*i,2*j] + d10[2*j+1,2*i+1]
            o01[2*i+1,2*j+1] = d01[2*i+1,2*j+1] + d10[2*j,2*i]
    return o01/2.

