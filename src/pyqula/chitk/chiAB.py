import numpy as np
import numba
from numba import jit,prange



@jit(nopython=True)
def chiAB_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta):
    """Compute the response function"""
    cutoff = delta/100 # cutoff for occupation difference
    beta = 1./T # thermal broadening
    out  = np.zeros(omegas.shape[0],dtype=np.complex128) # initialize
    n = len(ws1) # number of wavefunctions
    Aws2 = (A@ws2.T).T #[A@w for w in ws2] # compute all matrix elements
    Bws1 = (B@ws1.T).T #[B@w for w in ws1] # compute all matrix elements
    occs1 = (-np.tanh(beta*es1) + 1.)/2. # occupations
    occs2 = (-np.tanh(beta*es2) + 1.)/2. # occupations
    for i in range(n): # loop over wavefunctions
        oi = occs1[i] # first occupation
        for j in range(n): # loop over wavefunctions
            oj = occs2[j] # second occupation
            fac = oi - oj # occupation factor
            if np.abs(fac)<cutoff: continue # skip contribution if too small
            fac *= np.sum(np.conjugate(ws1[i])*Aws2[j]) # add the factor
            fac *= np.sum(np.conjugate(ws2[j])*Bws1[i]) # add the factor
            out = out + fac*(1./(es1[i]-es2[j] - omegas + 1j*delta))
    return out




@jit(nopython=True,parallel=True)
def chiAB_matrix_jit(ws1,es1,ws2,es2,energies,Ais,Bjs,temp,delta):
    """Compute the ChiAB matrix"""
    ni = len(Ais) # number of operators
    nj = len(Bjs) # number of operators
    out = np.zeros((ni,nj,len(energies)),dtype=np.complex128) # initialize
    for i in prange(ni): # loop
        for j in prange(nj): # loop
            out[i,j,:] = chiAB_jit(ws1,es1,ws2,es2,energies,Ais[i],
                    Bjs[j],temp,delta) # compute the response
    return out # return

