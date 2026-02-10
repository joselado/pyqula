import numpy as np
import numba
from numba import jit,prange



@jit(nopython=True)
def chiAB_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta):
    """Compute the response function.
    A and B are expected to be local operators, like Sz in site 0"""
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
    """Compute the full ChiAB matrix, element by element"""
    ni = len(Ais) # number of operators
    nj = len(Bjs) # number of operators
    out = np.zeros((ni,nj,len(energies)),dtype=np.complex128) # initialize
    for i in prange(ni): # loop over rows of the matrix
        for j in prange(nj): # loop over columns of the matrix
            out[i,j,:] = chiAB_jit(ws1,es1,ws2,es2,energies,Ais[i],
                    Bjs[j],temp,delta) # compute the response
    return np.transpose(out,(2,0,1)) # return transposed, first energy, then ij


@jit(nopython=True)
def chiAB_full_matrix_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta):
    """Compute the full matrix of the response function.
    A and B are full operators, for example the full Sz operator"""
    ### WARNING ###
    # this function assumes that the Hamiltonian is spinful and non-Nambu
    ### WARNING ###
    # initialize the result
    ni = ws1.shape[0]//2 # number of operators, dimension of matrix/2
    nj = ni # number of operators
    outm = np.zeros((ni,nj,len(omegas)),dtype=np.complex128) # initialize
    # now do the loop over wavefunctions
    cutoff = delta/100 # cutoff for occupation difference
    beta = 1./T # thermal broadening
    n = len(ws1) # number of wavefunctions
    Aws2 = (A@ws2.T).T # compute all the applied wavefunctions
    Bws1 = (B@ws1.T).T # compute all the applied wavefunctions 
    occs1 = (-np.tanh(beta*es1) + 1.)/2. # occupations
    occs2 = (-np.tanh(beta*es2) + 1.)/2. # occupations
    for i in range(n): # loop over wavefunctions
        oi = occs1[i] # first occupation
        for j in range(n): # loop over wavefunctions
            oj = occs2[j] # second occupation
            fac0 = oi - oj # occupation factor
            if np.abs(fac0)<cutoff: continue # skip contribution if too small
            wci = np.conjugate(ws1[i]) # get this wavefunction
            wcj = np.conjugate(ws2[j]) # get this wavefunction
            Awj = Aws2[j] # get this one
            Bwi = Bws1[i] # get this one
            for ii in range(ni): # loop over rows of the correlator
                for jj in range(nj): # loop over columns of the correlator
                    fac  = np.sum(wci[2*ii:2*ii+2]*Awj[2*ii:2*ii+2]) # add the factor
                    fac *= np.sum(wcj[2*jj:2*jj+2]*Bwi[2*jj:2*jj+2]) # add the factor
                    outm[ii,jj,:] += fac0*fac*(1./(es1[i]-es2[j] - omegas + 1j*delta)) # add contribution
    return outm # return result


@jit(nopython=True,parallel=True)
def chiAB_full_matrix_jit_kmesh(ws1k,es1k,ws2k,es2k,omegas,A,B,T,delta):
    """Parallelize over kpoints"""
    nk = es1k[:,0].shape[0] # number of kpoints
    ni = es1k[0,:].shape[0]//2 # number of operators, dimension of matrix/2
    nj = ni # number of operators
    out = np.zeros((nk,ni,nj,len(omegas)),dtype=np.complex128) # initialize
    for ik in prange(nk): # parallel loop over kpoints
        ws1 = ws1k[ik,:,:].copy()
        ws2 = ws2k[ik,:,:].copy()
        es1 = es1k[ik,:].copy()
        es2 = es2k[ik,:].copy()
        out[ik,:,:,:] = chiAB_full_matrix_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta)
    return out

import scipy.linalg as lg

def chiAB_matrix_ksum(h,ks,q,omegas,A,B,T,delta):
    """Perform the integral over k, using parallelization
    over kpoints"""
    # this function is currently only implemented for spinful
    # a spinless and Nmabu implementation needs to be done
    if h.has_eh: raise # not implemented for Nambu
    if not h.has_spin: raise # not implemented for spinless
    hk = h.get_hk_gen() # get generator
    n = h.intra.shape[0] # shape
    nk = len(ks) # number of kpoints
    # storages for all kpoints
    es1k = np.zeros((nk,n),dtype=np.float64) # storage
    es2k = np.zeros((nk,n),dtype=np.float64) # storage
    ws1k = np.zeros((nk,n,n),dtype=np.complex128) # storage
    ws2k = np.zeros((nk,n,n),dtype=np.complex128) # storage
    ik = 0 # counter
    for k in ks: # loop over kpoints
        m1 = hk(k) # get Hamiltonian
        es1,ws1 = lg.eigh(m1)
        ws1 = np.array(ws1.T,dtype=np.complex128)
        m2 = hk(k+q) # get Hamiltonian
        es2,ws2 = lg.eigh(m2)
        ws2 = np.array(ws2.T,dtype=np.complex128)
        # now store all
        es1k[ik,:] = es1[:]
        es2k[ik,:] = es2[:]
        ws1k[ik,:,:] = ws1[:,:]
        ws2k[ik,:,:] = ws2[:,:]
        ik += 1
    # and now perform the calculation in parallel
    out = chiAB_full_matrix_jit_kmesh(ws1k,es1k,ws2k,es2k,omegas,A,B,T,delta)
    out = np.transpose(out,(0,3,1,2)) # switch the indexes
    return np.mean(out,axis=0) # average over kpoints








