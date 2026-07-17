import numpy as np
import numba
from numba import jit,prange
from .. import algebra
from .. import parallel



def chiAB(h,q=None,nk=60,**kwargs):
    """Return the generalized response"""
    if q is not None: # q point is provided
        return chiAB_q(h,q=q,nk=nk,**kwargs)
    else:
        qs = h.geometry.get_kmesh(nk=nk) # get the kmesh
        out = [chiAB_q(h,q=q,nk=nk,
                       **kwargs) for q in qs] # get all the k kpoints
        return out[0][0],np.mean([o[1] for o in out],axis=0)


def chiAB_q(h,energies=np.linspace(-3.0,3.0,100),q=[0.,0.,0.],nk=60,
               pAs = None, pBs=None, # projectors for chi
               delta=0.1,T=None,A=None,B=None,projs=None,
               imode="mesh", # integration mode in momentum space
               ij_mode = "explicit", # loop over elements mode
               mode="matrix" # return object
               ):
    """Compute AB response function
       - energies: energies of the dynamical response
       - q: q-vector of the response
       - nk: number of k-points for the integration
       - delta: imaginary part
       - A: first operator
       - B: second operator
       - projs: local projection operators
       - imode: integration mode
       - ij_mode: loop ove elements mode
       - mode: output to return"""
    temp = T # redefine
    if temp is None: temp = delta # as delta
    hk = h.get_hk_gen() # get generator
    if pAs is None and pBs is None: # assume in put is A,B,projs
        if A is None or B is None:
            A = np.identity(h.intra.shape[0],dtype=np.complex128)
            B = A # initial operator
        else: # generate the operators to be evaluated in the lattice points
            A = h.get_operator(A)
            B = h.get_operator(B)
            A = algebra.todense(A.get_matrix())
            B = algebra.todense(B.get_matrix())
        # generate the projectors
        if projs is None:
            from .. import operators
            projs = [operators.index(h,n=[i]) for i in range(len(h.geometry.r))]
        else:
            ij_mode = "explicit" # do the loop explicitly
        pAs = np.array([pi@A for pi in projs]) # compute these projectors
        pBs = np.array([pi@B for pi in projs]) # compute these projectors
    else: # pAs and pBs provided on input
        ij_mode = "explicit" # do the loop explicitly
        pass
    ### now define the function to integrate
    def getk(k):
        m1 = hk(k) # get Hamiltonian
        es1,ws1 = algebra.eigh(m1)
        ws1 = np.array(ws1.T,dtype=np.complex128)
        m2 = hk(k+q) # get Hamiltonian
        es2,ws2 = algebra.eigh(m2)
        ws2 = np.array(ws2.T,dtype=np.complex128)
        def getAB(Ai,Bj): # compute for a single operator
            return chiAB_jit(ws1,es1,ws2,es2,energies,Ai,Bj,temp,delta)
        if mode=="matrix": # return a matrix
            # simplest implementation, not optimal
#            out = np.array([[getAB(pA,pB) for pA in pAs] for pB in pBs])
            # parallelized (over matrix elements) implementation
            parallel.set_num_threads() # set the number of threads
            out = chiAB_matrix(ws1,es1,ws2,es2,energies,pAs,pBs,temp,delta)
            return out # return array of matrices
        elif mode=="trace": # return the trace
            out = np.array([getAB(pi@A,pi@B) for pi in projs])
            return np.mean(out,axis=0) # sum over the first axis
        elif mode=="diagonal": # return the diagonal elements
            out = np.array([getAB(pi@A,pi@B) for pi in projs])
            return np.transpose(out,(1,0)) # return, first energy, then i
        else: raise # not implemented
    ks = h.geometry.get_kmesh(nk=nk) # get the kmesh
    # call in parallel
    if imode=="mesh": # do a mesh
        if ij_mode=="accelerated": # (maybe?) accelerated function
            parallel.set_num_threads() # set the number of threads
            out = chiAB_matrix_ksum(h,ks,q,energies,A,B,temp,delta)
        elif ij_mode=="explicit": # explicit function, this is preferred
            out = [getk(k) for k in ks] # call
            out = np.mean(out,axis=0) # sum over kpoints
        else: raise # not implemented
    elif imode=="adaptive": # do a mesh
        from . import integration
        if h.dimensionality==0: out = getk([0.]) # single point
        elif h.dimensionality==1:
            out = integration.integrate_matrix(lambda k: getk([k]),xlim=[0.,1.])
        elif h.dimensionality==2: # not implemented
            out = integration.integrate_matrix_2D(getk,
                    xlim=[0.,1.],ylim=[0.,1.])
        else: raise
    else: raise
    return energies,out





@jit(nopython=True)
def chiAB_jit(ws1,es1,ws2,es2,omegas,A,B,T,delta):
    """Compute the response function for a single (A,B) operator pair.
    A and B are expected to be local operators, like Sz in site 0"""
    cutoff = delta/100 # cutoff for occupation difference
    beta = 1./T # thermal broadening
    out  = np.zeros(omegas.shape[0],dtype=np.complex128) # initialize
    n = len(ws1) # number of wavefunctions
    Aws2 = (A@ws2.T).T #[A@w for w in ws2] # compute all matrix elements
    Bws1 = (B@ws1.T).T #[B@w for w in ws1] # compute all matrix elements
    occs1 = 1./(1. + np.exp(beta*es1)) # occupations
    occs2 = 1./(1. + np.exp(beta*es2)) # occupations
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
def chiAB_matrix(ws1,es1,ws2,es2,energies,Ais,Bjs,temp,delta):
    """Compute the full ChiAB matrix.
    A naive implementation calls chiAB_jit once per (i,j) pair of
    operators, which recomputes the O(n^3) Ai@ws2.T/Bj@ws1.T transforms
    from scratch for every pair even though the former only depends on i
    and the latter only on j. Precomputing those transforms once per row
    and once per column operator (instead of once per pair) turns this
    from an O(ni*nj*n^3) computation into O((ni+nj)*n^3 + ni*nj*n^2)."""
    ni = len(Ais) # number of row operators
    nj = len(Bjs) # number of column operators
    n = len(ws1) # number of wavefunctions
    cutoff = delta/100 # cutoff for occupation difference
    beta = 1./temp # thermal broadening
    occs1 = 1./(1. + np.exp(beta*es1)) # occupations
    occs2 = 1./(1. + np.exp(beta*es2)) # occupations
    cws1 = np.conjugate(ws1)
    cws2 = np.conjugate(ws2)
    ws1T = ws1.T
    ws2T = ws2.T
    MA = np.zeros((ni,n,n),dtype=np.complex128) # <a|Ai|b>, per row operator
    for i in prange(ni):
        MA[i] = cws1@(Ais[i]@ws2T)
    MB = np.zeros((nj,n,n),dtype=np.complex128) # <b|Bj|a>, per column operator
    for j in prange(nj):
        MB[j] = cws2@(Bjs[j]@ws1T)
    out = np.zeros((ni,nj,len(energies)),dtype=np.complex128) # initialize
    for i in prange(ni): # loop over rows of the matrix
        for a in range(n): # loop over wavefunctions of ws1
            oa = occs1[a] # first occupation
            for b in range(n): # loop over wavefunctions of ws2
                fac0 = oa - occs2[b] # occupation factor
                if np.abs(fac0)<cutoff: continue # skip contribution if too small
                denom = fac0*(1./(es1[a]-es2[b] - energies + 1j*delta))
                MAiab = MA[i,a,b]
                for j in range(nj): # loop over columns of the matrix
                    out[i,j,:] += MAiab*MB[j,b,a]*denom
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








