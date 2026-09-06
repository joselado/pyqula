import numpy as np
from .. import algebra

dagger = algebra.dagger
inv = algebra.inv # inverse
# Dyson equation solvers for Green's functions


def dysonNNN(ons,t1,t2,only_bulk=False,hs=None,energy=0.,delta=0.01,
             **kwargs):
    """Worksround to do RG with NNN"""
    from scipy.sparse import csc_matrix
    ons = csc_matrix(ons)
    t1 = csc_matrix(t1)
    t2 = csc_matrix(t2)
    ons_S = [[ons,t1],[dagger(t1),ons]] # supercell onsite
    # should this be the other way around (?)
    #hop_S = [[t2,t1],[t1*0.,t2]] # supercell hopping
    hop_S = [[t2,t1*0.],[t1,t2]] # supercell hopping
    from .. import algebra
    ons_S = algebra.bmat(ons_S) # as matrices
    hop_S = algebra.bmat(hop_S) # as matrices
    # perform the RG algorithm
    from .rg import green_renormalization
    gb_S,gs_S = green_renormalization(ons_S,hop_S,energy=energy,delta=delta,
                                      **kwargs)
    n = ons.shape[0] # size of the system
    if hs is not None: # surface onsite matrix provided
        gs_S = surface_onsite_dyson(gs_S,ons_S,hop_S,hs,energy,delta,n)
    gb = gb_S[0:n,0:n] # bulk Green function
    gs = gs_S[0:n,0:n] # bulk Green function
    if only_bulk: return gb
    else: return gb,gs


def surface_onsite_dyson(gs_S,ons_S,hop_S,hs,energy,delta,n):
    """Recompute the surface Green's function of a semi-infinite chain
    after replacing the outermost cell's onsite matrix by hs.

    The decimation solves gs = (ez - ons - hop gs hop^dag)^(-1), so
    hop gs hop^dag is the selfenergy of everything attached below the
    surface cell -- unchanged by what that one cell's onsite is. Putting
    hs in its place therefore just re-solves the same equation, which is
    exactly what green_kchain_NN does for the nearest-neighbor case. Here
    the chain is a chain of supercells, and only the first of the n-sized
    sub-cells is the actual surface, so hs replaces that block alone."""
    from .. import algebra
    ons_S = algebra.todense(ons_S) # dense, a block is about to be replaced
    hop_S = algebra.todense(hop_S)
    gs_S = algebra.todense(gs_S)
    N = ons_S.shape[0] # dimension of the supercell
    ez = (energy+1j*delta)*np.identity(N) # energy
    sigma = hop_S@gs_S@dagger(hop_S) # selfenergy of the rest of the chain
    ons2_S = np.array(ons_S,dtype=np.complex128) # copy to modify
    ons2_S[0:n,0:n] = algebra.todense(hs) # replace the surface onsite
    return inv(ez - ons2_S - sigma) # Dyson equation






def dysonLR(hops0,only_bulk=False,hs=None,energy=0.,delta=0.01,**kwargs):
    """Worksround to do RG with NNN"""
    from scipy.sparse import csc_matrix
    hops = [csc_matrix(m) for m in hops0] # to sparse
    hopsd = [csc_matrix(dagger(m)) for m in hops0] # to sparse
    zero = hops[0]*0. # zero
    ns = len(hops)-1 # number of supercells
    ons_S = [[zero.copy() for i in range(ns)] for j in range(ns)] # empty matrix
    hop_S = [[zero.copy() for i in range(ns)] for j in range(ns)] # empty matrix
    for i in range(ns):
        for j in range(ns):
            ij = i - j
            # intracell coupling
            if ij>0: m = hops[ij]
            elif ij<0: m = hopsd[abs(ij)]
            elif ij==0: m = hops[0] # intra
            ons_S[j][i] = m.copy() # store
            ij = j - i + ns
            # intercell coupling
            if 0<ij<=ns: m = hops[ij]
            else: m = zero.copy()
            hop_S[i][j] = m.copy() # store
    # perform the RG algorithm
    from .. import algebra
    ons_S = algebra.bmat(ons_S) # as matrices
    hop_S = algebra.bmat(hop_S) # as matrices
    from .rg import green_renormalization
    gb_S,gs_S = green_renormalization(ons_S,hop_S,energy=energy,delta=delta,
                                      **kwargs)
    n = hops[0].shape[0] # size of the system
    if hs is not None: # surface onsite matrix provided
        gs_S = surface_onsite_dyson(gs_S,ons_S,hop_S,hs,energy,delta,n)
    gb = gb_S[0:n,0:n] # bulk Green function
    gs = gs_S[0:n,0:n] # bulk Green function
    if only_bulk: return gb
    else: return gb,gs





