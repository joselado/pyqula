import numpy as np
from .. import algebra

dagger = algebra.dagger
inv = algebra.inv # inverse
# Dyson equation solvers for Green's functions


def dysonNNN(ons,t1,t2,only_bulk=False,**kwargs):
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
    gb_S,gs_S = green_renormalization(ons_S,hop_S,**kwargs)
    n = ons.shape[0] # size of the system
    gb = gb_S[0:n,0:n] # bulk Green function
    gs = gs_S[0:n,0:n] # bulk Green function
    if only_bulk: return gb
    else: return gb,gs






def dysonLR(hops0,only_bulk=False,**kwargs):
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
    gb_S,gs_S = green_renormalization(ons_S,hop_S,**kwargs)
    n = hops[0].shape[0] # size of the system
    gb = gb_S[0:n,0:n] # bulk Green function
    gs = gs_S[0:n,0:n] # bulk Green function
    if only_bulk: return gb
    else: return gb,gs





