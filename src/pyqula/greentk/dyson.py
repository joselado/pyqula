import numpy as np
from .. import algebra

dagger = algebra.dagger
inv = algebra.inv # inverse
# Dyson equation solvers for Green's functions

def dysonNNN_surface(ons,t1,t2,nit=100,energy=0.,delta=1e-4,**kwargs):
    """Dyson equation solver including onsite, t1 and t2"""
    print(ons,t1,t2) 
    n = ons.shape[0] # size of the system
    iden = np.identity(ons.shape[0],dtype=np.complex_) # identity
    em = iden*(energy+1j*delta) # imaginary energy matrix
    self1 = em*0. # initialize
    self2 = em*0. # initialize
    g1 = em*0.  # initialize GF
    g2 = em*0. # initialize
    self_last = t1@inv(em-ons)@dagger(t1) # self energy of new block on 2
#    self_last = dagger(t1)@inv(em-ons)@t1 # self energy of new block on 2
    for i in range(nit): # loop over iterations
        self1_old = self1*1. # store 
        self2_old = self2*1. # store
        self1 = dagger(t1)@g1@t1 # compute self1 for this iteration
        self2 = dagger(t2)@g2@t2 # compute self2 for this iteration
        g2 = inv(em - ons - self_last - self1_old - self2_old) # prelast GF
   #     g2 = inv(em - ons - self1 - self2) # prelast GF
        g1 = inv(em - ons - self1 - self2) # last GF 
    return g1,g2


def dysonNNN_bulk(ons,t1,t2,energy=0.,delta=1e-4,**kwargs):
    """Compute the bulk green function using the DYson equation"""
    # left and right green functions
    g1r,g2r = dysonNNN_surface(ons,t1,t2,energy=energy,delta=delta,**kwargs)
    g1l,g2l = dysonNNN_surface(ons,dagger(t1),dagger(t2),
            energy=energy,delta=delta,**kwargs)
    # left and right selfenergies
    n = ons.shape[0] # size of the system
    iden = np.identity(ons.shape[0],dtype=np.complex_) # identity
    em = iden*(energy+1j*delta) # imaginary energy matrix
    self1r = dagger(t1)@g1r@t1 # compute self1r
    self2r = dagger(t2)@g2r@t2 # compute self2r
    self1l = t1@g1l@dagger(t1) # compute self1l
    self2l = t2@g2l@dagger(t2) # compute self2l
    gb = inv(em - ons - self1r -self1l - self2r -self2l) # Dyson equation
    return gb # return bulk Green's function


def dysonNNN_buggy(ons,t1,t2,reverse=False,only_bulk=False,**kwargs):
    """Return bulk and edge green functions with the Dyson equation"""
    raise # something is not ok
    if reverse:
        t1 = dagger(t1)
        t2 = dagger(t2)
    gs = dysonNNN_surface(ons,t1,t2,**kwargs)[0] # surface
    gb = dysonNNN_bulk(ons,t1,t2,**kwargs) # bulk
    if only_bulk: return gb
    else: return gb,gs




def dysonNNN(ons,t1,t2,only_bulk=False,**kwargs):
    """Worksround to do RG with NNN"""
    from scipy.sparse import csc_matrix
    ons = csc_matrix(ons)
    t1 = csc_matrix(t1)
    t2 = csc_matrix(t2)
    ons_S = [[ons,t1],[dagger(t1),ons]] # supercell onsite
    hop_S = [[t2,t1*0.],[t1,t2]] # supercell hopping
    from .. import algebra
    ons_S = algebra.bmat(ons_S) # as matrices
    hop_S = algebra.bmat(hop_S).T # as matrices
    # perform the RG algorithm
    from .rg import green_renormalization
    gb_S,gs_S = green_renormalization(ons_S,hop_S,**kwargs)
    n = ons.shape[0] # size of the system
    gb = gb_S[0:n,0:n] # bulk Green function
    gs = gs_S[0:n,0:n] # bulk Green function
    if only_bulk: return gb
    else: return gb,gs


# this is not fully checked yet

