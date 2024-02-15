import numpy as np
from .. import algebra

dagger = algebra.dagger
inv = algebra.inv # inverse
# Dyson equation solvers for Green's functions

def dysonNNN_surface(ons,t1,t2,nit=100,energy=0.,delta=1e-4,**kwargs):
    """Dyson equation solver including onsite, t1 and t2"""
    n = ons.shape[0] # size of the system
    iden = np.identity(ons.shape[0],dtype=np.complex_) # identity
    em = iden*(energy+1j*delta) # imaginary energy matrix
    self1 = em*0. # initialize
    self2 = em*0. # initialize
    g1 = inv(em-ons) # initialize GF
    g2 = em*0. # initialize
    self02 = t1@inv(em-ons)@dagger(t1) # self energy of new block on 2
    for i in range(nit): # loop over iterations
        self1 = dagger(t1)@g1@t1 # compute self1 for this iteration
        self2 = dagger(t2)@g2@t2 # compute self2 for this iteration
        g2 = inv(inv(g1) - self02) # recompute for the next iteration
        g1 = inv(em - ons - self1 - self2) # Dyson equation in this iteration
    # one more iteration
    self1 = dagger(t1)@g1@t1 # compute self1 for this iteration
    self2 = dagger(t2)@g2@t2 # compute self2 for this iteration
    g1 = inv(em - ons - self1 - self2) # Dyson equation in this iteration
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
    self2l = t2@g2r@dagger(t2) # compute self2l
    gb = inv(em - ons - self1r -self1l - self2r -self2l) # Dyson equation
    return gb # return bulk Green's function


def dysonNNN(ons,t1,t2,reverse=False,only_bulk=True,**kwargs):
    """Return bulk and edge green functions with the Dyson equation"""
    if reverse:
        t1 = dagger(t1)
        t2 = dagger(t2)
    gs = dysonNNN_surface(ons,t1,t2,**kwargs)[0] # surface
    gb = dysonNNN_bulk(ons,t1,t2,**kwargs) # bulk
    if only_bulk: return gb
    else: return gb,gs



# this is not fully checked yet

