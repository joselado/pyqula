
# compute certain order parameters
import numpy as np
from .. import multihopping

def st(h,nk=10,p=1):
    """Compute the singlet order parameter"""
    from . import extract
    if p==1: h = extract.get_singlet_hamiltonian(h)
    elif p==-1: h = extract.get_triplet_hamiltonian(h)
    else: raise
    hk = h.get_hk_gen() # get Bloch Hamiltonian generator
    ks = h.geometry.get_kmesh(nk=nk) # get k-mesh
    def f(k):
        m = hk(k)
        m = m@m
        return np.trace(m).real # return the trace
    return np.mean([f(k) for k in ks])

def singlet(h,**kwargs): return st(h,p=1,**kwargs)

def triplet(h,**kwargs): return st(h,p=-1,**kwargs)

