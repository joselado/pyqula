import numpy as np
from ..spin import sx,sy,sz,bmat
from ..superconductivity import build_eh


def get_si(h,i=1):
    """Return a certain Pauli matrix for the full Hamiltonian"""
    if not h.has_spin: return None # no spin
    if i==1: si = sx # sx matrix
    elif i==2: si = sy # sy matrix
    elif i==3: si = sz # sz matrix
    else: raise # unknown pauli matrix
    if h.has_eh: ndim = h.intra.shape[0]//4 # half the dimension
    else: ndim = h.intra.shape[0]//2 # dimension
    if h.has_spin: # spinful system
      op = [[None for i in range(ndim)] for j in range(ndim)] # initialize
      for i in range(ndim): op[i][i] = si # store matrix
      op = bmat(op) # create matrix
    if h.has_eh: op = build_eh(op,is_sparse=True) # add electron and hole parts
    return op

# define the functions for the three spin components
get_sx = lambda h: get_si(h,i=1) # sx matrix
get_sy = lambda h: get_si(h,i=2) # sy matrix
get_sz = lambda h: get_si(h,i=3) # sz matrix





