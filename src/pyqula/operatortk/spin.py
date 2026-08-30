import numpy as np
from ..spin import sx,sy,sz,bmat
from ..superconductivity import build_eh


def get_si(h,i=1):
    """Return a certain Pauli matrix for the full Hamiltonian"""
    if not h.has_spin:
        # this used to return None, which the observables (get_vev,
        # get_bands, get_dos...) understand as "no operator", i.e. the
        # identity -- so a spin-projected quantity came back silently
        # equal to the unprojected one
        raise ValueError("a spin operator (sx/sy/sz) needs the spin degree "
          +"of freedom, but this Hamiltonian has has_spin=False. Build it "
          +"with g.get_hamiltonian(has_spin=True)")
    if i==1: si = sx # sx matrix
    elif i==2: si = sy # sy matrix
    elif i==3: si = sz # sz matrix
    else: raise ValueError("unknown Pauli matrix index "+str(i)
            +", expected 1 (x), 2 (y) or 3 (z)")
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





