import numpy as np
from ..green import gauss_inverse # calculate the desired green functions
from ..algebra import dagger,sqrtm
from .smatrix import enlarge_hlist,effective_tridiagonal_hamiltonian
from .smatrix import build_effective_hlist
from ..checkclass import is_iterable

def get_full_green(ht,energy,mode="right",delta=None,ic=0):
    """Build effective Hamiltonian at a certain energy"""
    if delta is None: delta = ht.delta
    # get the selfenergies, using the same coupling as the lead
    selfl = ht.get_selfenergy(energy,delta=delta,lead=0,pristine=True)
    selfr = ht.get_selfenergy(energy,delta=delta,lead=1,pristine=True)
    if ht.block_diagonal:
        ht2 = enlarge_hlist(ht) # get the enlaged hlist with the leads
    # selfenergy of the leads (coupled to another cell of the lead)
        gmatrix = effective_tridiagonal_hamiltonian(ht2.central_intra,
                                        selfl,selfr,
                                        energy=energy,
                                        delta=delta + ht.extra_delta_central)
        test_gauss = False # do the tridiagonal inversion
    else: # not block diagonal
        gmatrix = build_effective_hlist(ht,energy=energy,
                                          delta=delta,selfl=selfl,
                                        selfr=selfr)
    test_gauss = True # do the tridiagonal inversion
    if mode=="left": g = gauss_inverse(gmatrix,0,0,test=test_gauss)
    elif mode=="right": g = gauss_inverse(gmatrix,-1,-1,test=test_gauss)
    elif mode=="central": 
        if ic is None: # provide a list of all the Green's functions
            out = [] # empty list
            for ii in range(len(gmatrix)):
              g = gauss_inverse(gmatrix,i=ii,j=ii,test=test_gauss)
              out.append(g)
            return out
        elif is_iterable(ic): # an iterable  
            out = [] # empty list
            for ii in ic:
              g = gauss_inverse(gmatrix,i=ii+1,j=ii+1,test=test_gauss)
              out.append(g)
            return out
        else: # well implemented way, assume it is a number
          ii = 1+ic
          g = gauss_inverse(gmatrix,i=ii,j=ii,test=test_gauss)
    else: raise
    return g
