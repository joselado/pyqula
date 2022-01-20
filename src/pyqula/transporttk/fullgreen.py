import numpy as np
from ..green import gauss_inverse # calculate the desired green functions
from ..algebra import dagger,sqrtm
from .smatrix import enlarge_hlist,effective_tridiagonal_hamiltonian


def get_full_green(ht,energy,mode="right"):
    """Build effective Hamiltonian at a certain energy"""
    delta = ht.delta
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
  #    print(selfr)
    else: # not block diagonal
        gmatrix = build_effective_hlist(ht,energy=energy,
                                          delta=delta,selfl=selfl,
                                        selfr=selfr)
    test_gauss = True # do the tridiagonal inversion
    if mode=="left": g = gauss_inverse(gmatrix,0,0,test=test_gauss)
    elif mode=="right": g = gauss_inverse(gmatrix,-1,-1,test=test_gauss)
    elif mode=="central": g = gauss_inverse(gmatrix,1,1,test=test_gauss)
    else: raise
    return g
