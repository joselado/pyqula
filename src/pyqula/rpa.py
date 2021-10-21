import numpy as np
from . import algebra

# compute RPA response
def chi_AB_RPA(h,V,**kwargs):
    """Compute the RPA chi for an SCF object"""
#    raise # not finished
    from .chi import chiAB # get response function
    chis = chiAB(h,**kwargs) # get the non-interacting response functions
    iden = np.identity(chis[0].shape[0],dtype=np.complex) # identity
    chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    return np.array(chis_rpa)


def chi_AB_RPA_scf(scf):
    """Return the RPA response function"""
    if len(scf.v)==1: # just the onsite term
        return chi_AB_RPA(scf.hamiltonian,scf.v[(0,0,0)])
    else: raise # not implemented

