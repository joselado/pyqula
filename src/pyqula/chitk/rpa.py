import numpy as np
from .. import algebra

# compute general RPA response function
def chi_AB_RPA(h,V=None,**kwargs):
    """Compute the RPA chi for a hamiltonian"""
    from ..chi import chiAB # get response function
    es,chis = chiAB(h,mode="matrix",**kwargs) # non-interacting response
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    if V is not None: # finite interaction, RPA summation
        chis_rpa = [chi@algebra.inv(iden - V@chi) for chi in chis]
    else: chis_rpa = chis
    return es,np.array(chis_rpa)


def chi_AB_RPA_scf(scf):
    """Return the RPA response function for an SCF object"""
    if len(scf.v)==1: # just the onsite term
        return chi_AB_RPA(scf.hamiltonian,scf.v[(0,0,0)])
    else: raise # not implemented


def spinchi_pm_RPA(h,U=0.,v=[0.,0.,1.],**kwargs):
    """Compute the spin RPA response for a hamiltonian.
     - v is the chosen quantization axis of the ladder operators
     - U is the Hubbard interaction"""
     # v needs to be implemented
    sx = h.get_operator("sx") # spin operator, eigen +-1
    sy = h.get_operator("sy") # spin operator, eigen +-1
    sz = h.get_operator("sz") # spin operator, eigen +-1
    v = np.array(v) # convert to array
    sp = (sx + 1j*sy)/2. # ladder operator
    sm = (sx - 1j*sy)/2. # ladder operator
    from .chi import chiAB # get response function
    es,chis = chiAB(h,A=sp,B=sm,mode="matrix",**kwargs) # non-interacting response functions
    iden = np.identity(chis[0].shape[0],dtype=np.complex128) # identity
    # this factor 1/2 should be here
    chisrpa = [chi@algebra.inv(iden + U/2.*chi) for chi in chis] # RPA summation
    return es,np.array(chisrpa) # return energies and RPA response function




