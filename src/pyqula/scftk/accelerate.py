# variety of tricks to accelerate a mean field calculation
from . import hubbard


def scf_accelerate(scf):
    """Redefine methods to accelerate a mean field calculation"""
    if scf.mode=="Hubbard" and type(self.g)==type(0.2): # Hubbard mean field
        return hubbard.hubbardscf(scf.hamiltonian,U=scf.g,
                mix=scf.mixing,mf=scf.mf)
    return scf



