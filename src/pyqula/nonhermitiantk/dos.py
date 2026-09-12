
# woorkaround for non Hermitian Hamiltonians

from ..dos import get_dos_general

def get_dos(self,mode="ED",use_kpm=False,**kwargs):
    """Density of states of a non-Hermitian Hamiltonian.

    Only exact diagonalization is available here: the KPM and adaptive
    modes of the Hermitian get_dos_general both assume a real spectrum.
    The mode used to be overwritten with "ED" unconditionally, so every
    mode string -- a typo included -- silently returned the ED result."""
    if use_kpm or mode!="ED":
        raise NotImplementedError("the non-Hermitian DOS only implements "
                "mode='ED'; '"+str(mode)+"' assumes a real spectrum "
                "(the KPM and adaptive modes expand in Chebyshev "
                "polynomials of a Hermitian matrix)")
    return get_dos_general(self,mode="ED",**kwargs)

