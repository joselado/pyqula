
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
    if kwargs.get("biorthogonal",False):
        # get_bands would return the complex biorthogonal weights, and the
        # Lorentzian sum below keeps their real part at Re E only, which is
        # neither of the two spectral functions
        raise NotImplementedError("the non-Hermitian DOS weighs the states "
                "by their right eigenvectors only; the biorthogonal "
                "spectral function is in h.get_kdos_bands(biorthogonal=True)")
    return get_dos_general(self,mode="ED",**kwargs)

