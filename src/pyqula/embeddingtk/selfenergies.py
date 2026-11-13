import numpy as np


# routines to create selfenergies for different types of leads

class Selfenergy():
    """Self-energy object"""
    def __init__(self,m=None): # pass
        self.m = m
    def __call__(self,**kwargs):
        return self.m



def metal(H,potential):
    """Return the selfenergy of a metallic lead"""
    # create a selfenergy in a single edge (the left one) to kill magnetism
    hs = H.copy()*0.
    hs.add_onsite(lambda r: np.abs(potential(r))) # absolute value
    selfe = -1j*hs.intra # selfenergy on edge, beware of minus sign
    return selfe


def semimetal(*args,zero=0.):
    """Retarded selfenergy of a lead whose density of states grows as
    |E-zero|, a semimetal: -i Gamma |E-zero|, with Gamma the damping of
    metal(). Like every callable selfenergy it returns the retarded one,
    and embedded.object2selfenergy continues it below the real axis as
    Sigma(E-i0) = Sigma(E+i0)^dagger, which holds whatever the symmetries
    of the Hamiltonian, time reversal included; before, the retarded one
    was used for both signs of delta, and the spectral function of
    Embedded_Hamiltonian.get_ldos came out zero (bug_audit_6.md #13).
    |E-zero| has no analytic continuation off the real axis (it has no
    Kramers-Kronig real part), so it is marked analytic=False and the
    density matrices of the embeddings, contour integrals over complex
    energies, refuse it; the LDOS and the Green's function at real
    energies are right"""
    selfe = metal(*args) # selfenergy of a metal
    def f(energy=0.,**kwargs):
        return selfe*np.abs(energy-zero) # retarded selfenergy of a semimetal
    f.analytic = False # see the docstring
    return f



def get_selfenergy_from_potential(*args,mode="metal",**kwargs):
    if mode=="metal": return metal(*args,**kwargs)
    elif mode=="semimetal": return semimetal(*args,**kwargs)
    else:
        raise ValueError("unknown mode; the accepted ones are 'metal' and "
                "'semimetal'")


