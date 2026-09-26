# woorkaround for non Hermitian Hamiltonians

import numpy as np
from ..dos import get_dos_general

def get_dos(self,mode="ED",use_kpm=False,biorthogonal=False,**kwargs):
    """Density of states of a non-Hermitian Hamiltonian.

    Only exact diagonalization is available here: the KPM and adaptive
    modes of the Hermitian get_dos_general both assume a real spectrum.
    The mode used to be overwritten with "ED" unconditionally, so every
    mode string -- a typo included -- silently returned the ED result.

    biorthogonal: which of the two spectral functions of a non-Hermitian
    Hamiltonian is summed over the Brillouin zone, as in get_kdos_bands.
    False (the default) broadens each state at Re E with a Lorentzian of
    width delta, weighted by its right eigenvector if there is an
    operator; True is -Im Tr[O G]/pi with G = (w + i delta - H)^-1, which
    broadens each state by its lifetime as well (see dos_biorthogonal)."""
    if use_kpm or mode!="ED":
        raise NotImplementedError("the non-Hermitian DOS only implements "
                "mode='ED'; '"+str(mode)+"' assumes a real spectrum "
                "(the KPM and adaptive modes expand in Chebyshev "
                "polynomials of a Hermitian matrix)")
    if biorthogonal: return dos_biorthogonal(self,**kwargs)
    return get_dos_general(self,mode="ED",**kwargs)



def dos_biorthogonal(h,energies=np.linspace(-4.0,4.0,400),nk=100,
        delta=None,ks=None,random=False,operator=None,write=True,
        chunk=2000,**kwargs):
    """The Brillouin-zone average of the biorthogonal spectral function,
    -Im Tr[O G(k,w)]/pi with G = (w + i delta - H_k)^-1, from the
    eigenstates: sum_n w_n/(w + i delta - E_n) over every state of every
    kpoint, with w_n = <L_n|O|R_n>/<L_n|R_n> (one without an operator) and
    E_n complex. The k-mesh, the default broadening and the normalization
    are the ones of the right-eigenvector DOS (dos.dos_kmesh), so that the
    two differ only in the spectral function they sum"""
    from ..klist import kmesh
    from ..dos import write_dos
    if kwargs.get("eigmode","complex")!="complex":
        raise ValueError("the biorthogonal spectral function needs the "
                "complex eigenvalues, whose imaginary part broadens each "
                "state, so it takes no eigmode='"+str(kwargs["eigmode"])+"'")
    if ks is None: ks = kmesh(h.dimensionality,nk=nk)
    if delta is None: delta = 5./nk # as dos_kmesh
    if random: ks = [np.random.random(3) for k in ks]
    out = h.get_bands(kpath=ks,operator=operator,biorthogonal=True,
            write=False,**kwargs)
    es = np.asarray(out[1]) # complex eigenvalues of every kpoint
    ws = np.asarray(out[2]) if len(out)>2 else np.ones(len(es)) # weights
    energies = np.array(energies,dtype=float)
    ys = np.zeros(len(energies)) # accumulated -Im of the poles
    for i0 in range(0,len(es),chunk): # a block of poles at a time
        g = ws[None,i0:i0+chunk]/(energies[:,None] + 1j*delta
                - es[None,i0:i0+chunk])
        ys += -np.sum(g,axis=1).imag
    ys = ys/np.pi/len(ks) # per kpoint, as dos_kmesh normalizes
    if write: write_dos(energies,ys) # in a file
    return (energies,ys)
