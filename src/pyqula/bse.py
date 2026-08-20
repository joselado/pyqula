"""Bethe-Salpeter equation (excitons) on top of a mean-field Hamiltonian.

Public entry point composing the bsetk/ internals, in the same way chi.py
composes chitk/. The physics: an exciton is a bound electron-hole pair,
which the independent-particle spectrum of a mean-field Hamiltonian cannot
describe -- a mean field only ever gives transitions at e_c(k+Q)-e_v(k).
The BSE restores the electron-hole interaction by diagonalizing the
two-particle problem in the basis of those transitions,

  |X>_Q = sum_{vck} A_{vc}(k) c^dag_{c,k+Q} c_{v,k} |MF>

so that the exciton energies come out below the gap by their binding
energy, and the amplitudes A_{vc}(k) say which part of the Brillouin zone
the exciton is built from.

The formalism is the localized-orbital ("point-like orbitals") BSE of the
Xatu code, arXiv:2307.01572, extended here to the full (non-Tamm-Dancoff)
problem so that switching the direct term off recovers the RPA exactly and
can be cross-checked against chitk/rpa.py.

Because the interaction is read from the mean field itself (h.V) by
default, this is time-dependent Hartree-Fock on top of Hartree-Fock: the
same interaction generates the Fock self-energy already inside h and the
BSE kernel, so nothing is double counted.
"""

from .bsetk.solve import BSE
from .bsetk.bands import exciton_bands


def get_bse(h,**kwargs):
    """Return the solved BSE object for a Hamiltonian, see BSE"""
    return BSE(h,**kwargs)


def exciton_energies(h,n=None,**kwargs):
    """Return the exciton energies of a mean-field Hamiltonian"""
    return get_bse(h,**kwargs).get_energies(n=n)


def exciton_states(h,n=None,**kwargs):
    """Return (energies,amplitudes) of the excitons of a mean-field
    Hamiltonian. amplitudes[i] holds the electron-hole amplitudes
    A_{vc}(k) of exciton i, indexed by the flattened pair index whose
    (ik,iv,ic) meaning is in the BSE object's pairs.labels"""
    out = get_bse(h,**kwargs)
    es = out.get_energies(n=n)
    ws = out.amplitudes if n is None else out.amplitudes[0:n]
    return es,ws


def exciton_binding_energies(h,n=None,**kwargs):
    """Return the exciton binding energies, i.e. how far below the lowest
    independent-particle transition each exciton lies. Positive is bound"""
    return get_bse(h,**kwargs).get_binding_energies(n=n)
