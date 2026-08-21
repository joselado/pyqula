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

By default the BSE matrix is built and diagonalized densely, which is the
only route to the full non-Tamm-Dancoff spectrum but limits the k-mesh to
a few thousand electron-hole pairs. It does not have to be: the kernel is
EXACTLY a diagonal plus a fixed number of rank-one terms, one per non-zero
entry of the real-space interaction and independent of the mesh
(bsetk/factorize.py), so it can be applied without ever being assembled.
solver="iterative" does that and runs a block eigensolver; solver="qtt"
goes further and compresses the operator into a quantics matrix product
operator solved by DMRG, whose cost grows with log(nk) rather than nk.
Both are Tamm-Dancoff only. See bsetk/iterative.py and bsetk/qtt.py.

The same two-particle problem, restricted to electron-hole pairs whose
electron and hole have opposite spin, describes MAGNONS rather than
excitons -- bsetk/spinflip.py, exposed here as magnon_bands_tdhf /
magnon_energies / goldstone_residual. That route is what covers a
neighbor-shell (non-onsite) interaction in the spin channel, which the
site-basis RPA of chitk/spinchi.py structurally cannot; its own docstring
has the argument, and the Goldstone theorem is the test.

Passing screening="rpa" replaces the interaction of the direct (ladder)
term by the static RPA screened one W = eps^-1 v, built from the bands of
the mean field itself (screening.py), which is the GW-BSE-style
construction -- the exchange term keeps the bare interaction, as it must.
Read bsetk/screening.py before using it: a fitted Hubbard U is already an
effective screened interaction and must not be screened a second time.
"""

from .bsetk.solve import BSE
from .bsetk.bands import exciton_bands
from .bsetk.spinflip import (goldstone_residual, magnon_bands_tdhf,
                             magnon_energies)


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


def get_magnon_bands_tdhf(h,**kwargs):
    """Return the magnon bands of a mean-field Hamiltonian, from the
    spin-flip channel of the BSE, see bsetk.spinflip.magnon_bands_tdhf"""
    return magnon_bands_tdhf(h,**kwargs)


def get_magnon_energies(h,**kwargs):
    """Return the magnon energies at one momentum, see
    bsetk.spinflip.magnon_energies"""
    return magnon_energies(h,**kwargs)


def get_goldstone_residual(h,**kwargs):
    """Return how far a magnetic mean field is from having a zero-energy
    magnon at Q=0, see bsetk.spinflip.goldstone_residual"""
    return goldstone_residual(h,**kwargs)
