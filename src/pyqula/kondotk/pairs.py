"""What the routines of kondotk read from a Kondo lattice mean field: the
list of Kondo pairs (c_j, f_j), the hybridization V_j that the Hamiltonian
holds between the two orbitals of a pair, and a copy of the Hamiltonian
with that hybridization rescaled.

The hybridization is read from the matrix elements of the Hamiltonian, not
from the `hybridization` attribute of a converged loop, so a Hamiltonian
whose V_j was set by hand works too. The mean-field term is
V_j f^dag_j c_j + h.c., so V_j is the element [f_j, c_j] of the intracell
matrix."""
import numpy as np
from scipy import linalg

from .. import algebra
from .. import check
from ..multihopping import MultiHopping


def get_pairs(h, what):
    """Return the Kondo pairs of h, or raise if h is not a Kondo lattice.
    `what` is the noun phrase the message opens with."""
    if not hasattr(h, "_kondo_pairs"):
        raise TypeError(what+" needs a KondoLatticeHamiltonian "
                "(or the Hamiltonian returned by its "
                "get_mean_field_hamiltonian), which carries the list of "
                "Kondo pairs; got "+type(h).__name__)
    check.require_spin(h, what)
    if h.has_eh:
        raise ValueError(what+" needs a Hamiltonian without "
                "the Nambu degree of freedom")
    return list(h._kondo_pairs)


def select_sites(site, npairs):
    """Indices of the Kondo pairs that are computed"""
    if site is None: return list(range(npairs))
    sites = [int(site)] if np.ndim(site) == 0 else [int(s) for s in site]
    for s in sites:
        if not -npairs <= s < npairs:
            raise ValueError("site="+str(s)+" is not a Kondo pair of this "
                    "Hamiltonian, which has "+str(npairs)+" pairs (0 to "
                    +str(npairs-1)+")")
    return [s % npairs for s in sites]


def _intra(h):
    """Dense intracell matrix of h"""
    return np.array(algebra.todense(h.get_multihopping().get_dict()[(0, 0, 0)]),
            dtype=np.complex128)


def pair_hybridization(h, pairs):
    """Hybridization V_j of every Kondo pair, read from the Hamiltonian:
    the average over the two spins of the element [f_j, c_j]."""
    m = _intra(h)
    return np.array([(m[2*fi, 2*ci] + m[2*fi+1, 2*ci+1])/2.
            for (ci, fi) in pairs])


def scale_hybridization(h, scale, pairs):
    """Return h with the hybridization of every Kondo pair multiplied by
    scale (a copy, h itself when scale is 1). The f levels and the
    conduction bands are not touched."""
    if scale == 1.0: return h
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("hybridization_scale must be a positive number, "
                "the factor that multiplies the hybridization V; got "
                +str(scale))
    hop = dict(h.get_multihopping().get_dict())
    m = _intra(h)
    for (ci, fi) in pairs: # the 2x2 spin blocks between c and f of a pair
        m[2*ci:2*ci+2, 2*fi:2*fi+2] *= scale
        m[2*fi:2*fi+2, 2*ci:2*ci+2] *= scale
    hop[(0, 0, 0)] = m
    out = h.copy()
    out.set_multihopping(MultiHopping(hop))
    if hasattr(h, "hybridization"):
        out.hybridization = scale*np.asarray(h.hybridization)
    return out


def is_spin_degenerate(h):
    """True if h is the same for the two spins and does not mix them, the
    case of a Kondo lattice mean field without a Zeeman field or a
    spin-orbit coupling"""
    from ..chitk.localspinchi import is_spin_degenerate
    return is_spin_degenerate(h)


def spin_components(m, degenerate):
    """Eigenstates of the spinful matrix m, as (energies, components,
    multiplicity): components is the list of the amplitudes [site, state]
    of the spin components that are kept, and every state counts
    multiplicity times.

    In general these are the two spin components of the states of m. With
    degenerate=True, m is the same for the two spins and does not mix them,
    so the block of one spin is diagonalized, a matrix of half the size,
    and each of its states comes twice, once per spin."""
    if degenerate:
        es, vs = linalg.eigh(m[0::2, 0::2])
        return es, [vs], 2.
    es, vs = linalg.eigh(m)
    return es, [vs[0::2, :], vs[1::2, :]], 1.


def pair_weights(m, ci, fi, degenerate):
    """Return (energies, wcc, wff, wcf): the eigenvalues of the Bloch
    matrix m and the weights [pair, state] of every eigenstate on the two
    orbitals of the Kondo pairs (ci[j], fi[j]), summed over the spin,

        wcc = sum_s |<c_js|n>|^2 , wff = sum_s |<f_js|n>|^2 ,
        wcf = sum_s <c_js|n> <n|f_js> ,

    from one diagonalization, of half the size with degenerate=True (see
    spin_components)."""
    es, comps, mult = spin_components(m, degenerate)
    wcc = mult*sum(np.abs(u[ci, :])**2 for u in comps)
    wff = mult*sum(np.abs(u[fi, :])**2 for u in comps)
    wcf = mult*sum(u[ci, :]*np.conjugate(u[fi, :]) for u in comps)
    return es, wcc, wff, wcf
