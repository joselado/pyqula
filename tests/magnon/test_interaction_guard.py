"""What the TDHF magnon route accepts as an interaction, and why.

The Goldstone theorem the whole route is validated by needs the
interaction to be invariant under a global spin rotation. A density-density
interaction always is, onsite or not. An exchange (J1/J2/J3, SzSz)
interaction is NOT, as pyqula stores it: h.V holds its Ising part, which
is the only part expressible as a density-density matrix, while the
transverse rung J/2 (S+_i S-_j + h.c.) that would make it isotropic is a
spin-flip two-body term with no such representation.

That is not a small omission. Solving the Ising kernel anyway returns a
perfectly ordinary looking magnon dispersion gapped by of order J at Q=0
(measured 1.81 for J1=3 on the honeycomb), with nothing to say the
acoustic branch should have been at zero -- so the interaction is checked
up front instead.

The limitation is in this kernel, not in the mean field: VJinteraction
decouples the x and y exchange channels as well (by rotating the density
matrix into the frame where that axis is the computational z, see
scftk/spinspin.py:580), so an isotropic-J mean field is a genuinely
SU(2)-symmetric state and the site-basis RPA -- which rebuilds the x/y
vertices by replicating the z one -- keeps its Goldstone mode on it. That
is what test_the_site_basis_rpa_still_has_its_goldstone_mode_for_isotropic_j
below pins, so that this guard is never mistaken for a statement about the
physics of exchange mean fields.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.spinflip import check_su2_interaction
from pyqula.meanfield import VJinteraction
from pyqula.scftk.spinspin import _build_density_v, _build_v

NK = 6


def test_density_density_interactions_are_accepted():
    g = geometry.chain()
    h = g.get_hamiltonian()
    for kwargs in ({"U": 2.0}, {"V1": 1.0}, {"U": 2.0, "V1": 1.0, "V2": 0.3}):
        W = {d: 2*m for d, m in _build_density_v(h, **kwargs).items()}
        check_su2_interaction(W)  # must not raise


def test_an_exchange_interaction_is_refused():
    g = geometry.chain()
    h = g.get_hamiltonian()
    W = {d: 2*m for d, m in _build_v(h, J1=1.0).items()}
    with pytest.raises(ValueError):
        check_su2_interaction(W)


def test_a_zeeman_like_onsite_interaction_is_refused():
    """The onsite block is checked differently from the bond ones -- a
    Hubbard U looks spin dependent in this basis and is nevertheless
    spin-rotation invariant -- so the onsite check has to be exercised on
    its own. An up-up/down-down imbalance there is a one-body exchange
    field in disguise, and does break the symmetry."""
    m = np.zeros((2, 2), dtype=np.complex128)
    m[0, 1] = m[1, 0] = 1.0  # a plain Hubbard U, fine
    check_su2_interaction({(0, 0, 0): m})
    m[0, 0] = 0.3  # up-up only: a Zeeman term
    with pytest.raises(ValueError):
        check_su2_interaction({(0, 0, 0): m})


@pytest.mark.slow
def test_a_converged_exchange_mean_field_is_refused_end_to_end():
    """The same guard, reached through the public API on a Hamiltonian
    that really was converged with a J1 bond exchange rather than on a
    hand-built matrix."""
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), J1=3.0, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-8, mix=0.3,
                         maxite=2000)
    hmf = scf.hamiltonian
    assert abs(hmf.get_vev("sz")[0]) > 0.1  # it did order
    with pytest.raises(ValueError):
        hmf.get_goldstone_residual(nk=NK)
    with pytest.raises(ValueError):
        hmf.get_magnon_energies(nk=NK)


@pytest.mark.slow
def test_the_ising_kernel_really_would_have_been_gapped():
    """The measurement behind the guard: switching the check off returns a
    magnon spectrum with no Goldstone mode at all. Kept as a test so that
    the number in the error message stays true, and so that a future
    transverse-rung implementation has something to compare against."""
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), J1=3.0, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-8, mix=0.3,
                         maxite=2000)
    es = scf.hamiltonian.get_magnon_energies(nk=NK, check_su2=False, n=1)
    assert es[0].real > 1.0  # a gap of order J where zero was required


def _rpa_kernel_min_eigenvalue(h, nk, delta=1e-4, q=(0., 0., 0.)):
    """The smallest eigenvalue of the site-basis spin RPA kernel
    1 - V*chi0(q,w=0), built exactly the way chitk.spinchi._full_spin_U
    would but bypassing its onsite-only guard.

    Zero means the RPA has a collective mode at zero frequency, which at
    q=0 is the Goldstone mode; 1.0 means the vertex came out empty and the
    kernel is the identity."""
    from pyqula.chitk.rpa import (build_ops_projectors,
                                   _chi_ops_matrix_vectorized,
                                   interaction_at_q)
    from pyqula.chitk.spinchi import (_full_spin_operators, V2K_matrix,
                                       replicateU)
    Ss = _full_spin_operators(h)
    V = {d: 2*replicateU(V2K_matrix(m), n=3) for d, m in h.V.items()}
    pAs, pBs = build_ops_projectors(h, Ss)
    q = list(q)
    _, chis = _chi_ops_matrix_vectorized(h, ops=Ss, pAs=pAs, pBs=pBs, q=q,
                                          energies=np.array([0.0]),
                                          delta=delta, nk=nk)
    Vq = interaction_at_q(V, h, q)
    K = np.identity(Vq.shape[0], dtype=np.complex128) - Vq@chis[0]
    return np.min(np.abs(np.linalg.eigvals(K)))


@pytest.mark.slow
def test_the_site_basis_rpa_still_has_its_goldstone_mode_for_isotropic_j():
    """The guard above is about what this kernel can carry, NOT about the
    exchange mean field being symmetry-broken in some deeper way. The mean
    field is fine -- VJinteraction decouples the x and y channels too --
    and the site-basis RPA, which replicates the z vertex across the three
    spin channels, is exactly consistent with it for an isotropic J.

    So its Goldstone mode is intact, and this pins that: the kernel's
    smallest eigenvalue at q=0, w=0 is zero (to the delta^2 that a finite
    broadening leaves), while the same quantity at finite q, and on a
    non-magnetic reference, is order one."""
    g = geometry.honeycomb_lattice()
    nk = NK
    scf = VJinteraction(g.get_hamiltonian(), J1=3.0, filling=0.5,
                         mf="antiferro", nk=nk, maxerror=1e-10, mix=0.3,
                         maxite=3000)
    hmf = scf.hamiltonian
    assert len(hmf.V) > 1  # non-onsite, i.e. what the RPA gate rejects
    assert _rpa_kernel_min_eigenvalue(hmf, nk) < 1e-6  # Goldstone
    assert _rpa_kernel_min_eigenvalue(hmf, nk, q=(0.1, 0., 0.)) > 1e-2
    bare = g.get_hamiltonian()  # non-magnetic, same vertex
    bare.V = hmf.V
    assert _rpa_kernel_min_eigenvalue(bare, nk) > 1e-2


@pytest.mark.slow
def test_the_site_basis_rpa_has_no_vertex_at_all_for_a_v1_ordered_magnet():
    """The case where the site-basis RPA genuinely fails, and the reason
    the TDHF route exists. A chain ordered ferromagnetically by V1 alone
    (no U) is magnetic -- V1's Fock term is what orders it -- but the
    Sz_i Sz_j coefficient of a spin-independent V_ij is exactly zero, so
    the vertex is empty, the kernel is the identity, and there is no
    collective mode anywhere.

    Contrast with the isotropic-J test above: a non-onsite h.V is not by
    itself a problem for the RPA. It is a problem when the piece the
    vertex drops is one the mean field actually used."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    v = np.array([0., 0., 1.])
    h.add_exchange(1e-2*v)
    mf = h.copy()
    mf.add_exchange(0.5*v)
    nk = 200  # a 1d Stoner instability at low filling needs a fine mesh
    scf = VJinteraction(h, V1=1.1, filling=0.1, mf=mf, nk=nk, mix=0.2,
                         maxerror=1e-8, maxite=1000)
    hmf = scf.hamiltonian
    assert abs(hmf.get_vev("sz")[0]) > 0.05  # genuinely magnetic
    V = {d: 2*np.abs(m) for d, m in hmf.V.items()}
    assert max(np.max(np.abs(m)) for m in V.values()) > 0  # h.V is not empty
    # ... but the spin vertex extracted from it is
    assert abs(_rpa_kernel_min_eigenvalue(hmf, nk) - 1.0) < 1e-9
