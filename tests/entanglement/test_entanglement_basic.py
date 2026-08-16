"""Elementary invariants of the correlation-matrix entanglement entropy:
a region with no entanglement has S=0, a region and its complement have
the same entropy for a pure global state, and the different ways of
specifying the region must select the same sites."""

import numpy as np
import pytest

from pyqula import geometry


def _finite_chain(n=12, has_spin=False):
    """Open chain of n sites as a 0-dimensional Hamiltonian"""
    g = geometry.chain()
    g = g.supercell(n)
    g.dimensionality = 0  # cut the periodic boundary: a finite molecule
    return g.get_hamiltonian(has_spin=has_spin)


def test_filled_and_empty_systems_have_zero_entropy():
    """A completely full or completely empty band structure is a product
    state (C_A is the identity or zero), so every region is unentangled."""
    h = _finite_chain()
    bandwidth = np.max(np.abs(np.linalg.eigvalsh(np.array(h.intra))))
    s_full = h.get_entanglement_entropy(fermi=10 * bandwidth)
    s_empty = h.get_entanglement_entropy(fermi=-10 * bandwidth)
    assert abs(s_full) < 1e-12
    assert abs(s_empty) < 1e-12


def test_entropy_is_symmetric_under_swapping_the_two_halves():
    """For a pure global state rho_A and rho_B have the same nonzero
    spectrum, so S(A) = S(B) whatever the (here quite asymmetric) cut."""
    h = _finite_chain(n=12)
    for n_a in [1, 3, 5, 6, 9]:
        region_a = list(range(n_a))
        region_b = list(range(n_a, 12))
        s_a = h.get_entanglement_entropy(region=region_a)
        s_b = h.get_entanglement_entropy(region=region_b)
        assert abs(s_a - s_b) < 1e-10
        assert s_a > 0.0  # a gapless chain really is entangled


def test_region_can_be_given_as_indices_or_as_a_position_selector():
    """The list-of-indices, callable-on-positions and fraction-of-the-cell
    forms of `region` must all resolve to the same set of sites."""
    h = _finite_chain(n=12)
    x_mid = np.mean(h.geometry.r[:, 0])
    s_indices = h.get_entanglement_entropy(region=list(range(6)))
    s_callable = h.get_entanglement_entropy(region=lambda r: r[0] < x_mid)
    s_default = h.get_entanglement_entropy()
    assert abs(s_indices - s_callable) < 1e-12
    assert abs(s_indices - s_default) < 1e-12
    # half of a 14-cell ring, as a fraction and as explicit cells (which
    # are a different, but translated, set of 7 consecutive cells: the
    # ring is translationally invariant, so the entropy must agree)
    h_periodic = geometry.chain().get_hamiltonian(has_spin=False)
    s_fraction = h_periodic.get_entanglement_entropy(nsuper=14, region=0.5)
    s_cells = h_periodic.get_entanglement_entropy(nsuper=14,
                                                  region=list(range(7)))
    assert abs(s_fraction - s_cells) < 1e-10


def test_spinful_chain_doubles_the_entropy_of_the_spinless_one():
    """Spin is just another orbital index: two decoupled, identical spin
    channels carry exactly twice the entanglement of one."""
    h_spinless = _finite_chain(n=10, has_spin=False)
    h_spinful = _finite_chain(n=10, has_spin=True)
    s1 = h_spinless.get_entanglement_entropy()
    s2 = h_spinful.get_entanglement_entropy()
    assert abs(s2 - 2 * s1) < 1e-10


def test_degenerate_fermi_level_raises_instead_of_guessing():
    """A ring of 12 sites at half filling has states exactly at E=0: the
    ground state is degenerate, so there is no single Slater determinant
    to compute the entanglement of, and the occupied set must not be
    picked silently."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_entanglement_entropy(nsuper=12)
    # the same ring with 14 sites is non-degenerate and works
    assert h.get_entanglement_entropy(nsuper=14) > 0.0
