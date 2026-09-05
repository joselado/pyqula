import numpy as np
import pytest

from pyqula import geometry


def _spectrum(h):
    m = h.intra
    if hasattr(m, "todense"):
        m = m.todense()
    return np.sort(np.linalg.eigvalsh(np.array(m)))


def test_periodic_ring_has_the_exact_ring_spectrum():
    """set_finite_system zeroed h.dimensionality two lines before the
    `if periodic:` block tested it, so both branches were false by
    construction and periodic=True silently returned an open cluster.
    A 6-site ring of unit hoppings has eigenvalues 2cos(2 pi n/6)."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ring = h.set_finite_system(n=6, periodic=True)
    exact = np.sort(2. * np.cos(2. * np.pi * np.arange(6) / 6.))
    assert np.max(np.abs(_spectrum(ring) - exact)) < 1e-8


def test_periodic_differs_from_open():
    """The bug made the two bit-identical."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ring = _spectrum(h.set_finite_system(n=6, periodic=True))
    line = _spectrum(h.set_finite_system(n=6, periodic=False))
    assert np.max(np.abs(ring - line)) > 1e-3


def test_two_dimensional_periodic_closes_both_directions():
    """A 2x2 supercell of a square lattice wrapped in both directions has
    every site coupled to itself twice in each direction, so the spectrum
    collapses onto the four k-points of the 2x2 mesh."""
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    torus = _spectrum(h.set_finite_system(n=2, periodic=True))
    plane = _spectrum(h.set_finite_system(n=2, periodic=False))
    assert np.max(np.abs(torus - plane)) > 1e-3


def test_three_dimensional_periodic_is_refused():
    """Rather than silently returning an open cluster."""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        h.set_finite_system(n=2, periodic=True)
