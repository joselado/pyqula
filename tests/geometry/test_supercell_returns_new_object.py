"""`get_supercell` used to return `self` for the no-op cases (nsuper=1,
nsuper=None, a zero-dimensional system), so a caller that mutated the
"supercell" was silently mutating the object it had asked about."""
import numpy as np

from pyqula import geometry


def bandwidth(h):
    (k, e) = h.get_bands()
    return np.max(e) - np.min(e)


def test_a_trivial_supercell_does_not_leak_back_into_the_hamiltonian():
    """The n=1 iteration of a parameter sweep used to hand back the base
    Hamiltonian itself, so the Zeeman field it added stayed there for good."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    b0 = bandwidth(h)
    for n in [1, 2]:
        hn = h.get_supercell(n)
        assert hn is not h
        hn.add_zeeman([0., 0., 0.5])
    assert abs(bandwidth(h) - b0) < 1e-8, (bandwidth(h), b0)


def test_no_supercell_at_all_is_still_a_new_hamiltonian():
    """nsuper=None means "do nothing", not "hand the caller my own state"."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    b0 = bandwidth(h)
    hn = h.get_supercell(None)
    assert hn is not h
    hn.add_zeeman([0., 0., 0.5])
    assert abs(bandwidth(h) - b0) < 1e-8


def test_zero_dimensional_supercell_is_a_copy():
    """A 0d system has nothing to replicate, but the no-op must not alias
    -- for the Hamiltonian and for the geometry alike."""
    g = geometry.honeycomb_lattice().get_supercell(3)
    g.set_finite()
    h = g.get_hamiltonian()
    b0 = bandwidth(h)
    hs = h.get_supercell(2)
    assert hs is not h
    hs.add_zeeman([0., 0., 0.5])
    assert abs(bandwidth(h) - b0) < 1e-8
    gs = h.geometry.get_supercell(2)
    assert gs is not h.geometry
    gs.shift([1., 0., 0.])
    assert np.allclose(h.geometry.r, g.r)


def test_geometry_supercell_of_one_is_a_new_geometry():
    g = geometry.honeycomb_lattice()
    r0 = g.r.copy()
    gs = g.get_supercell(1)
    assert gs is not g
    gs.shift([1., 0., 0.])
    assert np.allclose(g.r, r0)


def test_supercell_accepts_an_array_of_repetitions():
    """(2,2,1) as a numpy array used to hit `nsuper==1` on a vector, which
    raises "truth value of an array is ambiguous"."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    n = len(h.geometry.r)
    for ns in [(2, 2, 1), [2, 2, 1], np.array([2, 2, 1])]:
        assert len(h.get_supercell(ns).geometry.r) == 4*n


def test_reduce_returns_a_new_hamiltonian_in_every_branch():
    """`reduce` returns a fresh spinless Hamiltonian when the spin can be
    dropped, and used to return `self` when it could not."""
    hs = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    assert hs.reduce() is not hs
    hm = geometry.honeycomb_lattice().get_hamiltonian()
    hm.add_zeeman([0., 0., 0.3])  # cannot be reduced to spinless
    b0 = bandwidth(hm)
    hr = hm.reduce()
    assert hr is not hm
    hr.add_zeeman([0., 0., 0.5])
    assert abs(bandwidth(hm) - b0) < 1e-8
