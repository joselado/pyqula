"""Public Hamiltonian and Geometry methods that had no test, no example and
no user-guide entry (bug_audit_2 finding 61).  They were smoke-probed and
found correct, so what is missing is a statement of the invariant that keeps
them correct -- which is what this file adds.

Every check here is an invariant or a second code path in the repo computing
the same quantity, never a recorded number."""
import numpy as np
import pytest

from pyqula import geometry


# --------------------------------------------- has_time_reversal_symmetry

def test_time_reversal_is_detected_on_the_textbook_cases():
    """A Zeeman field and a Haldane flux break time reversal; Kane-Mele
    spin-orbit and Rashba spin-orbit do not.  These four are the standard
    counterexamples, and they pin the difference between "breaks TRS" and
    "merely breaks SU(2)"."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    assert h.has_time_reversal_symmetry()
    for term, breaks in [(lambda x: x.add_zeeman([0., 0., 0.3]), True),
                         (lambda x: x.add_haldane(0.2), True),
                         (lambda x: x.add_kane_mele(0.2), False),
                         (lambda x: x.add_rashba(0.2), False)]:
        h1 = h.copy()
        term(h1)
        assert h1.has_time_reversal_symmetry() == (not breaks), term


def test_an_in_plane_field_also_breaks_time_reversal():
    """The magnetization direction must not matter: time reversal flips all
    three components."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    for b in [[0.3, 0., 0.], [0., 0.3, 0.], [0.1, -0.2, 0.15]]:
        h1 = h.copy()
        h1.add_zeeman(b)
        assert not h1.has_time_reversal_symmetry(), b


# ------------------------------------------------------------------ get_1dh

def test_get_1dh_reproduces_the_two_dimensional_bands_at_fixed_ky():
    """get_1dh(k=ky) is the 2d Bloch Hamiltonian with the transverse
    momentum frozen, so its spectrum at kx must be the 2d spectrum at
    (kx,ky) -- exactly, not to a tolerance of a k-mesh."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hk2 = h.get_hk_gen()
    for ky in [0., 0.23, 0.5]:
        h1 = h.get_1dh(k=ky)
        assert h1.dimensionality == 1
        hk1 = h1.get_hk_gen()
        for kx in [0., 0.1, 0.31, 0.5, 0.77]:
            e2 = np.sort(np.linalg.eigvalsh(np.array(hk2([kx, ky, 0.]))))
            e1 = np.sort(np.linalg.eigvalsh(np.array(hk1([kx, 0., 0.]))))
            assert np.allclose(e1, e2, atol=1e-10), (kx, ky, e1, e2)


def test_get_1dh_leaves_the_two_dimensional_hamiltonian_alone():
    """It sets dimensionality=1 on the geometry it returns; that must not be
    the geometry of the Hamiltonian it was asked about."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h1 = h.get_1dh(k=0.2)
    assert h.dimensionality == 2
    assert h.geometry.dimensionality == 2
    assert h1 is not h


# ------------------------------------------ fractional2real / real2fractional

def test_fractional_and_real_coordinates_are_mutual_inverses():
    """The two conversions are each other's inverse, so the round trip has
    to be the identity on a lattice whose vectors are not orthogonal."""
    for g in [geometry.honeycomb_lattice(),
              geometry.honeycomb_lattice().get_supercell(2),
              geometry.kagome_lattice(),
              geometry.cubic_lattice()]:
        g.get_fractional()
        r0 = np.array(g.r)
        g1 = g.copy()
        g1.fractional2real()  # rebuild the real positions from frac_r
        assert np.allclose(np.array(g1.r), r0, atol=1e-12), g1.r
        g2 = g.copy()
        g2.real2fractional()
        g2.fractional2real()
        assert np.allclose(np.array(g2.r), r0, atol=1e-12), g2.r


def test_fractional_coordinates_transform_with_the_lattice_vectors():
    """Independent check of what the fractional coordinates mean: r must be
    the fractional coordinates contracted with the lattice vectors."""
    g = geometry.honeycomb_lattice()
    g.get_fractional()
    a = np.array([g.a1, g.a2])
    for (r, f) in zip(g.r, g.frac_r):
        assert np.allclose(np.array(r)[0:2], (np.array(f)[0:2]@a)[0:2],
                atol=1e-10), (r, f)


# --------------------------------------------------------- same_hamiltonian

def test_same_hamiltonian_separates_equal_from_merely_similar():
    """It compares Bloch matrices at random k, so a term that only shows up
    away from the zone centre must still be caught."""
    np.random.seed(0)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    assert h.same_hamiltonian(h.copy())
    for term in [lambda x: x.add_zeeman([0., 0., 0.9]),
                 lambda x: x.add_haldane(0.5),   # zero at Gamma, nonzero at K
                 lambda x: x.add_onsite(0.1)]:
        h1 = h.copy()
        term(h1)
        assert not h.same_hamiltonian(h1), term


def test_same_hamiltonian_ignores_the_multicell_representation():
    """The multicell and non-multicell forms of one Hamiltonian are the same
    Hamiltonian -- the comparison is between Bloch matrices, not storage."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hm = h.get_multicell()
    assert h.same_hamiltonian(hm)
    assert hm.same_hamiltonian(h)


# -------------------------------------------------------- to_canonical_gauge

def test_the_canonical_gauge_is_a_unitary_change_of_basis():
    """The two gauges differ by a diagonal unitary, so the Bloch matrix must
    stay Hermitian and its eigenvalues must not move."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hk = h.get_hk_gen()
    for k in [[0., 0., 0.], [0.3, 0.17, 0.], [0.5, 0.5, 0.]]:
        m = np.array(hk(k))
        mc = np.array(h.to_canonical_gauge(m, k))
        assert np.allclose(mc, np.conjugate(mc.T), atol=1e-10), k
        e0 = np.sort(np.linalg.eigvalsh(m))
        e1 = np.sort(np.linalg.eigvalsh(mc))
        assert np.allclose(e0, e1, atol=1e-10), (k, e0, e1)


def test_the_canonical_gauge_is_the_identity_at_the_zone_centre():
    """At k=0 every Bloch phase is 1, so the transformation must do nothing
    at all -- the cheapest independent check that the phases are built from
    the fractional coordinates and not from something else."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    m = np.array(h.get_hk_gen()([0., 0., 0.]))
    mc = np.array(h.to_canonical_gauge(m, [0., 0., 0.]))
    assert np.allclose(m, mc, atol=1e-12)
