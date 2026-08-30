"""Combinations of the Nambu (electron-hole) degree of freedom with the
rest of the library that used to crash on the extra doubling."""
import numpy as np
import pytest

from pyqula import geometry


def test_orbital_field_commutes_with_turning_the_hamiltonian_nambu():
    """add_peierls indexed the site of an orbital assuming two orbitals per
    site, which is wrong for a Nambu Hamiltonian (four), and died with an
    IndexError. The hole block also has to pick up the *conjugate* Peierls
    phase, since it carries the opposite charge -- so adding the field
    before or after turn_nambu must give the same spectrum."""
    g = geometry.honeycomb_lattice().supercell(3)
    before = g.get_hamiltonian()
    before.add_peierls(0.02)
    before.turn_nambu()
    after = g.get_hamiltonian()
    after.turn_nambu()
    after.add_peierls(0.02)
    k = [0.1, 0.2, 0.]
    e1 = np.sort(np.linalg.eigvalsh(np.array(before.get_hk_gen()(k))))
    e2 = np.sort(np.linalg.eigvalsh(np.array(after.get_hk_gen()(k))))
    assert np.allclose(e1, e2, atol=1e-10)


def test_orbital_field_refuses_an_already_paired_hamiltonian():
    """There is no single Peierls phase for the anomalous term (a Cooper
    pair carries charge 2e, and an orbital field means vortices)."""
    h = geometry.honeycomb_lattice().supercell(3).get_hamiltonian()
    h.add_swave(0.2)
    with pytest.raises(NotImplementedError):
        h.add_peierls(0.02)


def test_filling_of_a_spinful_bdg_hamiltonian():
    """get_filling hit a bare `raise` for a spinful Nambu Hamiltonian. The
    Nambu spectrum is particle-hole symmetric, so counting negative
    eigenvalues is meaningless; the filling is the electron weight of the
    occupied BdG states, and at zero pairing it must reproduce the
    normal-state filling exactly."""
    g = geometry.chain()
    for shift in [0.0, -1.0, 0.7]:
        hn = g.get_hamiltonian()
        hn.add_onsite(shift)
        hb = hn.copy()
        hb.turn_nambu()
        assert abs(hn.get_filling(nk=40) - hb.get_filling(nk=40)) < 1e-9, shift
    hs = g.get_hamiltonian()
    hs.add_swave(0.3)
    assert abs(hs.get_filling(nk=40) - 0.5) < 1e-6  # half filled by symmetry


def test_berry_curvature_with_an_empty_occupied_manifold():
    """Every band above the Fermi energy leaves no occupied state at all;
    the Berry curvature is then zero, not an opaque TypeError from
    multiplying empty overlap matrices."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(10.0)
    assert np.allclose(h.get_berry_curvature(nk=4)[2], 0.0)
    assert abs(h.get_chern(nk=4)) < 1e-8
