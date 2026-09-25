"""Spin Chern number with the occupied states split by the sign of P s_z P
(Prodan), and mirror Chern number with the mirror z -> -z found by the
point-group code (h.get_spin_chern, h.get_mirror_chern).

The spin Chern number the package had, topology.spin_chern, integrates an
s_z-weighted Berry curvature, which is C_up - C_down only while s_z is
conserved; with Rashba coupling it is no longer an integer. The checks here
are the relations the split invariants have to obey with the independent
routes that exist: the s_z-weighted integral where s_z is conserved, the
Z2 invariant (C_s modulo 2) where time reversal holds, and the counting of
copies in a bilayer."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhamiltonian
from pyqula import topology


def _kane_mele(rashba=0., mass=0., exchange=0.):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_kane_mele(0.1)
    if rashba:
        h.add_rashba(rashba)
    if mass:
        h.add_sublattice_imbalance(mass)
    if exchange:
        h.add_exchange([0., 0., exchange])
    return h


def test_spin_chern_is_half_the_weighted_integral_when_sz_is_conserved(
        tmp_path, monkeypatch):
    """topology.spin_chern integrates to C_up - C_down, twice C_s; the
    mirror of a single layer is -i sigma_z, so C_M = -C_s"""
    monkeypatch.chdir(tmp_path)
    h = _kane_mele()
    cs = h.get_spin_chern(nk=16)
    assert np.isclose(cs, 1.)
    assert np.isclose(2*cs, topology.spin_chern(h, nk=24), atol=5e-2)
    assert np.isclose(h.get_mirror_chern(nk=16), -cs)


def test_spin_chern_stays_quantized_with_rashba(tmp_path, monkeypatch):
    """Rashba coupling breaks s_z conservation and the mirror, and C_s is
    still an integer whose parity is the Z2 invariant"""
    monkeypatch.chdir(tmp_path)
    h = _kane_mele(rashba=0.1)
    cs = h.get_spin_chern(nk=16)
    assert np.isclose(cs, 1.)
    assert (-1)**int(round(cs)) == topology.z2_invariant(h, nk=30, nt=30)
    with pytest.raises(ValueError, match="no mirror"):
        h.get_mirror_chern(nk=16)


def test_trivial_insulator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _kane_mele(mass=1.5)
    assert np.isclose(h.get_spin_chern(nk=16), 0., atol=1e-8)
    assert np.isclose(h.get_mirror_chern(nk=16), 0., atol=1e-8)


def test_split_invariants_survive_broken_time_reversal(tmp_path,
                                                      monkeypatch):
    """An out-of-plane exchange field breaks time reversal, so there is no
    Z2 invariant, but it keeps the mirror, and until the gap closes both
    split invariants stay where they were, with a vanishing Chern number"""
    monkeypatch.chdir(tmp_path)
    h = _kane_mele(exchange=0.3)
    assert not h.has_time_reversal_symmetry()
    assert np.isclose(h.get_spin_chern(nk=16), 1.)
    assert np.isclose(h.get_mirror_chern(nk=16), -1.)
    assert np.isclose(h.get_chern(nk=16), 0., atol=1e-8)


def test_bilayer_carries_two_copies(tmp_path, monkeypatch):
    """Two coupled quantum spin Hall layers: C_s = 2, an even number, so
    the Z2 invariant is trivial, and the two mirror sectors (bonding with
    one spin, antibonding with the other) carry opposite Chern numbers that
    cancel, C_M = 0"""
    monkeypatch.chdir(tmp_path)
    h = specialhamiltonian.multilayer_graphene(l=[0, 0], ti=0.3)  # AA
    h.add_kane_mele(0.1)
    assert np.isclose(h.get_spin_chern(nk=16), 2.)
    assert topology.z2_invariant(h, nk=30, nt=30) == 1
    assert np.isclose(h.get_mirror_chern(nk=16), 0., atol=1e-8)


def test_split_invariants_refuse_what_they_cannot_split(tmp_path,
                                                       monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _kane_mele(rashba=0.2)  # the gap closes: a metal
    with pytest.raises(ValueError, match="number of occupied states"):
        h.get_spin_chern(nk=16)
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="spinful"):
        h.get_spin_chern()
