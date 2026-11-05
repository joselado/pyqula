"""Symmetry eigenvalues of the occupied states and the invariants built from
them, each checked against an independent route of the package: the Chern
number modulo n from the C_n eigenvalues (Fang-Gilbert-Bernevig,
arXiv:1207.5767) against the Berry curvature of h.get_chern(), and the
parity Z2 (Fu-Kane, cond-mat/0611341) against the Wannier-center flow of
z2_invariant and z2_invariant_3d."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology
from pyqula.topologytk.symmetryeigenvalues import (rotation_operation,
                                                   inversion_operation)

HEXAGON = np.array([1.5, 0., 0.]) # a hexagon center of honeycomb_lattice()
BOND = np.zeros(3) # the bond center of honeycomb_lattice()


def _qwz(m):
    """Spinful square lattice with a spin-dependent hopping t s_z, Rashba
    coupling and an exchange m s_z: the Qi-Wu-Zhang model, C4 symmetric
    with the rotation acting on the spin"""
    h = geometry.square_lattice().get_hamiltonian().get_multicell()
    sz = np.diag([1., -1.])
    for t in h.hopping: t.m = np.array(t.m)@sz
    h.add_rashba(1.0)
    h.add_exchange([0., 0., m])
    return h


def _haldane(t2, spin=False):
    """Haldane model; spinful, the exchange 4 and the onsite 4 leave a
    single occupied band, so the (-1)^F factor is tested at odd filling"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=spin)
    h.add_haldane(t2)
    if spin:
        h.add_exchange([0., 0., 4.])
        h.add_onsite(4.)
    return h


@pytest.mark.parametrize("m", [1., -1., 3., -3., 5.])
def test_chern_mod_4_and_2_of_qwz(m, tmp_path, monkeypatch):
    """Both signs of the Chern number, and the trivial phase, so that a
    rotation with the wrong orientation (conjugate eigenvalues) fails"""
    monkeypatch.chdir(tmp_path)
    h = _qwz(m)
    c = int(round(h.get_chern(nk=40)))
    assert c == (0 if abs(m) > 4 else -int(np.sign(m)))
    for n in (4, 2):
        assert topology.chern_mod_n(h, n=n) == c % n
    assert h.get_chern_mod_n() == c % 4 # the largest rotation, C4


@pytest.mark.parametrize("t2", [0.1, -0.1])
@pytest.mark.parametrize("spin", [False, True])
def test_chern_mod_n_of_haldane(t2, spin, tmp_path, monkeypatch):
    """C6, C3 about two different centers, and C2, give the same Chern
    number, which does not depend on the rotation axis chosen"""
    monkeypatch.chdir(tmp_path)
    h = _haldane(t2, spin=spin)
    c = int(round(h.get_chern(nk=40)))
    assert abs(c) == 1
    g = h.geometry
    for (n, center) in [(6, HEXAGON), (3, HEXAGON), (3, g.r[0]), (2, BOND),
                        (2, HEXAGON)]:
        assert topology.chern_mod_n(h, n=n, center=center) == c % n


@pytest.mark.parametrize("spin", [False, True])
def test_rotation_eigenvalues_are_roots_of_unity(spin):
    """C_n^n is the identity for a spinless state and a 2 pi rotation, -1,
    for a spinful one"""
    h = _haldane(0.1, spin=spin)
    op = rotation_operation(6, HEXAGON)
    ev = h.get_symmetry_eigenvalues(op, [0., 0.])
    assert np.allclose(ev**6, -1. if spin else 1., atol=1e-8)
    for k in topology.invariant_momenta(h, rotation_operation(3, HEXAGON)):
        ev = topology.symmetry_eigenvalues(h, rotation_operation(3, HEXAGON), k)
        assert np.allclose(ev**3, -1. if spin else 1., atol=1e-8)


def test_invariant_momenta_of_c4_and_c3():
    h = _qwz(1.)
    assert len(topology.invariant_momenta(h, rotation_operation(4))) == 2
    assert len(topology.invariant_momenta(h, rotation_operation(2))) == 4
    h = _haldane(0.1)
    assert len(topology.invariant_momenta(h, rotation_operation(3, HEXAGON))) == 3
    assert len(topology.invariant_momenta(h, rotation_operation(6, HEXAGON))) == 1


def _kane_mele(s):
    """Kane-Mele model with the first-neighbor hopping along x multiplied
    by s, which keeps the inversion about the bond center; at s=2 the two
    Dirac cones merge at a time-reversal-invariant momentum, and beyond it
    the bonds along x dimerize the lattice into a trivial insulator"""
    g = geometry.honeycomb_lattice()

    def fun(r1, r2):
        dr = r1 - r2
        if abs(np.linalg.norm(dr) - 1.) > 1e-4: return 0.
        return s if abs(dr[1]) < 1e-4 else 1.
    h = g.get_hamiltonian(fun=fun, has_spin=True)
    h.add_kane_mele(0.1)
    return h


@pytest.mark.parametrize("s", [1., 1.5, 2.5, 4.])
def test_parity_z2_matches_wannier_flow_in_2d(s, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _kane_mele(s)
    z2 = topology.parity_z2(h)
    assert z2 == int(s < 2.)
    # z2_invariant returns the sign (-1)^nu
    assert (-1)**z2 == topology.z2_invariant(h, nk=30, nt=30)


def test_parity_z2_does_not_depend_on_the_inversion_center():
    """The bond center and the bond center moved by half a lattice vector
    are both inversion centers, with different parities at two of the four
    momenta, and give the same Z2"""
    h = _kane_mele(1.)
    g = h.geometry
    for center in [BOND, BOND + np.array(g.a1)/2., BOND + np.array(g.a2)/2.]:
        assert h.get_parity_z2(center=center) == 1


def _diamond(bond, s):
    """Fu-Kane-Mele diamond model with the first-neighbor bonds parallel to
    one of the four bond vectors multiplied by s"""
    g = geometry.diamond_lattice_minimal()
    A = np.array([g.a1, g.a2, g.a3])
    bonds = []
    for n in np.ndindex(3, 3, 3):
        d = g.r[1] + (np.array(n) - 1)@A - g.r[0]
        if abs(np.linalg.norm(d) - 1.) < 1e-4: bonds.append(d)
    dvec = bonds[bond]

    def fun(r1, r2):
        dr = r1 - r2
        if abs(np.linalg.norm(dr) - 1.) > 1e-4: return 0.
        return s if np.linalg.norm(np.cross(dr, dvec)) < 1e-4 else 1.
    h = g.get_hamiltonian(fun=fun, has_spin=True)
    h.add_kane_mele(0.05)
    return h


@pytest.mark.parametrize("bond,s", [(0, 0.7), (2, 1.3), (1, 3.1)])
def test_parity_z2_matches_wannier_flow_in_3d(bond, s, tmp_path, monkeypatch):
    """A weak, a strong and a trivial insulator, with the weak vector set
    by the bond"""
    monkeypatch.chdir(tmp_path)
    h = _diamond(bond, s)
    assert h.get_parity_z2() == topology.z2_invariant_3d(h, nk=30, nt=30)


def test_a_momentum_that_is_not_invariant_raises():
    h = _haldane(0.1)
    with pytest.raises(ValueError, match="not the same momentum"):
        h.get_symmetry_eigenvalues(rotation_operation(3, HEXAGON), [0.5, 0.])


def test_a_missing_rotation_raises():
    h = _haldane(0.1)
    with pytest.raises(ValueError, match="not a symmetry"):
        topology.chern_mod_n(h, n=4)


def test_a_gap_closing_raises():
    """At s=2 the Dirac cones of Kane-Mele merge at a time-reversal-invariant
    momentum, where the Kane-Mele term vanishes, so the gap closes there and
    the states below zero energy are not a representation of the inversion"""
    h = _kane_mele(2.)
    with pytest.raises(ValueError,
                       match="not mapped onto themselves|not an insulator"):
        h.get_parity_z2()


def test_a_metal_raises():
    """A Fermi level inside the bands of the Haldane model: each momentum
    alone has well-defined eigenvalues, but their number changes"""
    h = _haldane(0.1)
    h.add_onsite(-1.5)
    with pytest.raises(ValueError, match="not an insulator"):
        topology.chern_mod_n(h, n=3, center=HEXAGON)
    h = _kane_mele(1.)
    h.add_onsite(-1.5)
    with pytest.raises(ValueError, match="not an insulator"):
        h.get_parity_z2()


def test_parity_z2_needs_time_reversal():
    h = _kane_mele(1.)
    h.add_exchange([0., 0., 0.05])
    with pytest.raises(ValueError, match="time-reversal"):
        h.get_parity_z2()


def test_parity_z2_needs_inversion():
    h = _kane_mele(1.)
    h.add_sublattice_imbalance(0.1)
    with pytest.raises(ValueError, match="no inversion"):
        h.get_parity_z2()
