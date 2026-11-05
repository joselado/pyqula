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


# Corner charge (Benalcazar, Li and Hughes, arXiv:1809.02142), checked
# against the charge counted on a C_n-symmetric flake of whole unit cells:
# the ions carry nocc per cell, the corner multiplet is filled or emptied as
# a whole by putting the Fermi level in the largest spectral gap around the
# bulk filling, and the flake is cut into n pinwheel sectors of whole cells,
# each holding one corner, with the central cell shared equally.

def _kekule_cell(t0):
    """H_2^(6) of arXiv:1809.02142: six sites on a hexagon of radius 1,
    hopping t0 around the hexagon and 1 between hexagons; a corner charge
    1/2 for t0<1"""
    s3 = np.sqrt(3.)
    a1, a2 = np.array([3., 0, 0]), np.array([1.5, 1.5*s3, 0])
    sites = [np.array([np.cos(np.pi/3*j), np.sin(np.pi/3*j), 0.])
             for j in range(6)]
    R = {0: (1, 0), 1: (0, 1), 2: (-1, 1), 3: (-1, 0), 4: (0, -1), 5: (1, -1)}
    bonds = {}
    for j in range(6):
        bonds[(j, (j+1) % 6, (0, 0))] = t0
        bonds[(j, (j+3) % 6, R[j])] = 1.
    return a1, a2, sites, bonds, np.zeros(6), 3


def _stack_cell(top1b, top2c, t=0.2):
    """h_1b (+) h_2c of arXiv:1809.02142 at 3/8 filling: a 2D SSH model and
    two perpendicular SSH chains, weakly coupled, each in its topological
    phase (inter-cell hopping 1, intra-cell t) or in the trivial one (the
    two exchanged, the same spectrum). Each block alone has a polarization
    (1/2,1/2), so the stack has zero polarization only when both are
    topological or both trivial"""
    d = 0.3
    a1, a2 = np.array([1., 0, 0]), np.array([0., 1, 0])
    sites = [np.array([d, d, 0]), np.array([-d, d, 0]), np.array([-d, -d, 0]),
             np.array([d, -d, 0]), np.array([d, 0, 0]), np.array([0, d, 0]),
             np.array([-d, 0, 0]), np.array([0, -d, 0])]
    A1, B1 = (t, 1.) if top1b else (1., t)
    A2, B2 = (t, 1.) if top2c else (1., t)
    bonds = {}
    for j in range(4): bonds[(j, (j+1) % 4, (0, 0))] = A1
    bonds[(0, 1, (1, 0))] = B1; bonds[(3, 2, (1, 0))] = B1
    bonds[(0, 3, (0, 1))] = B1; bonds[(1, 2, (0, 1))] = B1
    bonds[(4, 6, (0, 0))] = A2; bonds[(5, 7, (0, 0))] = A2
    bonds[(4, 6, (1, 0))] = B2; bonds[(5, 7, (0, 1))] = B2
    for (i, j) in [(0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 6), (2, 7), (3, 4)]:
        bonds[(i, j, (0, 0))] = 0.05 # a coupling that keeps C4
    return a1, a2, sites, bonds, 0.6*np.ones(8), 3


def _bulk_from_cell(cell):
    a1, a2, sites, bonds, onsite, nocc = cell
    A = np.array([a1[:2], a2[:2]]).T
    S = np.array(sites)

    def ident(r):
        for (i, s) in enumerate(S):
            f = np.linalg.solve(A, (r - s)[:2])
            if np.allclose(f, np.round(f), atol=1e-6):
                return i, np.round(f).astype(int)
        raise AssertionError("not a site of the lattice")

    def fun(r1, r2):
        (i, R1), (j, R2) = ident(r1), ident(r2)
        dR = tuple(R2 - R1)
        if i == j and dR == (0, 0): return onsite[i]
        if (i, j, dR) in bonds: return bonds[(i, j, dR)]
        if (j, i, (-dR[0], -dR[1])) in bonds: return bonds[(j, i, (-dR[0], -dR[1]))]
        return 0.
    g = geometry.square_lattice()
    g.a1, g.a2 = a1, a2
    g.r = np.array(sites)
    g.r2xyz()
    g.get_fractional()
    return g.get_hamiltonian(fun=fun, has_spin=False)


def _flake_sector_charges(cell, n, L, corner_angle, square):
    """Total charge of each of the n pinwheel sectors of a flake, a square
    of (2L+1)^2 cells or a hexagon of cells of radius L"""
    a1, a2, sites, bonds, onsite, nocc = cell
    cells = [(i, j) for i in range(-L, L+1) for j in range(-L, L+1)
             if square or abs(i + j) <= L]
    index = {c: m for (m, c) in enumerate(cells)}
    ns = len(sites)
    H = np.zeros((ns*len(cells), ns*len(cells)))
    for (m, (i, j)) in enumerate(cells):
        for s in range(ns): H[m*ns + s, m*ns + s] = onsite[s]
        for ((s1, s2, dR), t) in bonds.items():
            c2 = (i + dR[0], j + dR[1])
            if c2 not in index: continue
            p, q = m*ns + s1, index[c2]*ns + s2
            H[p, q] = H[q, p] = t
    e, v = np.linalg.eigh(H)
    nb = nocc*len(cells) # electrons of a neutral flake
    w = e[nb - 6*n: nb + 6*n]
    k = np.argmax(np.diff(w))
    rho = np.sum(np.abs(v[:, e < (w[k] + w[k+1])/2.])**2, axis=1)
    sectors = np.zeros(n)
    for (m, (i, j)) in enumerate(cells):
        q = nocc - np.sum(rho[m*ns:(m+1)*ns]) # ions minus electrons
        if (i, j) == (0, 0):
            sectors += q/n
            continue
        r = i*a1 + j*a2
        ang = (np.arctan2(r[1], r[0]) - corner_angle + np.pi/n) % (2*np.pi)
        sectors[int(np.floor(ang/(2*np.pi/n) + 1e-9)) % n] += q
    return sectors


@pytest.mark.parametrize("cell,n,expected", [
    (_kekule_cell(0.5), 6, 0.5),
    (_kekule_cell(1.5), 6, 0.),
    (_stack_cell(True, True), 4, 0.75),
    (_stack_cell(False, False), 4, 0.),
])
def test_corner_charge_matches_the_flake(cell, n, expected):
    """The index gives the charge of every corner of the flake, modulo 1,
    and 3/4 for the stack is the 3|e|/4 of fig. 1(a) of the paper, which
    fixes the sign: the index of the paper is the electronic charge"""
    h = _bulk_from_cell(cell)
    q = h.get_corner_charge(n=n)
    assert abs(q - expected) < 1e-8
    assert abs(h.get_corner_charge() - expected) < 1e-8 # the largest C_n
    square = (n == 4)
    sectors = _flake_sector_charges(cell, n, 4, np.pi/4 if square else 0.,
                                    square)
    for s in sectors:
        assert abs((s - q + 0.5) % 1. - 0.5) < 1e-3


@pytest.mark.parametrize("cell,n", [(_kekule_cell(0.5), 6),
                                    (_stack_cell(True, True), 4)])
def test_c2_corner_index_is_the_charge_of_half_the_flake(cell, n):
    """With C2 only, the two sectors are halves of the flake, so the C2
    index is n/2 times the corner charge of the C_n index, modulo 1"""
    h = _bulk_from_cell(cell)
    q2 = h.get_corner_charge(n=2)
    assert abs((q2 - n/2*h.get_corner_charge(n=n) + 0.5) % 1. - 0.5) < 1e-8
    square = (n == 4)
    for s in _flake_sector_charges(cell, 2, 4, np.pi/4 if square else 0.,
                                   square):
        assert abs((s - q2 + 0.5) % 1. - 0.5) < 1e-3


@pytest.mark.parametrize("top", [(True, False), (False, True)])
def test_corner_charge_with_a_polarization_raises(top):
    """Only one block topological: a polarization (1/2,1/2), charged edges"""
    h = _bulk_from_cell(_stack_cell(*top))
    with pytest.raises(ValueError, match="polarization"):
        h.get_corner_charge(n=4)


def test_corner_charge_needs_a_spinless_time_reversal_insulator():
    h = _bulk_from_cell(_kekule_cell(0.5))
    with pytest.raises(NotImplementedError, match="K"):
        h.get_corner_charge(n=3)
    hs = h.copy()
    hs.add_haldane(0.05)
    with pytest.raises(ValueError, match="time-reversal"):
        hs.get_corner_charge()
    with pytest.raises(ValueError, match="spinless"):
        _haldane(0.1, spin=True).get_corner_charge()


@pytest.mark.parametrize("delta,expected", [(-0.3, 0.5), (0.3, 0.)])
def test_corner_charge_of_the_kekule_honeycomb(delta, expected):
    """The C6 cell of the honeycomb lattice with a Kekule modulation of the
    hopping: weaker bonds inside the hexagon of the cell give the phase of
    _kekule_cell(t0<1), with corner charge 1/2"""
    g = geometry.honeycomb_lattice_C6()
    h = g.get_hamiltonian(has_spin=False)
    h.add_kekule(delta)
    assert abs(h.get_corner_charge() - expected) < 1e-8
