"""Strong and weak Z2 indices, and the Chern vector, of three-dimensional
insulators, from the Wannier-center flow of the planes of the Brillouin
zone (topology.z2_invariant_3d, topology.chern_vector).

get_topological_invariant used to raise for every 3D Hamiltonian. The Z2
indices are checked against the Fu-Kane parity formula, an independent
route that needs only the inversion eigenvalues of the occupied states at
the eight time-reversal-invariant momenta (compile_symmetry gives the
inversion operator), on the Fu-Kane-Mele diamond model, where making one of
the four first-neighbor bonds weaker gives a weak topological insulator
and making it stronger a strong one, each with the weak vector set by the
bond. The Chern vector is checked on stacked Haldane layers against the
Chern number of one layer."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology
from pyqula.symmetrytk.pointgroup import SymmetryOperation, compile_symmetry


def _bonds(g):
    """The four first-neighbor bond vectors from site A of the diamond"""
    A = np.array([g.a1, g.a2, g.a3])
    out = []
    for n in np.ndindex(3, 3, 3):
        d = g.r[1] + (np.array(n) - 1)@A - g.r[0]
        if abs(np.linalg.norm(d) - 1.) < 1e-4:
            out.append(d)
    return out


def _scale_bond(h, dvec, s):
    """Multiply by s every first-neighbor hopping parallel to dvec"""
    g = h.geometry
    A = np.array([g.a1, g.a2, g.a3])

    def f(m, R):
        m = np.array(m.todense() if hasattr(m, "todense") else m,
                     dtype=complex)
        for i in range(2):
            for j in range(2):
                d = g.r[j] + np.array(R)@A - g.r[i]
                if abs(np.linalg.norm(d) - 1.) < 1e-4 and \
                        np.linalg.norm(np.cross(d, dvec)) < 1e-4:
                    m[2*i:2*i+2, 2*j:2*j+2] *= s
        return m
    h.intra = f(h.intra, [0, 0, 0])
    for t in h.hopping:
        t.m = f(t.m, t.dir)
    return h


def _fu_kane_mele(bond, s):
    g = geometry.diamond_lattice_minimal()
    h = g.get_hamiltonian()
    h.add_kane_mele(0.05)
    return _scale_bond(h, _bonds(g)[bond], s)


def _fu_kane(h):
    """nu0;(nu1 nu2 nu3) from the inversion eigenvalues at the eight
    time-reversal-invariant momenta, one per Kramers pair"""
    g = h.geometry
    P = compile_symmetry(h, SymmetryOperation(-np.eye(3),
                                              center=(g.r[0] + g.r[1])/2.))
    assert P is not None  # inversion about the bond midpoint
    hk = h.get_hk_gen()
    delta = {}
    for n in np.ndindex(2, 2, 2):
        k = np.array(n)/2.
        e, v = np.linalg.eigh(np.array(hk(k)))
        U = v[:, e < 0]
        Pk, kp = P.orbital_operator(k)
        xi = np.real(np.linalg.eigvals(U.conj().T@Pk@U))
        delta[n] = (-1)**int(round(np.sum(xi < 0)/2))
    nu0 = int(np.prod(list(delta.values())) < 0)
    nus = tuple(int(np.prod([d for (n, d) in delta.items() if n[i] == 1]) < 0)
                for i in range(3))
    return (nu0, nus)


@pytest.mark.parametrize("bond", [0, 1, 2, 3])
@pytest.mark.parametrize("s", [0.7, 1.3])
def test_z2_indices_match_the_fu_kane_parities(bond, s, tmp_path,
                                               monkeypatch):
    """A weaker bond gives a weak topological insulator (nu0=0) and a
    stronger one a strong topological insulator (nu0=1), with a weak vector
    that depends on the bond, so every direction of the Brillouin zone is
    exercised"""
    monkeypatch.chdir(tmp_path)
    h = _fu_kane_mele(bond, s)
    assert h.has_time_reversal_symmetry()
    z2 = h.get_topological_invariant(nk=30, nt=30)
    assert z2 == _fu_kane(h)
    assert z2[0] == int(s > 1.)
    assert z2[1] != (0, 0, 0)


def test_a_dominant_bond_gives_a_band_insulator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _fu_kane_mele(3, 3.1)  # beyond 3t, the dimer limit
    assert topology.z2_invariant_3d(h, nk=30, nt=30) == (0, (0, 0, 0))
    assert _fu_kane(h) == (0, (0, 0, 0))


def _stacked_haldane(plane, tz=0.1):
    """Spinless Haldane layers, stacked with a vertical hopping tz. With
    plane=3 the layers span a1 and a2, with plane=1 they span a2 and a3"""
    g = geometry.honeycomb_lattice()
    stack = np.array([0., 0., 2.])
    if plane == 3:
        g.a3 = stack
    else:
        g.a1, g.a2, g.a3 = stack, np.array(g.a1), np.array(g.a2)
    g.dimensionality = 3

    def fun(r1, r2):
        dr = r1 - r2
        if abs(dr[2]) < 1e-4 and abs(np.linalg.norm(dr) - 1.) < 1e-4:
            return 1.0
        if abs(abs(dr[2]) - 2.) < 1e-4 and np.linalg.norm(dr[:2]) < 1e-4:
            return tz
        return 0.0
    h = g.get_hamiltonian(fun=fun, has_spin=False)
    h.add_haldane(0.1)
    return h


def test_chern_vector_of_stacked_chern_insulators(tmp_path, monkeypatch):
    """Each layer carries the Chern number of the two-dimensional model,
    and the vector points along the stacking, whichever lattice vector that
    is; it does not depend on the plane of the zone it is evaluated on"""
    monkeypatch.chdir(tmp_path)
    h2 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h2.add_haldane(0.1)
    c = round(h2.get_chern())
    assert c != 0
    h = _stacked_haldane(3)
    assert not h.has_time_reversal_symmetry()
    assert h.get_topological_invariant(nk=20, nt=40) == (0, 0, c)
    assert topology.chern_vector(h, nk=20, nt=40, kfix=0.5) == (0, 0, c)
    assert topology.chern_vector(_stacked_haldane(1), nk=20, nt=40) == \
        (c, 0, 0)


def test_three_dimensional_routines_refuse_two_dimensions(tmp_path,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="three-dimensional"):
        topology.z2_invariant_3d(h)
    with pytest.raises(ValueError, match="three-dimensional"):
        topology.chern_vector(h)
