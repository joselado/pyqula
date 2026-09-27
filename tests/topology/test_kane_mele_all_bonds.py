"""Second-neighbor couplings (h.add_soc/add_kane_mele, h.add_haldane) on
multicell Hamiltonians reach every second neighbor.

The multicell branches of kanemele.add_kane_mele and add_haldane_like used
to add the coupling only to the cells the Hamiltonian already had a hopping
to. A first-neighbor Hamiltonian of the diamond lattice has hoppings to six
cells, (+-1,0,0), (0,+-1,0) and (0,0,+-1), while the twelve second
neighbors also sit in +-(1,-1,0), +-(0,1,-1) and +-(1,0,-1), so half of the
spin-orbit bonds of the Fu-Kane-Mele model were missing: the bands were not
cubic symmetric, and at the W point the spin-orbit splitting was 0.3266
instead of 0.5333 for add_soc(0.1). The two-dimensional lattices, whose
Hamiltonians already hold every cell within one lattice vector, must come
out the same as before."""
import itertools

import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology

# primitive vectors of diamond_lattice_minimal, in units of the cubic cell
# (the geometry is rotated afterwards, which leaves fractional coordinates
# unchanged): a point P given in units of 2pi/a has fractional coordinates
# PRIMITIVE@P
PRIMITIVE = np.array([[-.5, .5, 0.], [0., .5, .5], [-.5, 0., .5]])

_s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
      np.array([[1, 0], [0, -1]])]
_bonds = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/4.


def _fu_kane_mele(P, lso, t=1.0):
    """Spectrum of Eq. (4) of Fu, Kane and Mele, PRL 98, 106803 (2007),
    arXiv:cond-mat/0607699, at the point P (units of 2pi/a, a the cubic
    cell): first-neighbor hopping t and i(8 lso/a^2) s.(d1 x d2) between
    second neighbors, with d1, d2 the two bonds traversed, summed over all
    twelve second neighbors"""
    K = 2*np.pi*np.array(P)
    H = np.zeros((4, 4), dtype=complex)  # A up, A dn, B up, B dn
    hab = t*np.sum(np.exp(1j*_bonds@K))
    H[0:2, 2:4] = hab*np.eye(2)
    H[2:4, 0:2] = np.conj(hab)*np.eye(2)
    for p, q in itertools.permutations(range(4), 2):
        for (sl, d1, d2) in [(0, _bonds[p], -_bonds[q]),
                             (2, -_bonds[p], _bonds[q])]:
            v = np.cross(d1, d2)
            sv = sum(vi*si for (vi, si) in zip(v, _s))
            H[sl:sl+2, sl:sl+2] += 1j*8*lso*sv*np.exp(1j*K@(d1 + d2))
    return np.linalg.eigvalsh(H)


def _spectrum(h, k):
    return np.linalg.eigvalsh(np.array(h.get_hk_gen()(k)))


def _diamond_soc(lam):
    h = geometry.diamond_lattice().get_hamiltonian()
    h.add_soc(lam)
    return h


def test_diamond_spin_orbit_is_cubic_symmetric():
    """A generic point and its images under the permutations of the cubic
    axes have the same spectrum"""
    h = _diamond_soc(0.1)
    P0 = np.array([0.13, 0.37, 0.71])
    es = [_spectrum(h, PRIMITIVE@P0[list(p)])
          for p in itertools.permutations(range(3))]
    for e in es:
        assert np.max(np.abs(e - es[0])) < 1e-10


@pytest.mark.parametrize("P", [[1., .5, 0.], [.75, .75, 0.], [1., .25, 0.],
                               [.5, .5, .5], [.13, .37, .71]])
def test_diamond_spin_orbit_is_the_fu_kane_mele_model(P):
    """add_soc(lam) is the model of Fu, Kane and Mele with
    lso = 2 lam/3: their 8/a^2 is 3/2 with the first-neighbor distance set
    to 1. At W the bands are +-8 lso = +-0.5333 for lam=0.1"""
    lam = 0.1
    e = _spectrum(_diamond_soc(lam), PRIMITIVE@np.array(P))
    assert np.max(np.abs(e - _fu_kane_mele(P, 2*lam/3))) < 1e-10
    if P == [1., .5, 0.]:
        assert np.allclose(np.abs(e), 16*lam/3)


def test_stacked_haldane_layers_are_the_two_dimensional_model():
    """Uncoupled Haldane layers stacked along z have, at every (k1,k2,k3),
    the bands of one layer at (k1,k2)"""
    g = geometry.honeycomb_lattice()
    h2 = g.get_hamiltonian(has_spin=False)
    h2.add_haldane(0.1)
    g3 = g.copy()
    g3.a3 = np.array([0., 0., 2.])
    g3.dimensionality = 3

    def fun(r1, r2):
        dr = r1 - r2
        if abs(dr[2]) < 1e-4 and abs(np.linalg.norm(dr) - 1.) < 1e-4:
            return 1.0
        return 0.0
    h3 = g3.get_hamiltonian(fun=fun, has_spin=False)
    h3.add_haldane(0.1)
    for k in [[0.1, 0.3, 0.2], [1./3., 2./3., 0.], [0.27, -0.41, 0.5]]:
        e2 = _spectrum(h2, [k[0], k[1], 0.])
        assert np.max(np.abs(_spectrum(h3, k) - e2)) < 1e-10


def _honeycomb(multicell, spinful):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=spinful)
    if multicell:
        h.turn_multicell()
    return h


@pytest.mark.parametrize("multicell", [False, True])
def test_kane_mele_gap_of_graphene_is_unchanged(multicell):
    """add_soc(lam) on the honeycomb lattice is Kane-Mele's
    lambda_SO=(sqrt3/2)lam, with a gap 6 sqrt3 lambda_SO = 9 lam at K, on
    the default Hamiltonian and on a multicell one alike"""
    h = _honeycomb(multicell, True)
    h.add_soc(0.1)
    e = _spectrum(h, [1./3., 1./3., 0.])  # the K point
    assert abs(np.min(e[e > 0]) - np.max(e[e < 0]) - 0.9) < 1e-10
    ref = _honeycomb(False, True)
    ref.add_soc(0.1)
    for k in [[0.1, 0.3, 0.], [0.27, -0.41, 0.]]:
        assert np.max(np.abs(_spectrum(h, k) - _spectrum(ref, k))) < 1e-10


@pytest.mark.parametrize("multicell", [False, True])
@pytest.mark.parametrize("m,chern", [(0.72, 1), (0.96, 0)])
def test_haldane_phase_boundary_is_unchanged(multicell, m, chern,
                                             tmp_path, monkeypatch):
    """The Haldane amplitude is (sqrt3/2)t2, so the phase boundary at
    phi=pi/2 sits at |m| = 4.5 t2 = 0.9 for t2=0.2"""
    monkeypatch.chdir(tmp_path)  # topology.chern writes files
    h = _honeycomb(multicell, False)
    h.add_haldane(0.2)
    h.add_sublattice_imbalance(m)
    c = topology.chern(h, nk=20)
    assert abs(abs(c) - chern) < 1e-3
