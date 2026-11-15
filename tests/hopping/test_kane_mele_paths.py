"""The Kane-Mele and Haldane couplings average the orientation over every
path of a bond.

The coupling of the bond r_i -> r_j is i t s.(d1 x d2), with d1 = r_m - r_i
and d2 = r_j - r_m the two first-neighbor bonds through an intermediate
site r_m. On the honeycomb and the kagome lattices every bond has at most
one such path. On the triangular lattice the bonds at distance 1 and
sqrt(3) have two, with opposite orientations, and the code took the first
one found, so its sign followed the order of the candidates. The
maintainer chose the average over every path: one path on the honeycomb
and the kagome lattices, as before, and zero on the triangular lattice.

The reference below builds the coupling by hand from that definition,
looking for the intermediate sites in a large box of cells, and counts
the paths of every bond: one on the honeycomb and the kagome lattices, so
the average is the path the code found before.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry


_s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
      np.array([[1, 0], [0, -1]])]


def dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def reference(g, t, k, spinful, ncell=3):
    """The Kane-Mele (spinful) or Haldane (spinless) coupling at the
    momentum k, i t (d1 x d2) averaged over the paths, between every site
    of the cell 0 and every site of the cell R closer than sqrt(4.1).
    Returns the matrix and the largest number of paths of one bond, the
    pair of a site with itself left out"""
    r = np.array(g.r)
    n = len(r)
    cells = [(i, j) for i in range(-ncell, ncell+1)
             for j in range(-ncell, ncell+1)]
    rm = np.array([ri + c[0]*g.a1 + c[1]*g.a2 for c in cells for ri in r])
    ns = 2 if spinful else 1
    H = np.zeros((ns*n, ns*n), dtype=complex)
    most = 0
    for c in cells:
        phase = np.exp(2j*np.pi*(k[0]*c[0] + k[1]*c[1]))
        for i in range(n):
            for j in range(n):
                rj = r[j] + c[0]*g.a1 + c[1]*g.a2
                dr = rj - r[i]
                if dr@dr >= 4.1: continue
                d1 = rm - r[i]
                d2 = rj - rm
                on = (np.abs(np.sum(d1*d1, axis=1) - 1.) < 1e-5) & \
                     (np.abs(np.sum(d2*d2, axis=1) - 1.) < 1e-5)
                if not np.any(on): continue
                cr = np.cross(d1[on], d2[on])
                if dr@dr > 1e-8: # every path of a bond, not of a site
                    most = max(most, int(np.sum(on)))
                v = np.mean(cr, axis=0)
                if spinful:
                    sv = sum(v[a]*_s[a] for a in range(3))
                    H[2*i:2*i+2, 2*j:2*j+2] += 1j*t*sv*phase
                else:
                    H[i, j] += 1j*t*v[2]*phase
    return H, most


def added(h, add, k):
    """The coupling that add(h) puts in h, at k"""
    h1 = h.copy()
    add(h1)
    return dense(h1.get_hk_gen()(k)) - dense(h.get_hk_gen()(k))


LATTICES = {"honeycomb": geometry.honeycomb_lattice,
            "kagome": geometry.kagome_lattice,
            "triangular": geometry.triangular_lattice}

K = np.array([0.13, 0.29, 0.])


@pytest.mark.parametrize("name", list(LATTICES))
def test_kane_mele_is_the_average_over_the_paths(name):
    g = LATTICES[name]()
    h = g.get_hamiltonian(has_spin=True)
    dk = added(h, lambda x: x.add_kane_mele(0.1), K)
    H, most = reference(g, 0.1, K, spinful=True)
    if name == "triangular": assert most == 2 # two opposite paths
    else: assert most == 1 # one path, so as before the average
    e = eigh(dk, eigvals_only=True)
    e0 = eigh(H, eigvals_only=True)
    assert np.max(np.abs(e - e0)) < 1e-10


@pytest.mark.parametrize("name", list(LATTICES))
def test_haldane_is_the_average_over_the_paths(name):
    g = LATTICES[name]()
    h = g.get_hamiltonian(has_spin=False)
    dk = added(h, lambda x: x.add_haldane(0.1), K)
    H, most = reference(g, 0.1, K, spinful=False)
    e = eigh(dk, eigvals_only=True)
    e0 = eigh(H, eigvals_only=True)
    assert np.max(np.abs(e - e0)) < 1e-10


@pytest.mark.parametrize("add", [lambda x: x.add_kane_mele(0.1),
                                 lambda x: x.add_haldane(0.1)])
def test_no_coupling_on_the_triangular_lattice(add):
    """The two paths of every bond cancel, so nothing is added. Before, the
    first path found was kept, and the coupling was there"""
    h = geometry.triangular_lattice().get_hamiltonian()
    assert np.max(np.abs(added(h, add, K))) < 1e-12


def test_triangular_supercell_is_the_unit_cell():
    """The supercell of two cells along a1 has, at kS, the spectrum of the
    unit cell at the two momenta that fold onto it, whatever the order of
    the candidates for the intermediate site is"""
    M = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    h0 = geometry.triangular_lattice().get_hamiltonian()
    hs = h0.get_supercell(M)
    h0.add_kane_mele(0.1)
    hs.add_kane_mele(0.1)
    kS = np.array([0.37, 0.21, 0.])
    es = eigh(dense(hs.get_hk_gen()(kS)), eigvals_only=True)
    e0 = np.sort(np.concatenate([eigh(dense(h0.get_hk_gen()(
        [(kS[0] + j)/2., kS[1], 0.])), eigvals_only=True) for j in range(2)]))
    assert len(es) == len(e0)
    assert np.max(np.abs(es - e0)) < 1e-10
