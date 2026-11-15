"""The Kane-Mele and Haldane couplings of a multicell Hamiltonian reach
every cell a second neighbor sits in, whatever the choice of the cell.

The sheared cell a1' = a1, a2' = a2 + 6 a1 of the honeycomb lattice is the
same crystal with the same two sites (M has determinant 1), so at a
momentum kS its spectrum must be the one of the unit cell at M^-1 kS. Its
second neighbors sit in cells as far as (-6,1), and the couplings were
built on a fixed box of cells, two cells for Kane-Mele and int(10/|a1|)=5
for Haldane, so 4 and 2 of the 6 second-neighbor bonds were lost.
"""

import numpy as np
from scipy.linalg import eigh

from pyqula import geometry


SHEAR6 = [[1, 0, 0], [6, 1, 0], [0, 0, 1]]


def dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def same_spectrum(h1, hs, M, kS=(0.13, 0.37, 0.)):
    kS = np.array(kS)
    p = np.linalg.inv(np.array(M, dtype=float))@kS # the momentum of h1
    e1 = eigh(dense(h1.get_hk_gen()(p)), eigvals_only=True)
    es = eigh(dense(hs.get_hk_gen()(kS)), eigvals_only=True)
    assert len(e1) == len(es)
    return np.max(np.abs(e1 - es))


def test_kane_mele_on_a_sheared_cell():
    h0 = geometry.honeycomb_lattice().get_hamiltonian()
    h0.add_sublattice_imbalance(0.2)
    h0.turn_multicell()
    h1 = h0.copy()
    h1.add_kane_mele(0.05) # on the unit cell
    hs = h0.get_supercell(SHEAR6)
    hs.add_kane_mele(0.05) # on the sheared cell
    assert same_spectrum(h1, hs, SHEAR6) < 1e-10


def test_haldane_on_a_sheared_cell():
    h0 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h0.add_sublattice_imbalance(0.3)
    h0.turn_multicell()
    h1 = h0.copy()
    h1.add_haldane(0.1) # on the unit cell
    hs = h0.get_supercell(SHEAR6)
    hs.add_haldane(0.1) # on the sheared cell
    assert same_spectrum(h1, hs, SHEAR6) < 1e-10


def test_kane_mele_tolerance_reaches_every_bond():
    """add_kane_mele(t,tol=...) looks for the intermediate site of a
    second-neighbor path within tol of the first-neighbor distance 1. On
    the single-cell storage of a 2D Hamiltonian the tolerance was passed to
    the cell and to tx and ty but not to txy and txmy, so on a lattice
    stretched by 0.2 percent, with tol=1e-2, the second neighbors at a1+a2
    were dropped. The multicell storage passes it everywhere, and the two
    storages must give the same spectrum"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    g = h.geometry
    s = 1.002 # stretch, first-neighbor distance 1.002
    g.r = g.r*s
    g.r2xyz()
    g.a1 = g.a1*s
    g.a2 = g.a2*s
    hA = h.copy()
    hA.add_kane_mele(0.05, tol=1e-2) # single-cell storage
    hB = h.copy()
    hB.turn_multicell()
    hB.add_kane_mele(0.05, tol=1e-2) # multicell storage
    assert not hA.is_multicell
    # the bond at a1+a2 enters as 2i sin(2 pi (k1+k2)), so k1+k2 is not 1/2
    k = np.array([0.13, 0.29, 0.])
    eA = eigh(dense(hA.get_hk_gen()(k)), eigvals_only=True)
    eB = eigh(dense(hB.get_hk_gen()(k)), eigvals_only=True)
    e0 = eigh(dense(h.get_hk_gen()(k)), eigvals_only=True)
    assert np.max(np.abs(eB - e0)) > 1e-3 # the coupling is there
    assert np.max(np.abs(eA - eB)) < 1e-10
