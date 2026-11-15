"""The supercell of a Hamiltonian along any integer matrix M,
h.get_supercell(M) with A_supercell = M A.

A supercell is the same crystal written with a larger cell, so at a Bloch
momentum kS of the supercell its spectrum must be the union of the spectra
of the unit cell at the N = |det M| momenta M^-1 (kS + q) that fold onto it.
This holds for any term of the unit cell (spin-orbit coupling, an exchange
field, complex hoppings, pairing), which is what a supercell built from the
geometry alone would miss.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry


def dense(m):
    return np.array(m.todense()) if hasattr(m,"todense") else np.array(m)


def folded(M,kS,dim):
    """The momenta of the unit cell that fold onto kS"""
    Minv = np.linalg.inv(np.array(M,dtype=float))
    out = []
    rng = range(-6,7)
    for j0 in rng:
        for j1 in (rng if dim>1 else [0]):
            for j2 in (rng if dim>2 else [0]):
                p = np.mod(Minv@(kS+np.array([j0,j1,j2],dtype=float)),1.)
                if not any(np.allclose(p,u,atol=1e-9) for u in out):
                    out.append(p)
    return out


def check(h,M,kS=(0.13,0.37,0.29)):
    hs = h.get_supercell(M)
    kS = np.array(kS)
    kS[h.dimensionality:] = 0.
    es = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    ps = folded(M,kS,h.dimensionality)
    N = int(round(abs(np.linalg.det(np.array(M,dtype=float)))))
    assert len(ps)==N
    assert hs.intra.shape[0]==N*h.intra.shape[0]
    e0 = np.sort(np.concatenate([eigh(dense(h.get_hk_gen()(p)),
        eigvals_only=True) for p in ps]))
    assert np.max(np.abs(es-e0))<1e-10
    return hs


SQRT3 = [[2,1,0],[-1,1,0],[0,0,1]]
SEVEN = [[3,1,0],[-1,2,0],[0,0,1]]
SHEAR = [[2,0,0],[1,3,0],[0,0,1]]


@pytest.mark.parametrize("M",[SQRT3,SEVEN,SHEAR])
def test_spinful_honeycomb_with_rashba_and_exchange(M):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0.1,0.2,0.3])
    h.add_sublattice_imbalance(0.2)
    check(h,M)


@pytest.mark.parametrize("M",[SQRT3,SEVEN])
def test_haldane_complex_second_neighbors(M):
    """Complex second-neighbor hoppings: the orientation of every bond
    matters, so a wrong direction of a hopping changes the spectrum; the
    sublattice imbalance breaks the inversion that would map k to -k and
    hide a reversed direction"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    h.add_sublattice_imbalance(0.3)
    check(h,M)


def test_nambu_pairing():
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_rashba(0.2)
    h.add_swave(0.3)
    check(h,SEVEN)


def test_chain_and_cubic():
    h = geometry.chain().get_hamiltonian()
    h.add_exchange([0.,0.,0.2])
    check(h,[[3,0,0],[0,1,0],[0,0,1]])
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(0.1)
    check(h,[[1,1,0],[0,1,1],[1,0,1]]) # two cells, along face diagonals


def test_same_as_the_supercell_of_the_geometry():
    """For a model the geometry builds by itself, the supercell of h and
    the Hamiltonian of the supercell geometry are the same matrices up to
    the order of the sites, so their spectra agree at every momentum"""
    g = geometry.honeycomb_lattice()
    hs1 = g.get_hamiltonian().get_supercell(SEVEN)
    hs2 = g.get_supercell(SEVEN).get_hamiltonian()
    k = np.array([0.31,0.17,0.])
    e1 = eigh(dense(hs1.get_hk_gen()(k)),eigvals_only=True)
    e2 = eigh(dense(hs2.get_hk_gen()(k)),eigvals_only=True)
    assert np.max(np.abs(e1-e2))<1e-10


def test_ultracell_of_a_matrix_supercell():
    """The ultracell of the sqrt3 x sqrt3 supercell of h, with a Kekule
    modulation of the hoppings, is that supercell with every state kept"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.2)
    hs = h.get_supercell(SQRT3)
    hs.add_onsite(lambda r: 0.2*np.cos(r[0]))
    u = h.get_ultracell(hs)
    kS = np.array([0.23,0.41,0.])
    e1 = eigh(u.get_hk_gen()(kS),eigvals_only=True)
    e2 = eigh(dense(hs.get_hk_gen()(kS)),eigvals_only=True)
    assert np.max(np.abs(e1-e2))<1e-11
