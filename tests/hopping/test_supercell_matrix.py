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


SHEARED = [[[1,0,0],[2,1,0],[0,0,1]], # a2' = 2a1 + a2, det 1
        [[1,0,0],[6,1,0],[0,0,1]], # a site in the primitive cell (5,1)
        [[2,0,0],[5,1,0],[0,0,1]]] # det 2, sites in the cells (4,1), (5,1)


@pytest.mark.parametrize("M",SHEARED)
def test_matrix_with_entries_larger_than_its_determinant(M):
    """A sheared supercell whose entries are larger than |det M| keeps all
    its sites: the geometry has |det M| times the sites of the unit cell,
    one copy of each, and the Hamiltonian has the folded spectrum"""
    g = geometry.honeycomb_lattice()
    N = int(round(abs(np.linalg.det(np.array(M,dtype=float)))))
    gs = g.get_supercell(M)
    assert len(gs.r)==N*len(g.r)
    primal = np.array(gs.supercell_primal_index)
    assert all(np.sum(primal==i)==N for i in range(len(g.r)))
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    h.add_sublattice_imbalance(0.3)
    check(h,M)


def test_matrix_a_rounding_error_from_integers():
    """A supercell matrix computed in floating point, with its entries a
    rounding error below the integers (1.9999999999998 for 2), is the
    supercell of those integers, and a matrix with an entry far from an
    integer is refused instead of truncated to another supercell"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.2)
    check(h,np.array(SQRT3,dtype=float)*(1.-1e-13))
    with pytest.raises(ValueError):
        h.get_supercell(np.array(SQRT3,dtype=float)+0.5*np.identity(3))


def test_geometry_rounds_a_float_matrix():
    """The geometry path rounds a float supercell matrix too: the cell and
    the recorded supercell_matrix, which the unfolding reads, are those of
    the integers, and an entry far from an integer is refused"""
    g = geometry.honeycomb_lattice()
    gs = g.get_supercell(np.array(SQRT3,dtype=float)*(1.-1e-13))
    assert np.array_equal(np.array(gs.supercell_matrix),np.array(SQRT3))
    assert len(gs.r)==3*len(g.r)
    with pytest.raises(ValueError,match="integer entries"):
        g.get_supercell(np.array(SQRT3,dtype=float)+0.5*np.identity(3))


def spectrum(h,k):
    return eigh(dense(h.get_hk_gen()(np.array(k,dtype=float))),
            eigvals_only=True)


def test_matrix_of_the_dimensionality_is_padded():
    """A dxd supercell matrix on a d-dimensional lattice is the 3x3 one
    with 1 along the missing directions, so both give the same supercell,
    and a matrix of another shape is refused with a message"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.2)
    h.add_sublattice_imbalance(0.2)
    hs2 = h.get_supercell([[2,1],[-1,1]])
    hs3 = check(h,SQRT3)
    assert hs2.intra.shape==hs3.intra.shape
    k = [0.23,0.41,0.]
    assert np.max(np.abs(spectrum(hs2,k)-spectrum(hs3,k)))<1e-10
    with pytest.raises(ValueError,match="supercell matrix"):
        h.get_supercell([[2,1,0],[-1,1,0]])
    h = geometry.chain().get_hamiltonian()
    h.add_exchange([0.,0.,0.2])
    hs1 = h.get_supercell([[3]])
    hs3 = check(h,[[3,0,0],[0,1,0],[0,0,1]])
    assert hs1.intra.shape==hs3.intra.shape
    k = [0.17,0.,0.]
    assert np.max(np.abs(spectrum(hs1,k)-spectrum(hs3,k)))<1e-10


def test_the_geometry_pads_a_matrix_of_its_dimensionality():
    """g.get_supercell with a dxd matrix is the supercell of the padded 3x3
    one, on the geometry path as on the Hamiltonian path, with or without
    the primitive geometry stored, and another shape is refused"""
    for g,M2 in [(geometry.honeycomb_lattice(),[[2,1],[-1,1]]),
            (geometry.triangular_lattice(),[[1,0],[6,1]]),
            (geometry.chain(),[[3]])]:
        d = g.dimensionality
        M3 = np.identity(3,dtype=int)
        M3[0:d,0:d] = M2
        for primal in [False,True]:
            g2 = g.get_supercell(M2,store_primal=primal)
            g3 = g.get_supercell(M3,store_primal=primal)
            assert np.array_equal(g2.r,g3.r)
            for a in ["a1","a2","a3"]:
                assert np.array_equal(getattr(g2,a),getattr(g3,a))
            assert (getattr(g2,"primal_geometry",None) is not None)==primal
        h = g.get_hamiltonian(has_spin=False)
        assert np.array_equal(h.get_supercell(M2).geometry.r,g3.r)
    with pytest.raises(ValueError,match="supercell matrix"):
        geometry.honeycomb_lattice().get_supercell([[2,1,0],[-1,1,0]])


def test_ncut_is_refused_for_a_matrix():
    """The repetitions take a range cut ncut, a matrix supercell keeps
    every hopping, so ncut with a matrix is refused with a message"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(TypeError,match="no meaning for a matrix supercell"):
        h.get_supercell(SQRT3,ncut=5)
    hs = h.get_supercell([2,2],ncut=5) # the repetitions keep the cut
    assert hs.intra.shape[0]==4*h.intra.shape[0]


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
