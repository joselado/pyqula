"""The Bott index and the spin Bott index of a periodic supercell at k=0
(topologytk/bott.py) against the k-space Chern and spin Chern numbers,
which are computed by an independent route (a Berry curvature on a k-mesh
of the primitive cell)"""
import numpy as np
import pytest
from pyqula import geometry


def haldane(t2,mass=0.,swap=False):
    g = geometry.honeycomb_lattice()
    if swap: g.a1,g.a2 = g.a2.copy(),g.a1.copy()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(mass)
    return h


def kane_mele(soc,rashba,mass):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(soc)
    h.add_rashba(rashba)
    h.add_sublattice_imbalance(mass)
    return h


@pytest.mark.parametrize("t2,mass",[(0.1,0.),(-0.1,0.),(0.1,0.3),(0.1,1.0),
                                    (-0.1,1.0)])
def test_bott_index_equals_the_chern_number(t2,mass):
    h = haldane(t2,mass)
    c = h.get_chern(nk=30)
    assert abs(c-np.round(c))<1e-6
    for n in (4,7): # an integer that does not depend on the size
        assert abs(h.get_supercell(n).get_bott_index()-np.round(c))<1e-8


@pytest.mark.parametrize("t2",[0.1,-0.1])
def test_bott_index_follows_the_order_of_the_lattice_vectors(t2):
    """The sign follows a1 and a2, as get_chern does: both flip together"""
    for swap in (False,True):
        h = haldane(t2,swap=swap)
        c = h.get_chern(nk=30)
        assert abs(h.get_supercell(5).get_bott_index()-c)<1e-6
    b = haldane(t2).get_supercell(5).get_bott_index()
    bs = haldane(t2,swap=True).get_supercell(5).get_bott_index()
    assert abs(b+bs)<1e-8 and abs(abs(b)-1.)<1e-8


def test_bott_index_does_not_depend_on_the_origin():
    """A rigid shift of every site multiplies U and V by phases"""
    h = haldane(0.1).get_supercell(5)
    b0 = h.get_bott_index()
    h.geometry.r = np.array(h.geometry.r)+np.array([0.37,-1.21,0.])
    h.geometry.r2xyz()
    assert abs(h.get_bott_index()-b0)<1e-10


@pytest.mark.parametrize("seed",[0,1,2])
def test_bott_index_of_a_disordered_chern_insulator(seed):
    """Anderson disorder well below the gap keeps B=1 in every sample,
    a number no k-space routine can give, since the sample has no
    translation symmetry"""
    rng = np.random.default_rng(seed)
    h = haldane(0.15).get_supercell(8)
    w = 1.0 # disorder strength, against a clean gap of 1.35
    h.add_onsite(w*(rng.random(len(h.geometry.r))-0.5)) # one value per site
    assert abs(h.get_bott_index()-1.)<1e-8


def test_bott_index_vanishes_in_the_anderson_insulator():
    """Disorder much larger than the gap localizes every state, B=0, so the
    test above can fail"""
    rng = np.random.default_rng(0)
    h = haldane(0.15).get_supercell(8)
    h.add_onsite(16.*(rng.random(len(h.geometry.r))-0.5))
    assert abs(h.get_bott_index())<1e-6


@pytest.mark.parametrize("soc,rashba,mass",[(0.1,0.,0.1),(0.1,0.05,0.1),
                                            (0.2,0.2,0.2),(0.1,0.05,0.8)])
def test_spin_bott_index_equals_the_spin_chern_number(soc,rashba,mass):
    h = kane_mele(soc,rashba,mass)
    cs = h.get_spin_chern(nk=30,write=False)
    assert abs(cs-np.round(cs))<1e-6
    for n in (4,6):
        hs = h.get_supercell(n)
        assert abs(hs.get_spin_bott_index()-np.round(cs))<1e-8
        assert abs(hs.get_bott_index())<1e-8 # time reversal, B=0


def test_bott_index_refuses_what_it_cannot_compute():
    h = haldane(0.1)
    with pytest.raises(ValueError,match="spinful"):
        h.get_supercell(3).get_spin_bott_index() # spinless
    with pytest.raises(ValueError,match="dimensionality 1"):
        geometry.chain().get_hamiltonian().get_bott_index()
    with pytest.raises(ValueError,match="projector on the occupied states"):
        geometry.honeycomb_lattice().get_hamiltonian(
                has_spin=False).get_supercell(3).get_bott_index() # Dirac points
