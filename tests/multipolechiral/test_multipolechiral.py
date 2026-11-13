"""Multipole chiral number (Benalcazar and Cerjan, arXiv:2109.06892) of the
BBH quadrupole model: an integer that changes across the dimerization
transition and counts the sublattice-resolved zero modes at a corner of the
open flake"""
import numpy as np
import pytest
from pyqula import geometry, specialhamiltonian, algebra
from pyqula.topologytk import multipolechiral as mc

L = 10 # unit cells of the sample along each direction


def sample(delta,cells=(L,L)):
    """Supercell of the BBH model and its chiral operator, the sublattice
    (-1)^(x+y) of the square lattice with the first site on A"""
    hs = specialhamiltonian.square_2OTI(delta).get_supercell(cells)
    r = hs.geometry.r-hs.geometry.r[0]
    S = np.diag((-1.)**np.round(r[:,0]+r[:,1]))
    return hs,S


def corner_modes(hs,S,cells=(L,L),tol=1e-3):
    """Zero modes of the open flake in each corner quadrant, those on A
    minus those on B, keyed by the corner (0,0), (L,0), (0,L) and (L,L)"""
    m = np.array(algebra.todense(hs.intra),dtype=complex)
    (e,v) = np.linalg.eigh(m)
    z = np.abs(e)<tol
    w = np.real(np.diag(v[:,z]@np.conjugate(v[:,z].T)))*np.diag(S).real
    idx = mc.cell_indices(hs,cells)
    lowx,lowy = idx[:,0]<cells[0]/2,idx[:,1]<cells[1]/2
    return {"00":np.sum(w[lowx&lowy]),"L0":np.sum(w[~lowx&lowy]),
            "0L":np.sum(w[lowx&~lowy]),"LL":np.sum(w[~lowx&~lowy])}


@pytest.mark.parametrize("boundary",["periodic","open"])
@pytest.mark.parametrize("delta,expected",[(-0.5,0),(-0.2,0),(0.5,1)])
def test_transition(delta,expected,boundary):
    """0 when the hopping inside the cell dominates, 1 when the hopping
    between cells does"""
    hs,S = sample(delta)
    n = hs.get_multipole_chiral_number(cells=(L,L),chiral=S,
            boundary=boundary)
    assert abs(n-expected)<1e-8


def test_corner_zero_modes():
    """Independent route: the zero modes at the corner (0,0) of the open
    flake, on A minus on B, are N_xy, and the adjacent corners have -N_xy"""
    hs,S = sample(0.5)
    c = corner_modes(hs,S)
    for boundary in ["periodic","open"]:
        n = hs.get_multipole_chiral_number(cells=(L,L),chiral=S,
                boundary=boundary)
        assert abs(c["00"]-n)<1e-2 and abs(c["LL"]-n)<1e-2
        assert abs(c["L0"]+n)<1e-2 and abs(c["0L"]+n)<1e-2
    # exchanging the sublattices flips the sign
    assert abs(hs.get_multipole_chiral_number(cells=(L,L),chiral=-S)+1)<1e-8


def test_rectangular_sample():
    """A sample with Lx different from Ly gives the same number and the
    same corner count"""
    cells = (8,12)
    hs,S = sample(0.5,cells=cells)
    c = corner_modes(hs,S,cells=cells)
    for boundary in ["periodic","open"]:
        n = hs.get_multipole_chiral_number(cells=cells,chiral=S,
                boundary=boundary)
        assert abs(n-1)<1e-8 and abs(c["00"]-1)<1e-2


def test_chiral_disorder():
    """Random complex hoppings on the bonds, which keep the chiral symmetry
    and keep the gap open, do not change the number"""
    hs,S = sample(0.5)
    rng = np.random.default_rng(7)
    m = np.array(algebra.todense(hs.intra),dtype=complex)
    r = rng.uniform(-1.,1.,m.shape)+1j*rng.uniform(-1.,1.,m.shape)
    r = np.triu(r,1)
    r = r+np.conjugate(r.T)
    hd = hs.copy()
    hd.intra = m+0.3*r*np.abs(m) # only on the existing bonds, A to B
    e = np.linalg.eigvalsh(algebra.todense(hd.get_hk_gen()([0.,0.,0.])))
    assert np.min(np.abs(e))>0.5 # smaller than the gap of 1.41
    for boundary in ["periodic","open"]:
        n = hd.get_multipole_chiral_number(cells=(L,L),chiral=S,
                boundary=boundary)
        assert abs(n-1)<1e-8
    assert abs(corner_modes(hd,S)["00"]-1)<1e-2


def test_stacked_pair():
    """Two copies of the model coupled by a Rashba term, which hops between
    the sublattices, give 2"""
    hs,S = sample(0.5)
    h2 = hs.copy()
    h2.turn_spinful() # two copies, one per spin
    h2.add_rashba(0.2) # couples them and keeps the chiral symmetry
    S2 = np.kron(S,np.identity(2))
    n = h2.get_multipole_chiral_number(cells=(L,L),chiral=S2)
    assert abs(n-2)<1e-8
    assert abs(corner_modes(h2,S2)["00"]-2)<1e-2


def test_default_chiral_has_a_fixed_sign():
    """get_sublattice() two-colors from a random site, so its global sign
    is random; chiral=None pins the first orbital on A and gives the same
    number as the explicit operator, whatever the coloring"""
    hs,S = sample(0.5,cells=(6,6))
    n0 = hs.get_multipole_chiral_number(cells=(6,6),chiral=S)
    assert abs(n0-1)<1e-8
    for seed in range(4):
        np.random.seed(seed)
        hs.geometry.get_sublattice()
        n = hs.get_multipole_chiral_number(cells=(6,6))
        assert abs(n-n0)<1e-8
        n = hs.get_multipole_chiral_number(cells=(6,6),chiral="sublattice")
        assert abs(n-n0)<1e-8


def test_refusals():
    hs,S = sample(0.5,cells=(4,4))
    h1 = hs.copy()
    h1.add_onsite(0.1) # breaks the chiral symmetry
    with pytest.raises(ValueError,match="anticommute"):
        h1.get_multipole_chiral_number(cells=(4,4),chiral=S)
    with pytest.raises(ValueError,match="two-dimensional"):
        geometry.chain().get_hamiltonian().get_multipole_chiral_number(
                cells=(4,4))
    with pytest.raises(ValueError,match="periodic"):
        hs.get_multipole_chiral_number(cells=(4,4),chiral=S,boundary="pbc")
    with pytest.raises(ValueError,match="cells"):
        hs.get_multipole_chiral_number(chiral=S)
    with pytest.raises(ValueError,match="unit cells"):
        hs.get_multipole_chiral_number(cells=(3,4),chiral=S)
    with pytest.raises(ValueError,match="sublattice"):
        hs.get_multipole_chiral_number(cells=(4,4)) # no sublattice labels
    with pytest.raises(TypeError,match="Operator"):
        hs.get_multipole_chiral_number(cells=(4,4),chiral=lambda v: v)
    h0,S0 = sample(0.,cells=(4,4)) # at the transition, gapless on the torus
    with pytest.raises(ValueError,match="gap is closed"):
        h0.get_multipole_chiral_number(cells=(4,4),chiral=S0)
