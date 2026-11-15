"""The range of a coupling given as a function, in the classical spin,
Ising and lattice-gas models.

A coupling Jij(r1,r2) is taken on every pair of sites within rcut, the
periodic images folded onto the cell, as the Vr of the mean field. A
crystal does not depend on the cell it is written in, so the energy per
site of a state must be the same in a cell and in a supercell, a function
must give what the same shells given as a list give, and on a large cell
every site of a uniform state must have the same local energy. The old
builder kept the neighboring cells only, so a coupling two cells away on a
small cell was dropped, and on a large cell the pairs depended on where a
site sits in the cell.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.classicalspin import SpinModel
from pyqula.latticegas import LatticeGas
from pyqula.latticeising import LatticeIsing


def tail(r1,r2):
    """A coupling exp(-d), zero at zero distance"""
    d = np.linalg.norm(np.array(r1)-np.array(r2))
    return np.exp(-d) if d>1e-6 else 0.


def second_neighbor(r1,r2):
    """A coupling 0.7 at distance 2 only"""
    d = np.linalg.norm(np.array(r1)-np.array(r2))
    return 0.7*float(abs(d-2.)<1e-6)


def build(model,g,**kwargs):
    """The model on g with a coupling, its energy for a state x of the
    sites (a pattern of +-1), and its local energies"""
    if model=="spin":
        m = SpinModel(g)
        m.add_heisenberg(**kwargs)
        def energy(x):
            m.theta = np.where(np.array(x)>0,0.,np.pi) ; m.phi = 0.*m.theta
            return m.energy()
    elif model=="ising":
        m = LatticeIsing(g)
        m.add_interaction(**kwargs)
        def energy(x):
            m.s = np.array(x,dtype=float)
            return m.get_energy()
    else:
        m = LatticeGas(g)
        m.add_interaction(**kwargs)
        def energy(x):
            m.den = (np.array(x,dtype=float)+1.)/2.
            return m.get_energy()
    return m,energy


MODELS = ["spin","ising","gas"]


@pytest.mark.parametrize("model",MODELS)
def test_a_function_gives_the_shells_of_a_list(model):
    """On the chain of one site, a second-neighbor coupling as a function
    and as the list [0,J2] give the same energy; the old builder dropped
    the function, two cells away"""
    g = geometry.chain()
    x = [1.]
    _,e1 = build(model,g,Jij=second_neighbor)
    _,e2 = build(model,g,Jij=[0.,0.7])
    assert abs(e1(x)-e2(x))<1e-10
    assert abs(e1(x))>0.1 # the coupling is there


@pytest.mark.parametrize("model",MODELS)
def test_cell_and_supercell_give_the_same_energy_per_site(model):
    """An alternating state on the chain written with two and with six
    sites per cell, with an exponential tail up to rcut=4"""
    gs = [geometry.chain().get_supercell(n) for n in (2,6)]
    es = []
    for g in gs:
        _,e = build(model,g,Jij=tail,rcut=4.)
        x = [(-1)**int(round(r[0])) for r in g.r] # alternating
        es.append(e(x)/len(g.r))
    assert abs(es[0]-es[1])<1e-10


@pytest.mark.parametrize("model",["ising","gas"])
def test_uniform_state_has_a_uniform_local_energy(model):
    """A 6x6 cell of the square lattice in a uniform state, with a tail
    exp(-d/3) up to the default rcut: every site sees the same pairs"""
    g = geometry.square_lattice().get_supercell(6)
    def slow(r1,r2):
        d = np.linalg.norm(np.array(r1)-np.array(r2))
        return np.exp(-d/3.) if d>1e-6 else 0.
    m,e = build(model,g,Jij=slow)
    e(np.ones(len(g.r)))
    loc = np.array(m.get_local_energy())
    assert np.max(np.abs(loc-loc[0]))<1e-10
    assert abs(loc[0])>1e-3


def test_uniform_spin_state_has_a_uniform_local_energy():
    """The same for the classical spins, on a smaller cell"""
    g = geometry.square_lattice().get_supercell(4)
    m,e = build("spin",g,Jij=tail,rcut=3.)
    e(np.ones(len(g.r)))
    loc = np.array(m.get_local_energy())
    assert np.max(np.abs(loc-loc[0]))<1e-6


def test_finite_system_keeps_every_pair_and_rcut_needs_a_function():
    """Without rcut a finite system keeps every pair, however far, and an
    rcut given with a list of shells is refused"""
    g = geometry.chain().get_supercell(8)
    g.dimensionality = 0 # an open chain of 8 sites
    m = LatticeIsing(g)
    m.add_interaction(Jij=lambda r1,r2: 1.0*(np.linalg.norm(
        np.array(r1)-np.array(r2))>1e-6))
    assert np.sum(np.abs(m.j)>0)==8*7 # every ordered pair
    with pytest.raises(ValueError):
        LatticeIsing(geometry.chain()).add_interaction(Jij=[1.],rcut=3.)
