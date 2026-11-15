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


@pytest.mark.parametrize("model",MODELS)
def test_a_hopping_generator_gives_the_shells_of_a_list(model):
    """The same for a coupling given as a HoppingGenerator, here the
    neighbor-shell generator NNG: on the chain of one site its second
    shell is two cells away"""
    from pyqula.specialhopping import NNG
    g = geometry.chain()
    x = [1.]
    _,e1 = build(model,g,Jij=NNG(g,[0.,0.7]))
    _,e2 = build(model,g,Jij=[0.,0.7])
    assert abs(e1(x)-e2(x))<1e-10
    assert abs(e1(x))>0.1


@pytest.mark.parametrize("model",MODELS)
def test_a_coupling_with_no_pair_in_range_adds_nothing(model):
    """With rcut=0.5 on the chain of spacing 1 no pair is in range, so the
    coupling adds nothing and the energy is zero. The spin model used to
    crash there, when it stacked an empty list of exchange tensors next to
    the array of 3x3 tensors"""
    g = geometry.chain().get_supercell(2)
    _,e = build(model,g,Jij=tail,rcut=0.5)
    assert abs(e([1.,-1.]))<1e-12


@pytest.mark.parametrize("model",MODELS)
def test_a_hopping_generator_is_checked_as_a_function(model):
    """The coupling of a pair is one real number. A HoppingGenerator that
    is imaginary, or that couples a pair from one end only, raises as the
    same coupling written as a function does. The generator path had no
    check, and the models keep only the real part, so the imaginary
    coupling became no coupling and the one-sided one half of a bond"""
    from pyqula.specialhopping import NNG, HoppingGenerator
    g = geometry.chain().get_supercell(2)
    fm = NNG(g,[1.]).f # first neighbors
    imaginary = HoppingGenerator(lambda rs1,rs2: 1j*np.array(fm(rs1,rs2)))
    with pytest.raises(ValueError, match="Jij must be real"):
        build(model,g,Jij=imaginary)
    def forward(rs1,rs2): # from x to x+1 only
        return np.array([[1.*(abs(b[0]-a[0]-1.)<1e-6) for b in rs2]
                         for a in rs1])
    with pytest.raises(ValueError, match="Jij is not reciprocal"):
        build(model,g,Jij=HoppingGenerator(forward))
    _,e = build(model,g,Jij=NNG(g,[1.])) # a real reciprocal one is taken
    assert abs(e([1.,1.]))>0.1


def exp_with_onsite(r1,r2):
    """A coupling exp(-d), 1 at zero distance"""
    return np.exp(-np.linalg.norm(np.array(r1)-np.array(r2)))


def coulomb(r1,r2):
    """1/d, infinite at zero distance (numpy gives inf there)"""
    return 1./np.linalg.norm(np.array(r1)-np.array(r2))


@pytest.mark.parametrize("model",MODELS)
def test_the_pair_of_a_site_with_itself_is_dropped(model):
    """A coupling given as a function or as a generator is taken between
    pairs of sites, never for a site with itself. The value J(r_i,r_i)
    was kept: exp(-d) put J(0)=1 on every diagonal, a shift of mu in the
    gas, and a 1/r tail raised. The pair of a site with its own images
    stays: on the chain of one site with rcut=1.5 the only pairs are the
    two images at distance 1, so the energy of the uniform state is
    2*exp(-1) in absolute value, without the 1 of J(0)"""
    from pyqula.specialhopping import HoppingGenerator
    def exp_matrix(rs1,rs2):
        dr = np.array(rs1)[:,None,:]-np.array(rs2)[None,:,:]
        return np.exp(-np.linalg.norm(dr,axis=2))
    g = geometry.chain()
    for Jij in (exp_with_onsite,HoppingGenerator(exp_matrix)):
        _,e = build(model,g,Jij=Jij,rcut=1.5)
        assert abs(abs(e([1.]))-2.*np.exp(-1.))<1e-12
    # 1/r gives what the same tail with a zero at zero distance gives
    g = geometry.chain().get_supercell(2)
    x = [1.,1.]
    _,e1 = build(model,g,Jij=coulomb,rcut=3.)
    def soft(r1,r2): # 1/d between two sites, written zero at zero distance
        d = np.linalg.norm(np.array(r1)-np.array(r2))
        return 1./d if d>1e-6 else 0.
    _,e2 = build(model,g,Jij=soft,rcut=3.)
    assert abs(e1(x)-e2(x))<1e-12
    assert abs(e1(x))>0.1
