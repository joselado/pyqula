"""The Fermi search of the simplex integration (dmtk/simplexdm.py,
fermi_from_energies): a safeguarded Newton iteration on the charge, with
the simplex DOS as its derivative, against the bracketing root search on
the sum of the weights."""
import numpy as np
import pytest
from scipy.optimize import brentq

from pyqula import geometry
from pyqula.dmtk import simplexdm


def _charge(dim,nk,es,mu,T):
    """the charge as the sum of the weights the density matrix uses"""
    return float(np.sum(simplexdm.simplex_weights(dim,nk,es,mu=mu,T=T)))


def _brentq_reference(dim,nk,es,filling,T):
    """the bracketing search the Newton iteration replaced, on the sum of
    the weights: brentq on the whole band range, and the middle of the
    interval on which the charge is held to 1e-12 when there is one"""
    nb = es.shape[1]
    target = filling*nb
    emin,emax = float(np.min(es)),float(np.max(es))
    pad = 1e-6+50.*T
    def n(mu): return _charge(dim,nk,es,mu,T)
    tol = 1e-12*nb
    lo,hi = emin-pad,emax+pad
    xtol = 1e-14*max(1.,abs(emax)+abs(emin))
    mu0 = brentq(lambda mu: n(mu)-target,lo,hi,xtol=xtol)
    eps = 1e-8*max(1.,emax-emin)
    if abs(n(mu0-eps)-target)>tol or abs(n(mu0+eps)-target)>tol:
        return mu0
    m1 = brentq(lambda mu: n(mu)-(target-tol),lo,mu0,xtol=xtol)
    m2 = brentq(lambda mu: n(mu)-(target+tol),mu0,hi,xtol=xtol)
    return (m1+m2)/2.


def _energies(h,nk):
    return simplexdm._Mesh(h,nk).energies()


def _metal(name):
    if name=="chain": g = geometry.chain()
    elif name=="square": g = geometry.square_lattice()
    else: g = geometry.cubic_lattice()
    return g.get_hamiltonian(has_spin=False)


def _gapped(asymmetric):
    """honeycomb with a sublattice imbalance; the second-neighbor hopping
    breaks the symmetry E -> -E of the two bands"""
    tij = [1.,0.2] if asymmetric else None
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False,tij=tij)
    h.add_sublattice_imbalance(0.6 if asymmetric else 0.5)
    return h


@pytest.fixture
def counted(monkeypatch):
    """count the charge-and-DOS evaluations of a search"""
    n = [0]
    f = simplexdm._charge_and_dos
    def g(*args,**kwargs):
        n[0] += 1
        return f(*args,**kwargs)
    monkeypatch.setattr(simplexdm,"_charge_and_dos",g)
    return n


def _search(dim,nk,es,filling,T,counted):
    counted[0] = 0
    mu = simplexdm.fermi_from_energies(dim,nk,es,filling,T=T)
    return mu,counted[0]


@pytest.mark.parametrize("name,nk", [("chain",101),("square",32),("cubic",8)])
@pytest.mark.parametrize("T", [0.,1e-3,1e-2])
def test_metal_matches_the_bracketing_search(name,nk,T,counted):
    """In a metal the root is unique: the Newton iteration and brentq on the
    sum of the weights give the same level, the weights at that level hold
    the filling, and the search takes at most five evaluations"""
    h = _metal(name)
    dim = h.dimensionality
    es = _energies(h,nk)
    mu,nev = _search(dim,nk,es,0.3,T,counted)
    assert abs(mu-_brentq_reference(dim,nk,es,0.3,T))<1e-12
    assert abs(_charge(dim,nk,es,mu,T)-0.3*es.shape[1])<1e-12
    assert nev<=5


@pytest.mark.parametrize("asymmetric", [False,True])
@pytest.mark.parametrize("T", [0.,1e-3,1e-2])
def test_insulator_returns_the_middle_of_the_gap(asymmetric,T,counted):
    """With the filling in the gap the charge is exactly flat between the
    top of the valence vertices and the bottom of the conduction vertices
    (at T>0 beyond the cutoff of the Fermi function, which this gap
    exceeds), and the search returns the middle; the bracketing search
    lands somewhere in the gap"""
    h = _gapped(asymmetric)
    nk = 32
    es = _energies(h,nk)
    ev,ec = float(np.max(es[:,0])),float(np.min(es[:,1]))
    assert ec-ev>2.*simplexdm._tbreaks[-1]*1e-2 # a plateau at every T here
    mu,nev = _search(2,nk,es,0.5,T,counted)
    assert abs(mu-(ev+ec)/2.)<1e-12
    assert abs(_charge(2,nk,es,mu,T)-1.)<1e-12
    assert nev<=5
    # the bracketing search is not always the middle: here at T=0 the flat
    # charge is 1e-16 below the target, so brentq stops at the conduction
    # edge, and the check at mu0 +- eps sees the band and keeps it
    ref = _brentq_reference(2,nk,es,0.5,T)
    assert ev<=ref<=ec


@pytest.mark.parametrize("dim", [1,2,3])
@pytest.mark.parametrize("T", [0.,1e-2])
def test_dos_is_the_derivative_of_the_charge(dim,T):
    """The DOS of the kernel is dN/dmu, against a finite difference of the
    charge, with a flat band at mu so that its Fermi step counts at T>0"""
    nk = {1:40,2:12,3:6}[dim]
    rng = np.random.default_rng(dim)
    es = np.concatenate([rng.normal(size=(nk**dim,2)),
            np.full((nk**dim,1),0.01)],axis=1)
    for mu in (-0.3,0.,0.4):
        c,d = simplexdm.simplex_charge_dos(dim,nk,es,mu=mu,T=T)
        h = 1e-6
        fd = (_charge(dim,nk,es,mu+h,T)-_charge(dim,nk,es,mu-h,T))/(2.*h)
        assert abs(c-_charge(dim,nk,es,mu,T))<1e-12
        assert abs(d-fd)<1e-6*max(1.,abs(d))


def test_flat_band_at_the_filling():
    """A flat band at T=0 makes the charge jump, and with the filling inside
    the jump the level is the energy of the flat band, as the bracketing
    search finds it; at T>0 the step of the flat band is smooth, and both
    searches agree again"""
    nk = 16
    disp = _energies(_metal("square"),nk)
    es = np.concatenate([disp,np.full((nk*nk,1),-0.5)],axis=1)
    es.sort(axis=1)
    n_below = _charge(2,nk,es,-0.5-1e-9,0.)
    filling = (n_below+0.5)/es.shape[1] # in the middle of the jump
    mu = simplexdm.fermi_from_energies(2,nk,es,filling,T=0.)
    assert abs(mu+0.5)<1e-12
    for T in (1e-3,1e-2):
        mu = simplexdm.fermi_from_energies(2,nk,es,filling,T=T)
        assert abs(mu-_brentq_reference(2,nk,es,filling,T))<1e-12
