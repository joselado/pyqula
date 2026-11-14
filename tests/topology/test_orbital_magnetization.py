"""The orbital magnetization of a finite flake, topologytk/orbitalmagnetization.py.

The Chebyshev route must give the trace of the eigenstates, the total must
not depend on the origin of the positions, time reversal must flip its
sign, and its derivative with the Fermi level in the bulk gap must be the
Streda slope C A/(2 pi), carried by the edge states.
"""
import numpy as np
import pytest

from pyqula import islands
from pyqula import geometry
from pyqula import topology


def _flake(n,t2,w=0.,seed=0):
    """Haldane flake (hexagonal island of the honeycomb lattice) with
    uniform onsite disorder of width w"""
    g = islands.get_geometry(name="honeycomb",n=n,nedges=6,clean=True)
    h = g.get_hamiltonian(has_spin=False)
    if t2!=0.: h.add_haldane(t2)
    if w>0.:
        rng = np.random.default_rng(seed)
        h.add_onsite(lambda r: w*(rng.random()-0.5))
    return h


FERMIS = np.array([-1.,-0.2,0.,0.3,1.5])


def test_chebyshev_expansion_matches_the_eigenstates():
    """With the exact trace and a resolution well below the temperature,
    the expansion gives sum_n f(E_n) <n|m_z|n> of a disordered flake"""
    h = _flake(4,0.2,w=0.5)
    ref = h.get_orbital_magnetization(fermi=FERMIS,T=0.1,mode="ED")
    m = h.get_orbital_magnetization(fermi=FERMIS,T=0.1,delta=0.01,
            trace="exact")
    assert np.max(np.abs(m-ref)) < 1e-3*np.max(np.abs(ref))
    # the random vectors are unbiased
    ms,es = h.get_orbital_magnetization(fermi=FERMIS,T=0.1,delta=0.01,
            ntries=20,seed=2,return_error=True)
    assert np.all(np.abs(ms-ref) < 4.*es)


def test_time_reversal_and_the_origin():
    """The flux of the Haldane coupling flips the magnetization, a real
    Hamiltonian has none, and a translation of the flake changes nothing,
    since Tr[f(H) v] = 0"""
    h = _flake(4,0.2,w=0.5)
    m = h.get_orbital_magnetization(fermi=FERMIS,T=0.1,mode="ED")
    assert np.max(np.abs(m)) > 1. # a real signal
    mr = _flake(4,-0.2,w=0.5).get_orbital_magnetization(fermi=FERMIS,T=0.1,
            mode="ED")
    assert np.max(np.abs(m+mr)) < 1e-10*np.max(np.abs(m))
    m0 = _flake(4,0.,w=0.5).get_orbital_magnetization(fermi=FERMIS,T=0.1,
            mode="ED")
    assert np.max(np.abs(m0)) < 1e-10
    hs = h.copy()
    hs.geometry.r = hs.geometry.r + np.array([3.7,-1.2,0.])
    hs.geometry.r2xyz()
    ms = hs.get_orbital_magnetization(fermi=FERMIS,T=0.1,mode="ED")
    assert np.max(np.abs(ms-m)) < 1e-10*np.max(np.abs(m))


@pytest.mark.parametrize("t2",[0.2,-0.2])
def test_streda_slope_is_the_chern_number(t2,tmp_path,monkeypatch):
    """In the bulk gap, dM/dE_F = C A/(2 pi) with C the Chern number of the
    periodic model, the Maxwell relation dM/dmu = dN/dB of the Streda
    formula. The edge states carry it, and the area they enclose is smaller
    than the flake by a strip along the edge, so on 600 sites the slope is
    0.91 of its limit, and closer to it on a larger flake"""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    hp = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    hp.add_haldane(t2)
    c = round(topology.chern(hp,nk=20))
    assert c != 0
    efs = np.linspace(-0.3,0.3,7)
    ratios = []
    for n in [7,10]:
        h = _flake(n,t2)
        area = len(h.geometry.r)*3.*np.sqrt(3.)/4. # area per site
        m = h.get_orbital_magnetization(fermi=efs,T=0.02,mode="ED")
        ratios.append(np.polyfit(efs,m,1)[0]/(c*area/(2.*np.pi)))
    assert 0.85 < ratios[0] < ratios[1] < 1.


def test_unsupported_options_raise():
    h = _flake(3,0.2)
    with pytest.raises(ValueError,match="ED"):
        h.get_orbital_magnetization(mode="dense")
    hp = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError,match="flake"):
        hp.get_orbital_magnetization()
    hn = h.copy()
    hn.turn_nambu()
    with pytest.raises(NotImplementedError,match="Nambu"):
        hn.get_orbital_magnetization()
