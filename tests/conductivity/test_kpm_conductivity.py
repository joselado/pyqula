"""The Chebyshev (KPM) Kubo conductivity of conductivitytk/kpmkubo.py
against the band calculation of conductivity.py.

The KPM path builds the same velocity operator sparsely and expands the
spectral density of two currents in Chebyshev polynomials, so on a small
cell it must reproduce the Lehmann sum: exactly for the moments, to the
kernel line shape for the spectrum, and to the stochastic error bar for
random vectors. The Hall part is pinned against the Chern number.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula import topology
from pyqula.conductivitytk import kpmkubo
from pyqula.conductivitytk import kubo


def _disordered_cell(seed=1):
    """Gapped honeycomb (sublattice imbalance 0.5, gap 1) in a 3x3
    supercell, 18 orbitals, with random onsite disorder of width 0.3, below
    the gap"""
    rng = np.random.default_rng(seed)
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    h.intra = h.intra + np.diag(0.3*(rng.random(len(g.r))-0.5))
    return h


def _haldane(t2):
    """Spinless Haldane model with the Fermi level in the gap, as in
    test_optical_conductivity.py"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.shift_fermi(0.3)
    return h


def test_chebyshev_moments_match_the_diagonalization():
    """The double moments from the Chebyshev recursion with the exact
    trace equal Tr[v_a T_n(H) v_b T_m(H)] built from the eigenstates"""
    h = _disordered_cell()
    st = kpmkubo._Setup(h,nk=2,npol=60)
    mu = next(kpmkubo.moments(st,trace="exact"))
    ref = kpmkubo.exact_moments(st)
    assert np.max(np.abs(mu-ref)) < 1e-10*np.max(np.abs(ref))
    # a tiny memory budget stores a few polynomials at a time and runs the
    # left set again for every block, with the same result
    assert kpmkubo._block_size(18,18,2,st.M,1e-4) < st.M
    mu = next(kpmkubo.moments(st,trace="exact",memory=1e-4))
    assert np.max(np.abs(mu-ref)) < 1e-10*np.max(np.abs(ref))


def test_kpm_absorption_matches_the_band_calculation():
    """With the Lorentz kernel a transition gives a Lorentzian line of half
    width delta, the line shape of integration="grid". The two then agree
    to the kernel's deviation from a pure Lorentzian, a few percent of the
    peak at delta=0.05 on the same k-mesh"""
    h = _disordered_cell()
    ws = np.linspace(0.,4.,21)
    ws,s0 = h.get_optical_conductivity(energies=ws,nk=2,T=0.05,delta=0.05)
    ws,s1 = h.get_optical_conductivity(energies=ws,nk=2,T=0.05,delta=0.05,
            integration="kpm",kernel="lorentz",trace="exact")
    for a in range(2):
        ref = s0[:,a,a].real
        assert np.max(np.abs(s1[:,a,a].real-ref)) < 0.05*np.max(ref)
        assert np.max(np.abs(s1[:,a,a].imag)) < 1e-10*np.max(ref)


def test_random_vectors_agree_with_the_exact_trace():
    """The stochastic trace is unbiased: with fixed seeds the estimate lies
    within a few standard errors of the exact trace, and the error bar
    shrinks with the number of random vectors"""
    h = _disordered_cell()
    ws = np.linspace(0.5,2.5,9)
    kw = dict(energies=ws,nk=1,T=0.05,delta=0.2,npol=40)
    ws,ex = kpmkubo.optical_conductivity(h,trace="exact",**kw)
    errs = []
    for ntries in [4,16]:
        ws,s,e = kpmkubo.optical_conductivity(h,ntries=ntries,seed=7,
                return_error=True,**kw)
        d = np.abs(s[:,0,0]-ex[:,0,0])
        assert np.all(d < 4.*e[:,0,0] + 1e-3*np.max(ex[:,0,0].real))
        errs.append(np.mean(e[:,0,0]))
    assert errs[1] < errs[0]


@pytest.mark.parametrize("t2",[0.2,-0.2])
def test_kubo_bastin_hall_conductivity_is_the_chern_number(t2,tmp_path,
        monkeypatch):
    """The Kubo-Bastin DC tensor holds the Fermi sea, so in the gap of a
    Chern insulator sigma_xy 2 pi = -C, the sign of conductivity.py, for
    both signs of the flux, while sigma_xx vanishes. A scan over Fermi
    levels inside the gap gives the same plateau"""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    h = _haldane(t2)
    c = round(topology.chern(h,nk=20))
    assert c != 0
    s = h.get_dc_conductivity(fermi=[-0.2,0.,0.2],nk=12,T=0.01,npol=200,
            trace="exact")
    assert np.allclose(2.*np.pi*s[:,0,1],-c,atol=5e-3)
    assert np.allclose(s[:,1,0],-s[:,0,1],atol=1e-6)
    assert np.max(np.abs(s[:,0,0])) < 1e-4


def test_kpm_sum_rule_weight():
    """The diamagnetic weight Tr[f(H) d2H/dK2]/V from a single Chebyshev
    expansion equals kubo.sum_rule_weight on the same k-mesh"""
    h = _disordered_cell()
    W0 = kubo.sum_rule_weight(h,nk=2,T=0.05)
    W1 = kpmkubo.sum_rule_weight(h,nk=2,T=0.05,trace="exact")
    assert np.allclose(W0,W1,atol=1e-4*np.max(np.abs(W0)))


def test_f_sum_rule_of_a_finite_flake():
    """For a finite system the velocity is v = i[H,X] and the f-sum rule
    int Re sigma_xx dw = pi Tr[f(H) (-[[H,X],X])] holds exactly, at any
    size, so the integral of the KPM absorption over all frequencies must
    give pi times the weight. (On a k-mesh the same identity needs the
    k-integral converged, which a supercell at the Gamma point reaches
    only when it is large.)"""
    rng = np.random.default_rng(2)
    g = geometry.honeycomb_lattice().get_supercell(3)
    g.dimensionality = 0 # an 18-site flake with open edges
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    h.intra = h.intra + np.diag(0.3*(rng.random(len(g.r))-0.5))
    W = kpmkubo.sum_rule_weight(h,T=0.05,trace="exact")[0,0]
    # the same weight from the eigenstates
    H = np.array(h.intra)
    x = np.array(g.r)[:,0]
    tau = -H*(x[None,:]-x[:,None])**2
    es,U = np.linalg.eigh(H)
    f = 1./(1.+np.exp(es/0.05))
    Wex = np.sum(f*np.real(np.diag(U.conj().T@tau@U)))
    assert abs(W-Wex) < 1e-4*abs(Wex)
    ws = np.linspace(0.,7.,281)
    ws,s = kpmkubo.optical_conductivity(h,energies=ws,T=0.05,delta=0.1,
            trace="exact",component="xx")
    weight = 2.*np.trapezoid(s.real,ws) # Re sigma_xx is even in w
    assert abs(weight-np.pi*Wex) < 0.01*np.pi*Wex


def test_unsupported_options_raise():
    h = _disordered_cell()
    with pytest.raises(ValueError,match="jackson"):
        h.get_optical_conductivity(integration="kpm",kernel="gauss")
    with pytest.raises(ValueError,match="exact"):
        h.get_optical_conductivity(integration="kpm",trace="hutch")
    with pytest.raises(NotImplementedError):
        h.get_optical_conductivity(integration="kpm",intraband=False)
