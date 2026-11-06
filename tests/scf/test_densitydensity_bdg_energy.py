"""The total energy of a Bogoliubov-de Gennes mean field (attractive onsite
U, s-wave) from the three density-matrix integrations of Vinteraction,
"ed", "triangle" and "qtci". The paired state is gapped, so on the same
mesh the three must agree: exactly for the triangle (every triangle is
full or empty), to the interpolation tolerance for qtci. With mu=None the
Fermi level comes from the normal state, and the energy adds back mu times
the requested charge, so it is the grand potential Omega = <H - mu N> plus
a constant: it is stationary in the pairing amplitude at fixed mu, and it
obeys Hellmann-Feynman in U. Before this was settled qtci added mu times
the charge of the paired state instead, which differs from "ed" by 1.9e-2
here."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.superconductivity import get_eh_sector
from pyqula.scftk.densitydensity import Vinteraction

U = -2.0
FILLING = 0.3
NK = 16 # a power of two, so qtci runs on the same 16x16 mesh
MU = -1.68 # the fixed chemical potential, close to the one of the filling


def _h0():
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    return h


def _scf(integration, mu=None, U=U):
    np.random.seed(1)
    scf = Vinteraction(_h0(), U=U, filling=FILLING, mu=mu, mf="swave",
            nk=NK, integration=integration, solver="broyden_mixing",
            maxerror=1e-10, load_mf=False, write=False, verbose=0)
    assert scf.converged
    return scf


def _expectations(ht, hfree, nk=NK):
    """Expectation values in the ground state of the trial BdG Hamiltonian
    ht, on the uniform mesh: <H_free> with
    <(1/2) Psi^dag A Psi + (1/2) Tr a_e> = mean_k [Tr(P_k A_k) + Tr a_e(k)]/2,
    P_k the projector on the negative-energy states, and the onsite
    occupations of P averaged over k"""
    ks = hfree.geometry.get_kmesh(nk=nk)
    ft, f0 = ht.get_hk_gen(), hfree.get_hk_gen()
    n = ht.intra.shape[0]
    kin, pm = 0., np.zeros((n, n), dtype=np.complex128)
    for k in ks:
        e, u = np.linalg.eigh(ft(k))
        uo = u[:, e < 0.]
        p = uo@uo.conj().T # P[a,b] = <Psi^dag_b Psi_a>
        a = f0(k)
        tre = np.trace(get_eh_sector(a, i=0, j=0)).real
        kin += (np.trace(p@a).real+tre)/2.
        pm += p
    return kin/len(ks), pm/len(ks)


def _omega(ht, hfree):
    """Grand potential of the ground state of ht, written out by hand:
    <H_free - mu N> + U (n_up n_dn - |<c^dag_up c_dn>|^2 + |F|^2), with the
    Wick contractions of U n_up n_dn and F = <c^dag_up c^dag_dn>. The two
    entries of the electron-hole block are F and -F, and the same-spin
    ones vanish, so sum |block|^2 = 2 |F|^2. Returns Omega and the charge"""
    kin, pm = _expectations(ht, hfree)
    pe = get_eh_sector(pm.T, i=0, j=0) # <c^dag_i c_j>
    pa = get_eh_sector(pm.T, i=0, j=1)
    nup, ndn, m = pe[0, 0].real, pe[1, 1].real, pe[0, 1]
    f2 = np.sum(np.abs(pa)**2)/2.
    return kin+U*(nup*ndn-abs(m)**2+f2), nup+ndn


def _free(mu):
    h = _h0()
    h.shift_fermi(-mu)
    return h


def test_bdg_total_energy_is_the_same_for_the_three_integrations_at_fixed_mu():
    """At a fixed mu there is no Fermi search, and the gapped paired state
    is integrated exactly by the triangles of the same mesh"""
    e = {i: _scf(i, mu=MU).total_energy for i in ("ed", "triangle", "qtci")}
    assert abs(e["triangle"]-e["ed"]) < 1e-8
    assert abs(e["qtci"]-e["ed"]) < 1e-6


def test_bdg_total_energy_is_the_same_for_the_three_integrations_at_a_filling():
    """With mu=None, "ed" counts the eigenvalues of the normal state on the
    mesh while "triangle" and "qtci" integrate them on the triangles, so the
    Fermi levels differ by 3e-2 at nk=16 and the energies by 2e-4, which is
    second order (the energy is stationary in mu up to N - N0). The two
    triangle paths share their Fermi search and agree to the interpolation
    tolerance. Adding mu times the charge of the paired state instead of
    the requested one moved qtci by 1.9e-2"""
    e = {i: _scf(i).total_energy for i in ("ed", "triangle", "qtci")}
    assert abs(e["triangle"]-e["ed"]) < 1e-3
    assert abs(e["qtci"]-e["triangle"]) < 1e-6


@pytest.mark.parametrize("integration", ["ed", "triangle", "qtci"])
def test_bdg_total_energy_is_the_grand_potential_plus_mu_times_the_filling(
        integration):
    """The reported energy at mu=None is Omega at the converged Fermi level
    plus mu times the requested charge, filling times the number of
    electron states, and not mu times the charge of the paired state"""
    scf = _scf(integration)
    h = scf.hamiltonian
    om, charge = _omega(h, _free(h.fermi))
    assert abs(charge-2*FILLING) > 1e-3 # the two charges do differ
    tol = 1e-6 if integration=="qtci" else 1e-9
    assert abs(scf.total_energy-(om+h.fermi*2*FILLING)) < tol


def test_bdg_total_energy_is_stationary_in_the_pairing_at_fixed_mu():
    """Scale the pairing of the converged mean field by 1+x and recompute
    the grand potential at the same mu: the first-order change vanishes,
    while the one of <H> = Omega + mu N does not, since N moves with the
    pairing at fixed mu. The reported energy is the stationary value"""
    scf = _scf("ed", mu=MU)
    h = scf.hamiltonian
    hfree = _free(MU)
    ha = h.get_anomalous_hamiltonian()
    x = 1e-3
    om, en = [], []
    for s in (-x, 0., x):
        o, n = _omega(h+s*ha, hfree)
        om.append(o)
        en.append(o+MU*n)
    assert abs(om[1]-scf.total_energy) < 1e-9
    d_om = (om[2]-om[0])/(2*x)
    d_en = (en[2]-en[0])/(2*x)
    assert abs(d_om) < 1e-6
    assert abs(d_en) > 1e-3 # <H> is not stationary at fixed mu (8e-3)
    assert om[0] > om[1] and om[2] > om[1] # and it is a minimum


def test_bdg_total_energy_obeys_hellmann_feynman_at_fixed_mu():
    """dOmega/dU at fixed mu is <n_up n_dn>, which is only true when the
    energy is stationary in the mean fields"""
    dU = 1e-3
    ep = _scf("ed", mu=MU, U=U+dU).total_energy
    em = _scf("ed", mu=MU, U=U-dU).total_energy
    scf = _scf("ed", mu=MU)
    kin, pm = _expectations(scf.hamiltonian, _free(MU))
    pe = get_eh_sector(pm.T, i=0, j=0)
    pa = get_eh_sector(pm.T, i=0, j=1)
    nn = pe[0, 0].real*pe[1, 1].real-abs(pe[0, 1])**2+np.sum(np.abs(pa)**2)/2.
    assert abs((ep-em)/(2*dU)-nn) < 1e-6
