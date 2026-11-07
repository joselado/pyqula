"""The total energy of a Bogoliubov-de Gennes mean field (attractive onsite
U, s-wave) from the three density-matrix integrations of Vinteraction,
"ed", "triangle" and "qtci". The paired state is gapped, so on the same
mesh the three must agree: exactly for the triangle (every triangle is
full or empty), to the interpolation tolerance for qtci. At a fixed mu the
energy is the grand potential Omega = <H - mu N>: it is stationary in the
pairing amplitude, and it obeys Hellmann-Feynman in U. With mu=None the
Fermi level is found on the charge of the paired state
(scftk.superscf.BdGFermiLevel), so the paired state holds the requested
filling, and the energy Omega + mu N0 is the expectation value <H> at
that filling. Before, the Fermi level came from the normal state, and the
paired state held 0.606 electrons per cell instead of 0.600 here."""
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


def _scf(integration, mu=None, U=U, filling=FILLING):
    np.random.seed(1)
    scf = Vinteraction(_h0(), U=U, filling=filling, mu=mu, mf="swave",
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
    """With mu=None the Fermi level holds the charge of the paired state,
    which is gapped and so continuous in mu on the mesh of "ed" as on the
    triangles: the three find the same Fermi level and the same energy.
    With the Fermi level of the normal state, the staircase count of "ed"
    put it 3e-2 away from the triangle at nk=16, and the energies 2e-4"""
    scf = {i: _scf(i) for i in ("ed", "triangle", "qtci")}
    e = {i: s.total_energy for (i, s) in scf.items()}
    mu = {i: s.hamiltonian.fermi for (i, s) in scf.items()}
    assert abs(e["triangle"]-e["ed"]) < 1e-8
    assert abs(e["qtci"]-e["triangle"]) < 1e-6
    assert abs(mu["triangle"]-mu["ed"]) < 1e-8
    assert abs(mu["qtci"]-mu["triangle"]) < 1e-6


@pytest.mark.parametrize("filling", [0.3, 0.2])
@pytest.mark.parametrize("integration", ["ed", "triangle", "qtci"])
def test_bdg_mean_field_holds_the_filling_and_its_energy_is_h(integration,
        filling):
    """With mu=None the paired state holds the requested charge, filling
    times the number of electron states (2*filling here, spinful), and the
    reported energy is Omega at the converged Fermi level plus mu times
    that charge, which is now <H> = Omega + mu N, with Omega written out by
    hand from the projector on the mesh. h.get_filling() reads the same
    charge back"""
    scf = _scf(integration, filling=filling)
    h = scf.hamiltonian
    om, charge = _omega(h, _free(h.fermi))
    tol = 1e-6 if integration=="qtci" else 1e-8
    assert abs(charge-2*filling) < tol
    assert abs(scf.charge_error) < tol
    assert abs(scf.total_energy-(om+h.fermi*charge)) < tol
    assert abs(h.get_filling(nk=NK)-filling) < 1e-6


def test_bdg_mean_field_holds_the_filling_through_vjinteraction():
    """h.get_mean_field_hamiltonian of a spinful Hamiltonian goes through
    VJinteraction, which finds its Fermi level in the same way"""
    np.random.seed(1)
    hscf = _h0().get_mean_field_hamiltonian(U=U, filling=FILLING,
            mf="swave", nk=NK, maxerror=1e-9, verbose=0)
    assert abs(hscf.get_filling(nk=NK)-FILLING) < 1e-6
    ref = _scf("ed").hamiltonian # the same state through Vinteraction
    assert abs(hscf.fermi-ref.fermi) < 1e-6


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_bdg_mean_field_without_pairing_keeps_the_normal_fermi_level(
        integration):
    """A Nambu Hamiltonian with a repulsive U converges to a state with no
    pairing, where the paired charge is the normal one: the Fermi level is
    the one of the normal state, as before, and the energy is Omega at
    that level plus mu N0, the same number a run at that fixed mu gives"""
    np.random.seed(1)
    kw = dict(U=2.0, mf="ferro", nk=NK, integration=integration,
            solver="broyden_mixing", maxerror=1e-10, load_mf=False,
            write=False, verbose=0)
    scf = Vinteraction(_h0(), filling=FILLING, **kw)
    assert scf.converged
    h = scf.hamiltonian
    hn = h.copy()
    hn.shift_fermi(h.fermi) # undo the shift, the search is on h itself
    if integration=="triangle":
        from pyqula.dmtk import simplexdm
        mu0 = simplexdm.get_fermi4filling(hn, FILLING, nk=NK, T=1e-7)
    elif integration=="qtci":
        from pyqula.qtcitk.densitymatrix_qtci import get_fermi4filling_qtci
        mu0 = get_fermi4filling_qtci(hn, FILLING, nk=NK, T=1e-7)
    else: mu0 = hn.get_fermi4filling(FILLING, nk=NK, T=1e-7)
    assert abs(h.fermi-mu0) < 1e-10
    fixed = Vinteraction(_h0(), filling=FILLING, mu=h.fermi, mf=scf.mf, **{
        k: v for (k, v) in kw.items() if k!="mf"})
    assert abs(scf.total_energy-(fixed.total_energy+h.fermi*2*FILLING)) < 1e-9


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


def _swave(integration, U=-1.0, nk=32):
    np.random.seed(1)
    scf = Vinteraction(_h0(), U=U, filling=FILLING, mf="swave", nk=nk,
            integration=integration, solver="broyden_mixing",
            maxerror=1e-9 if integration!="qtci" else 1e-8, load_mf=False,
            write=False, verbose=0)
    assert scf.converged
    dm = scf.hamiltonian.get_density_matrix(ds=[(0,0,0)], nk=nk, T=1e-7)
    return np.max(np.abs(get_eh_sector(dm[(0,0,0)], i=0, j=1)))


def test_triangle_and_qtci_bdg_keep_the_pairing_of_ed():
    """For a BdG Hamiltonian the triangle density matrix is the step one of
    exact diagonalization on the same mesh, since the sorted bands are
    never partly occupied. Its Fermi level used to come from the triangle
    count of the normal state, a different number, and with U=-1 at
    filling 0.3 the triangle lost the pairing that "ed" finds. Now the
    three integrations give the same pairing on the same mesh"""
    a = {i: _swave(i) for i in ("ed", "triangle", "qtci")}
    assert a["ed"] > 1e-2
    assert abs(a["triangle"]-a["ed"]) < 1e-6
    assert abs(a["qtci"]-a["ed"]) < 1e-5


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_nambu_normal_state_has_the_magnetization_of_the_normal_state(
        integration):
    """A repulsive U from a ferromagnetic guess, with and without the Nambu
    basis: the Nambu state has no pairing, so it must be the state of the
    normal Hamiltonian. Before, the triangle put the Nambu state at
    another charge, where it was a ferromagnet (mz=0.006 at nk=64) while
    the normal one was not. The Nambu triangle and qtci are the step of
    exact diagonalization on the same mesh, so the reference is the normal
    state with "ed" at T=0; the continuous charge of the normal triangle is
    out of reach of a sorted BdG spectrum.

    At filling 0.35 the requested charge is 2867.2 electrons on the mesh,
    and a star of k-points of one spin sits at zero energy. The normal
    state at T=0 holds 2869.536 electrons: the Fermi search rounds the
    target to a whole number of states, and the smearing of 1e-15 reads
    the rounding of the degenerate energies of the star. The Nambu state
    fills the star in part and holds the requested charge exactly
    (scftk.superscf.ShellFilling). Since the star is of one spin, the
    occupation of the other spin is the one of the normal state, and the
    whole charge difference sits in the spin of the star, which moves the
    magnetization by exactly that difference (mz=0.0022461 against
    0.0028164); the energy does not move, since the star is at zero
    energy"""
    nk = 64 # at nk=32 the normal state at T=0 does not converge
    out = []
    for nambu, integ in ((False, "ed"), (True, integration)):
        h = geometry.square_lattice().get_hamiltonian(has_spin=True)
        if nambu: h.setup_nambu_spinor()
        np.random.seed(1)
        scf = Vinteraction(h, U=2.0, filling=0.35, mf="ferro", nk=nk, T=0.,
                integration=integ, solver="broyden_mixing",
                maxerror=1e-9 if integ!="qtci" else 1e-8,
                load_mf=False, write=False, verbose=0)
        assert scf.converged
        out.append(scf)
    normal, nambu = out
    assert abs(nambu.charge_error) < 1e-10 # the charge is held
    assert abs(nambu.total_energy-normal.total_energy) < 1e-8
    # the occupations of the two spins, per cell
    na = np.real(normal.dm[(0, 0, 0)].diagonal())
    nb = np.real(get_eh_sector(nambu.dm[(0, 0, 0)], i=0, j=0).diagonal())
    dn = 0.7-np.sum(na) # the requested charge minus the normal one
    d = np.sort(np.abs(nb-na)) # one spin unchanged, the other takes dn
    assert d[0] < 1e-8
    assert abs(d[1]-abs(dn)) < 1e-8
