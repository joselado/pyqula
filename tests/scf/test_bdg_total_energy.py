import numpy as np

from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector


def test_nambu_band_energy_matches_the_normal_state():
    """A Nambu (BdG) description of a state with NO pairing describes the
    same state as the normal-state Hamiltonian it was built from, so it
    must have the same total energy.

    Summing the BdG spectrum below the Fermi level is not that energy:
    H = (1/2) Psi^dag H_BdG Psi + (1/2) Tr h, so sum_{E<0} E_BdG =
    2*E_normal - Tr h. The invariant asserted here is the equality of the
    two descriptions, not either side of that identity."""
    for gf in [geometry.chain, geometry.honeycomb_lattice]:
        for mu in (0.0, 0.7):
            h = gf().get_hamiltonian()
            h.shift_fermi(-mu)
            hn = h.copy()
            hn.setup_nambu_spinor() # the same state, in the Nambu basis
            assert abs(h.get_total_energy(nk=20)
                       - hn.get_total_energy(nk=20)) < 1e-8


def test_nambu_scf_total_energy_matches_the_normal_state(tmp_path,
                                                         monkeypatch):
    """Same invariant, one level up: a repulsive intersite interaction on
    a Nambu Hamiltonian converges to EXACTLY zero pairing, so the whole
    mean-field chain (band energy + the mu*N un-shift + the double-counting
    energy) must reproduce the normal-state answer. It is the chain that
    has to be consistent: the band energy and the mu*N term used to be on
    a doubled (Nambu) scale while the double-counting term was not."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    out = dict()
    for nambu in (False, True):
        h = g.get_hamiltonian()
        if nambu: h.setup_nambu_spinor()
        h0 = h.copy()
        s = meanfield.Vinteraction(h, V1=1.0, filling=0.5, nk=20,
                mf="ferroZ", mix=0.3, maxerror=1e-8, maxite=1000,
                load_mf=False, verbose=0)
        if nambu: # check the premise: the converged state has no pairing
            mfm = np.array(s.hamiltonian.intra) - np.array(h0.intra)
            assert np.max(np.abs(get_eh_sector(mfm, i=0, j=1))) < 1e-8
        out[nambu] = s.total_energy
    assert abs(out[True]-out[False]) < 1e-6


# The pairing part of the interaction energy. The band energy of a BdG
# mean-field Hamiltonian counts the whole interaction energy twice, the
# anomalous part included, and scf.total_energy used to subtract only the
# normal (Hartree-Fock) half of that double counting. The checks below do
# not rely on the formula being right in its own terms: the chain compares
# with the BCS value |Delta|^2/|U|, the generic system with the variational
# stationarity of the mean-field energy, which a wrong prefactor on the
# pairing term turns into a first-order slope.

from pyqula import superconductivity as sc
from pyqula.multihopping import MultiHopping
from pyqula.scftk.densitydensity import (get_mf, get_dc_energy,
        electron_dimension)
from pyqula.scftk.superscf import get_mf_bdg, get_dc_energy_anomalous


def _expect(X, dm):
    """<X> for a Nambu matrix dict X: (1/2) sum_d Tr(dm[d].T X[d]) plus
    the constant (1/2) Tr of the electron block. Checked against the band
    energy in test_expectation_value_reproduces_the_band_energy."""
    out = sum(0.5*np.sum(np.array(dm[d])*np.array(X[d])) for d in X)
    e00 = sc.get_eh_sector(np.array(X[(0,0,0)]), i=0, j=0)
    return (out + 0.5*np.trace(e00)).real


def _generic_nambu():
    """No inversion, no spin rotation symmetry, complex hoppings"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_haldane(0.1)
    h.add_rashba(0.2)
    h.add_exchange([0.1, 0.2, 0.3])
    h.add_onsite(0.2)
    h.setup_nambu_spinor()
    return h


def test_expectation_value_reproduces_the_band_energy():
    nk = 6
    h = _generic_nambu()
    h.add_pairing(delta=0.2, mode="swave")
    h.add_pairing(delta=0.15, mode="px")
    h = h.get_multicell().get_dense()
    X = h.get_multihopping().get_dict()
    dm = h.get_density_matrix(ds=list(X.keys()), nk=nk)
    assert abs(_expect(X, dm) - h.get_total_energy(nk=nk)) < 1e-10


def test_attractive_hubbard_chain_gives_the_bcs_pairing_energy(tmp_path,
                                                               monkeypatch):
    """Onsite U<0 on a chain: the pairing field is Delta = U <c_dn c_up>,
    so the pairing energy is U|<c_dn c_up>|^2 = -|Delta|^2/|U| per site and
    its double counting +|Delta|^2/|U|"""
    monkeypatch.chdir(tmp_path)
    U = -2.0
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(0.3)
    h.setup_nambu_spinor()
    s = meanfield.Vinteraction(h, U=U, filling=0.4, nk=20, mf="swave",
            mix=0.5, maxerror=1e-11, maxite=3000, load_mf=False, verbose=0)
    hs = s.hamiltonian
    delta = np.abs(sc.get_eh_sector(np.array(hs.intra), i=0, j=1)).max()
    assert delta > 0.1  # the premise: a paired state
    dm = hs.get_density_matrix(ds=[(0,0,0)] + list(s.v.keys()), nk=20)
    dca = get_dc_energy_anomalous(get_mf(s.v, dm, has_eh=True), dm)
    assert abs(dca - delta**2/abs(U)) < 1e-8
    dme = {k: sc.get_eh_sector(m, i=0, j=0) for (k, m) in dm.items()}
    etot = (hs.get_total_energy(nk=20)
            + hs.fermi*electron_dimension(hs)*0.4
            + get_dc_energy(s.v, dme) + delta**2/abs(U))
    assert abs(s.total_energy - etot) < 1e-8


def test_paired_mean_field_energy_is_stationary(tmp_path, monkeypatch):
    """E[rho] = <H0 - mu N> + E_int[rho] is stationary at the
    self-consistent state. Scale the converged pairing field by lambda,
    rebuild rho(lambda), and evaluate E with the pairing energy
    <MF_anomalous[rho]>/2: the slope at lambda=1 must fall as eps^2. The
    normal field is scaled as a control. Then scf.total_energy must be that
    same E plus mu N."""
    monkeypatch.chdir(tmp_path)
    nk, filling = 6, 0.4
    h = _generic_nambu()
    h0 = h.copy()
    s = meanfield.Vinteraction(h, U=-2.5, V1=-0.8, filling=filling, nk=nk,
            mf="swave", mix=0.5, maxerror=1e-12, maxite=5000,
            load_mf=False, verbose=0)
    hs = s.hamiltonian.get_multicell().get_dense()
    v, mu = s.v, s.hamiltonian.fermi
    ds = sorted(set([(0,0,0)] + list(v.keys())
                    + list(hs.get_multihopping().get_dict().keys())))
    H0 = h0.get_multicell().get_dense()
    H0.shift_fermi(-mu)
    X0 = H0.get_multihopping().get_dict()
    def energy(dm, c=1.0):
        dme = {k: sc.get_eh_sector(np.array(m), i=0, j=0)
               for (k, m) in dm.items()}
        mfa = get_mf_bdg(v, dm, compute_normal=False)
        return (_expect(X0, dm) - get_dc_energy(v, dme)
                + c*0.5*_expect(mfa, dm))
    dm1 = hs.get_density_matrix(ds=ds, nk=nk)
    mft = get_mf(v, dm1, has_eh=True)
    mfa = get_mf_bdg(v, dm1, compute_normal=False)
    mfn = (MultiHopping(mft) - MultiHopping(mfa)).get_dict()
    assert max(np.abs(np.array(m)).max() for m in mfa.values()) > 0.1
    nel = filling*electron_dimension(hs)
    assert abs(energy(dm1) + mu*nel - s.total_energy) < 1e-8
    def slope(P, eps, c=1.0):
        es = []
        for lam in (1-eps, 1+eps):
            H = hs.copy()
            H.set_multihopping(MultiHopping(H.get_multihopping().get_dict())
                               + (lam-1)*MultiHopping(P))
            es.append(energy(H.get_density_matrix(ds=ds, nk=nk), c=c))
        return abs(es[1]-es[0])/(2*eps)
    for P in (mfa, mfn):
        s3, s4 = slope(P, 1e-3), slope(P, 1e-4)
        assert s4 < 1e-6 and s4 < s3/30  # zero slope, eps^2 residue
    # the check discriminates: a pairing energy counted twice is not
    # stationary
    assert slope(mfa, 1e-3, c=2.0) > 1e-2


def test_exchange_pairing_energy_obeys_hellmann_feynman(tmp_path,
                                                        monkeypatch):
    """The same check through the public API, for pairing induced by an
    antiferromagnetic exchange alone (the rotated x/y channels of
    Jinteraction). At fixed mu, with H = H0 + lambda*H_int, the slope of
    the grand potential at lambda=1 is the interaction energy, which is the
    band energy minus the grand potential. Without the pairing double
    counting the two differ by 1e-2 here; with it by roughly 1e-7, the
    eps^2 error of the finite difference."""
    monkeypatch.chdir(tmp_path)
    h = geometry.bichain().get_hamiltonian()
    h.turn_nambu()
    def run(lam, **kwargs):
        return meanfield.Jinteraction(h.copy(), Jx1=2.0*lam, Jy1=2.0*lam,
                Jz1=2.0*lam, nk=20, mix=0.15, maxite=20000, verbose=0,
                **kwargs)
    guess = h.copy()
    guess.add_swave(0.1*np.exp(0.7j))
    s = run(1.0, mf=guess, filling=0.3, maxerror=1e-9)
    hs, mu = s.hamiltonian, s.hamiltonian.fermi
    s1 = run(1.0, mf=hs, mu=mu, maxerror=1e-12)
    delta = np.abs(get_eh_sector(np.array(s1.hamiltonian.intra),
                                 i=0, j=1)).max()
    assert delta > 0.03  # the premise: exchange-driven pairing
    eint = s1.hamiltonian.get_total_energy(nk=20) - s1.total_energy
    eps = 1e-3
    op = run(1.0+eps, mf=hs, mu=mu, maxerror=1e-12).total_energy
    om = run(1.0-eps, mf=hs, mu=mu, maxerror=1e-12).total_energy
    assert abs((op-om)/(2*eps) - eint) < 1e-5


def test_kpm_density_matrix_carries_the_pairing_energy(tmp_path,
                                                       monkeypatch):
    """The KPM backend computes only the entries of the density matrix the
    mean field reads, which for the anomalous part is the (0,1) block. The
    pairing double counting must come out of that sparse matrix the same as
    out of the full one; reading the (1,0) block as well, which the sparse
    matrix leaves at zero, loses part of it."""
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian()
    h.add_rashba(0.2)
    h.add_onsite(0.2)
    h.setup_nambu_spinor()
    s = meanfield.Vinteraction_kpm(h, U=-2.5, V1=-0.8, filling=0.4, nk=10,
            mf="swave", mix=0.5, maxerror=1e-5, maxite=500, verbose=0,
            npol=200)
    full = s.hamiltonian.get_density_matrix(
            ds=[(0,0,0)] + list(s.v.keys()), nk=10)
    sparse = get_dc_energy_anomalous(get_mf(s.v, s.dm, has_eh=True), s.dm)
    exact = get_dc_energy_anomalous(get_mf(s.v, full, has_eh=True), full)
    assert exact > 0.1  # the premise: a paired state
    assert abs(sparse - exact) < 1e-4
