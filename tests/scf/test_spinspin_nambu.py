import numpy as np

from pyqula import geometry
from pyqula import meanfield
from pyqula.superconductivity import get_eh_sector

MAXERROR = 1e-6


def _pauli(m):
    mx = (m[0, 1] + m[1, 0]).real/2
    my = (m[1, 0] - m[0, 1]).imag/2
    mz = (m[0, 0] - m[1, 1]).real/2
    return np.array([mx, my, mz])


def test_szsz_on_nambu_matches_normal_state_electron_sector():
    """SzSz on a BdG (Nambu) Hamiltonian must converge to a purely magnetic
    state -- densitydensity()'s existing has_eh dispatch already handles
    the electron+anomalous Wick decoupling generically for any v matrix,
    with no changes needed in SzSz itself -- whose electron sector exactly
    matches the non-Nambu SzSz result, with zero anomalous (pairing) mean
    field spontaneously generated (SzSz alone has no attractive channel to
    seed pairing)."""
    g = geometry.chain()

    h_normal = g.get_hamiltonian(has_spin=True)
    scf_normal = meanfield.SzSz(h_normal, J1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, filling=0.2)
    assert scf_normal.converged
    m_normal = scf_normal.hamiltonian.get_magnetization()

    h_nambu = g.get_hamiltonian(has_spin=True)
    h_nambu.setup_nambu_spinor()
    scf_nambu = meanfield.SzSz(h_nambu, J1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, filling=0.2)
    assert scf_nambu.converged
    mf_full = np.array(scf_nambu.hamiltonian.intra) - np.array(scf_nambu.hamiltonian0.intra)
    ee = get_eh_sector(mf_full, i=0, j=0)
    eh = get_eh_sector(mf_full, i=0, j=1)

    assert np.max(np.abs(eh)) < 1e-8, "SzSz should not spontaneously pair"
    # the electron sector's own z-magnetization must match the non-Nambu run
    mz_nambu = (ee[0, 0] - ee[1, 1]).real/2
    mz_normal = np.mean(m_normal[:, 2])
    assert np.isclose(mz_nambu, mz_normal, atol=1e-3), (mz_nambu, mz_normal)


def test_sxsx_sysy_szsz_on_nambu_are_rotations_of_each_other():
    """SzSz/SxSx/SySy on a BdG Hamiltonian must remain rotationally
    consistent: same total energy, and the electron-sector mean field
    ordering along the intended axis, with zero spurious pairing --
    verifying that rotate_spin.global_spin_rotation (used, unmodified, by
    SxSx/SySy's rotate-solve-rotate-back trick) is correct for Nambu
    Hamiltonians too."""
    g = geometry.chain()
    results = {}
    for axis, fn, guess in [("z", meanfield.SzSz, "ferroZ"),
                             ("x", meanfield.SxSx, "ferroX"),
                             ("y", meanfield.SySy, "ferroY")]:
        h = g.get_hamiltonian(has_spin=True)
        h.setup_nambu_spinor()
        scf = fn(h, J1=-2.0, mf=guess, nk=10, maxerror=MAXERROR, mix=0.3,
                filling=0.2)
        assert scf.converged, f"axis {axis} did not converge"
        mf_full = np.array(scf.hamiltonian.intra) - np.array(scf.hamiltonian0.intra)
        ee = get_eh_sector(mf_full, i=0, j=0)
        eh = get_eh_sector(mf_full, i=0, j=1)
        assert np.max(np.abs(eh)) < 1e-8, f"axis {axis}: spurious pairing"
        results[axis] = (scf.total_energy, _pauli(ee))

    etots = np.array([results[a][0] for a in "xyz"])
    assert np.max(np.abs(etots - np.mean(etots))) < 1e-4, etots

    for axis, expected_component in [("x", 0), ("y", 1), ("z", 2)]:
        m = results[axis][1]
        assert abs(m[expected_component]) > 0.05, (axis, m)
        others = [abs(m[i]) for i in range(3) if i != expected_component]
        assert max(others) < 1e-3, (axis, m)


def test_jinteraction_isotropic_on_nambu_preserves_su2_symmetry():
    """Jinteraction (isotropic Jx=Jy=Jz, ferromagnetic sign) on a BdG
    Hamiltonian, seeded with a random-direction MAGNETIC-only guess, must
    converge to a moment collinear with it (SU(2) symmetry preserved), with
    no spuriously induced pairing: the zero-pairing state is an exact fixed
    point of get_mf_bdg's decoupling (Wick-contracting a zero anomalous
    density matrix always returns zero anomalous mean field, see
    _run_anisotropic_scf's docstring), so seeding purely magnetically stays
    purely magnetic here even though Jinteraction's Nambu path can, in
    general, also induce pairing from J now (see
    test_jinteraction_afm_isotropic_induces_rvb_pairing below, which seeds
    randomly -- including in the pairing channel -- instead)."""
    g = geometry.chain()
    rng = np.random.default_rng(4)
    for _ in range(3):
        v = rng.random(3) - 0.5
        v = v/np.linalg.norm(v)
        h = g.get_hamiltonian(has_spin=True)
        h.setup_nambu_spinor()
        guess = h.copy()
        guess.add_exchange(0.1*v)
        scf = meanfield.Jinteraction(h, Jx1=-2.0, Jy1=-2.0, Jz1=-2.0,
                mf=guess, nk=10, maxerror=MAXERROR, mix=0.2, maxite=300,
                filling=0.2)
        assert scf.converged
        mf_full = np.array(scf.hamiltonian.intra) - np.array(scf.hamiltonian0.intra)
        ee = get_eh_sector(mf_full, i=0, j=0)
        eh = get_eh_sector(mf_full, i=0, j=1)
        assert np.max(np.abs(eh)) < 1e-8
        m = _pauli(ee)
        assert np.linalg.norm(m) > 0.05
        cos_angle = np.dot(m/np.linalg.norm(m), v)
        assert cos_angle > 1 - 1e-3, (v, m, cos_angle)


def test_jinteraction_afm_isotropic_induces_rvb_pairing():
    """Antiferromagnetic isotropic J (NO V/U at all) must be able to
    spontaneously induce a purely superconducting, gauge-invariant BdG mean
    field on its own: Wick's theorem does not care that Sa_i Sa_j started
    life as an exchange interaction rather than a density-density one (it
    is one, in the spin-orbital basis -- see _build_v's docstring), so
    mean-field theory can equally decouple an antiferromagnetic exchange
    into a singlet-pairing channel instead of a Neel one -- the same
    physics behind RVB (resonating-valence-bond) theories of
    exchange-driven superconductivity.

    Seeded with an explicit, small onsite s-wave pairing term at a random
    U(1) phase (rather than mf="random", which draws independent noise for
    every matrix entry with no coherent overlap with the actual
    zero-magnetization singlet-pairing instability -- empirically found to
    converge to the trivial zero-pairing fixed point on a sizeable fraction
    of draws instead, making the test flaky), the resulting gap and total
    energy must be independent of the seed's arbitrary phase, the pairing
    analogue of the magnetic order parameter's direction being arbitrary in
    test_jinteraction_isotropic_on_nambu_preserves_su2_symmetry above -- and
    the gap must be a real, non-numerical-noise magnitude, not just
    "technically nonzero". Contrast with
    test_jinteraction_isotropic_on_nambu_preserves_su2_symmetry (same J
    magnitude, ferromagnetic sign): a ferromagnetic instability has no
    singlet-pairing tendency to decouple into, and stays unpaired even
    with a random (not just magnetic) seed -- see
    test_jinteraction_fm_isotropic_stays_unpaired_even_with_random_seed."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    rng = np.random.default_rng(9)
    gaps = []
    etots = []
    for _ in range(3):
        phase = np.exp(1j*2*np.pi*rng.random())
        h = h0.copy()
        h.turn_nambu()
        guess = h.copy()
        guess.add_swave(0.1*phase)
        scf = meanfield.Jinteraction(h, Jx1=2.0, Jy1=2.0, Jz1=2.0,
                mf=guess, nk=20, maxerror=MAXERROR, mix=0.15, maxite=3000,
                filling=0.3)
        assert scf.converged
        gaps.append(scf.hamiltonian.get_gap())
        etots.append(scf.total_energy)
    gaps = np.array(gaps)
    etots = np.array(etots)
    assert np.min(gaps) > 0.05, gaps
    assert np.max(np.abs(gaps - np.mean(gaps))) < 1e-4, gaps
    assert np.max(np.abs(etots - np.mean(etots))) < 1e-4, etots


def test_jinteraction_fm_isotropic_stays_unpaired_even_with_random_seed():
    """The ferromagnetic-sign counterpart of
    test_jinteraction_afm_isotropic_induces_rvb_pairing: isotropic J with a
    ferromagnetic sign, seeded with a fully random guess (including a
    random pairing component, unlike test_jinteraction_isotropic_on_nambu_
    preserves_su2_symmetry's purely-magnetic guess), must still relax back
    to zero pairing -- a ferromagnetic instability has no singlet-pairing
    channel to decouple into, so a randomly-seeded pairing amplitude simply
    decays away under the SCF iteration instead of being sustained."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    for _ in range(3):
        h = h0.copy()
        h.turn_nambu()
        scf = meanfield.Jinteraction(h, Jx1=-2.0, Jy1=-2.0, Jz1=-2.0,
                mf="random", nk=20, maxerror=MAXERROR, mix=0.15, maxite=3000,
                filling=0.3)
        assert scf.converged
        mf_full = np.array(scf.hamiltonian.intra) - np.array(scf.hamiltonian0.intra)
        eh = get_eh_sector(mf_full, i=0, j=1)
        assert np.max(np.abs(eh)) < 1e-3, np.max(np.abs(eh))


def test_jinteraction_anisotropic_magnetic_seed_no_spurious_pairing():
    """A single anisotropic exchange channel (Jz only, no Jx/Jy and no
    V/U), seeded with a purely magnetic (zero-pairing) guess, must remain
    at EXACTLY zero pairing: the zero-pairing state is an exact fixed point
    of get_mf_bdg's decoupling (a zero anomalous density matrix always
    Wick-contracts to a zero anomalous mean field, regardless of v), so a
    single active exchange channel with nothing seeding pairing should not
    spontaneously break into a superconducting state -- unlike
    test_jinteraction_afm_isotropic_induces_rvb_pairing, this uses only Jz
    (not the isotropic Jx=Jy=Jz that has an SU(2)-symmetric singlet-pairing
    channel available)."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    h = h0.copy()
    h.add_exchange([0.3, 0., 0.])
    h.turn_nambu()
    scf = meanfield.Jinteraction(h, Jz1=-2.0, mf=h.copy(), nk=20,
            maxerror=MAXERROR, mix=0.2, maxite=1000, filling=0.3)
    assert scf.converged
    mf_full = np.array(scf.hamiltonian.intra) - np.array(scf.hamiltonian0.intra)
    eh = get_eh_sector(mf_full, i=0, j=1)
    assert np.max(np.abs(eh)) < 1e-8, np.max(np.abs(eh))


def test_vjinteraction_v_only_on_nambu_matches_vinteraction():
    """VJinteraction with only V1 (no exchange) on a BdG Hamiltonian must
    reduce exactly to Vinteraction's existing, already-validated Nambu
    (spin-triplet superconductivity) treatment, including total_energy.

    This also regression-tests a real bug in densitydensity.py's total
    energy computation (shared by Vinteraction and, through vd,
    VJinteraction), found and fixed while checking that VJinteraction
    gives the same result in a supercell as in a minimal cell:
    get_dc_energy(v, dm) assumes dm's shape matches v's (2n, never
    Nambu-doubled), but the total-energy code used to pass it the full,
    un-extracted Nambu-sized density matrix for a BdG Hamiltonian, silently
    reading the wrong entries -- giving a total energy that was not even
    consistent between a primitive cell and a supercell of the same
    system (see test_vjinteraction_nambu_supercell_consistency)."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()

    h1 = h0.copy()
    h1.add_exchange([0.3, 0., 0.])
    h1.turn_nambu()
    scf_v = meanfield.Vinteraction(h1, V1=-2.0, filling=0.3, mf="random",
            nk=20, maxerror=MAXERROR)

    h2 = h0.copy()
    h2.add_exchange([0.3, 0., 0.])
    h2.turn_nambu()
    scf_vj = meanfield.VJinteraction(h2, V1=-2.0, filling=0.3, mf="random",
            nk=20, maxerror=MAXERROR)

    assert scf_v.converged and scf_vj.converged
    assert np.isclose(scf_v.hamiltonian.get_gap(), scf_vj.hamiltonian.get_gap(),
            atol=1e-3)
    assert np.isclose(scf_v.total_energy, scf_vj.total_energy, atol=1e-3), \
        (scf_v.total_energy, scf_vj.total_energy)


def test_vjinteraction_isotropic_J_and_V_on_nambu_preserves_su2_symmetry():
    """Combining an SU(2)-symmetric V1 pairing channel with an isotropic J1
    (ferromagnetic-sign, uniform order) exchange on a BdG Hamiltonian must
    keep the total energy independent of the (arbitrary) exchange-field
    direction used to seed the SCF loop."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    rng = np.random.default_rng(5)
    etots = []
    for _ in range(3):
        v = rng.random(3) - 0.5
        v = v/np.linalg.norm(v)
        h = h0.copy()
        h.add_exchange(0.3*v)
        h.turn_nambu()
        scf = meanfield.VJinteraction(h, V1=-1.0, J1=-0.3, filling=0.3,
                mf="random", nk=20, maxerror=MAXERROR, mix=0.15, maxite=500)
        assert scf.converged
        etots.append(scf.total_energy)
    etots = np.array(etots)
    assert np.max(np.abs(etots - np.mean(etots))) < 1e-3, etots


def test_vjinteraction_afm_and_fm_J_with_attractive_V_preserve_su2_symmetry():
    """Combining an attractive V1 (inducing superconductivity) with an
    isotropic J1 exchange must keep the total energy independent of the
    (arbitrary) direction of the fixed exchange field used to seed the
    symmetry breaking, for BOTH signs of J1: J1<0 (ferromagnetic, uniform
    order -- a uniform field) and J1>0 (antiferromagnetic, staggered/Neel
    order -- a staggered [+v,-v] field, since Neel order needs at least a
    2-site cell to represent). The SCF's own mean-field guess is an
    unrelated random one in both cases (mf="random"), matching the
    established convention for this kind of check (see
    test_rotational_symmetry_sc.py's _gap_for_random_direction).

    The antiferromagnetic+SC combination converges much more slowly than
    the ferromagnetic one (monotonic but shallow, not oscillating -- a
    genuine competition between the two orders, not a bug), hence the
    larger mix/maxite for that branch."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    rng = np.random.default_rng(6)

    etots_fm = []
    for _ in range(2):
        v = rng.random(3) - 0.5
        v = v/np.linalg.norm(v)
        h = h0.copy()
        h.add_exchange(0.3*v) # fixed uniform field -> ferromagnetic order
        h.turn_nambu()
        scf = meanfield.VJinteraction(h, V1=-2.0, J1=-1.0, filling=0.3,
                mf="random", nk=20, maxerror=MAXERROR, mix=0.15, maxite=2000)
        assert scf.converged
        etots_fm.append(scf.total_energy)
    etots_fm = np.array(etots_fm)
    assert np.max(np.abs(etots_fm - np.mean(etots_fm))) < 1e-4, etots_fm

    etots_afm = []
    for _ in range(2):
        v = rng.random(3) - 0.5
        v = v/np.linalg.norm(v)
        h = h0.copy()
        h.add_exchange([v*0.3, -v*0.3]) # fixed staggered field -> Neel order
        h.turn_nambu()
        scf = meanfield.VJinteraction(h, V1=-2.0, J1=1.0, filling=0.3,
                mf="random", nk=20, maxerror=MAXERROR, mix=0.3, maxite=3000)
        assert scf.converged
        etots_afm.append(scf.total_energy)
    etots_afm = np.array(etots_afm)
    assert np.max(np.abs(etots_afm - np.mean(etots_afm))) < 1e-4, etots_afm
