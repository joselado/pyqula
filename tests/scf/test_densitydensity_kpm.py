import numpy as np
import pytest
from scipy.special import expit

from pyqula import geometry
from pyqula import islands
from pyqula import specialhopping
from pyqula import inout
from pyqula import meanfield
from pyqula import kpm
from pyqula.kpmtk.densitymatrix_kpm import (get_dm_kpm, _dm_kpm_from_needed,
        get_total_energy_kpm, get_fermi4filling_kpm)
from pyqula.kpmtk.bandwidth import estimate_bandwidth
from pyqula.selfconsistency.densitydensity import (get_mf, get_dc_energy,
        random_hermitian_guess, mf_matches_hamiltonian)
# mf_file: densitydensity_kpm.py's own copy, not densitydensity.py's --
# Vinteraction_kpm (exercised below) reads/writes THAT one. Both currently
# hardcode the identical "MF.pkl" literal, but importing from
# densitydensity.py here would silently stop testing the real path if that
# ever changes.
from pyqula.selfconsistency.densitydensity_kpm import mf_file


def _v1_interaction_dict(h, V1=1.0):
    """Build the same spin-doubled first-neighbor interaction dictionary
    Vinteraction/Vinteraction_kpm build internally, for a frozen-Hamiltonian
    cross-check (bypassing the SCF loop entirely)."""
    nd = h.geometry.neighbor_distances()
    mgenerator = specialhopping.distance_hopping_matrix([V1/2., 0., 0.], nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    v = hv.get_hopping_dict()
    for d in list(v.keys()):
        m = v[d]; n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                m1[2*i, 2*j] = m[i, j]
                m1[2*i+1, 2*j] = m[i, j]
                m1[2*i, 2*j+1] = m[i, j]
                m1[2*i+1, 2*j+1] = m[i, j]
        v[d] = m1
    return v


def test_get_dm_kpm_matches_full_dm_for_v1_interaction():
    """get_dm_kpm's selectively-computed density matrix must reproduce the
    same mean field / double-counting energy as exact diagonalization's
    full dense density matrix, for a first-neighbor (V1) interaction on a
    frozen Hamiltonian -- isolating get_dm_kpm's own correctness from any
    SCF convergence-path sensitivity (two independent SCF trajectories can
    settle into distinct, individually valid fixed points even when both
    density-matrix backends are correct)."""
    h = islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()
    h.add_sublattice_imbalance(0.2)  # seed a nontrivial charge pattern
    v = _v1_interaction_dict(h)

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=1)
    dm_kpm = get_dm_kpm(h, v, nk=1, npol=300, scale=None)

    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_kpm[d]))
        assert diff < 1e-2, f"direction {d}: |mf_ed-mf_kpm|={diff}"
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_kpm))
    assert ediff < 1e-2


def test_hubbard_kpm_transverse_mean_field_matches_ed():
    """Regression test: hubbard_kpm builds the onsite Hubbard interaction
    v[(0,0,0)] asymmetrically (mirroring densitydensity.hubbard's own
    convention, which relies on exact diagonalization always returning a
    dense dm block regardless of v's sparsity pattern). If
    required_elements ever again requests only the (i,j) entries where v
    is nonzero -- instead of also requesting the transposed (j,i) entry
    normal_term_ij actually reads for a self-paired onsite direction -- the
    entire transverse-spin (non-collinear) part of the Hubbard mean field
    is silently computed from a zero density matrix instead of its true
    value. A Hamiltonian with a non-collinear (x-direction) exchange field
    gives a density matrix with a genuine nonzero transverse-spin
    component, exercising exactly that path."""
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3)
    h = g.get_hamiltonian()
    h.add_exchange([0.3, 0., 0.])  # transverse (non-z) exchange field
    n = h.intra.shape[0]

    v = dict()
    zero = np.zeros((n, n), dtype=np.complex128)
    for i in range(n//2): zero[2*i, 2*i+1] = 1.5  # hubbard_kpm's own v construction
    v[(0, 0, 0)] = zero

    dm_ed = h.get_density_matrix(ds=[(0, 0, 0)], nk=1)
    dm_kpm = get_dm_kpm(h, v, nk=1, npol=300, scale=None)

    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)

    transverse_ed = np.array([mf_ed[(0, 0, 0)][2*i+1, 2*i] for i in range(n//2)])
    assert np.max(np.abs(transverse_ed)) > 1e-3, \
        "test setup didn't actually produce a nonzero transverse mean field"
    transverse_kpm = np.array([mf_kpm[(0, 0, 0)][2*i+1, 2*i] for i in range(n//2)])
    diff = np.max(np.abs(transverse_ed - transverse_kpm))
    assert diff < 1e-2, \
        f"KPM transverse-spin Hubbard mean field diverges from ED: {diff}"


def test_get_dm_kpm_rejects_spinless_bdg():
    """get_dm_kpm's Nambu index mapping (required_elements_eh/
    _local_nambu_index) assumes a spinful Nambu Hamiltonian (4 Nambu slots
    per site); a spinless BdG Hamiltonian uses a different (2 slots per
    site) convention it does not implement, and must fail loudly rather
    than silently computing wrong density-matrix entries."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.setup_nambu_spinor()  # spinless_nambu: has_eh=True, has_spin=False
    assert h.has_eh and not h.has_spin
    v = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    with pytest.raises(NotImplementedError):
        get_dm_kpm(h, v, nk=4)


def test_get_dm_kpm_matches_full_dm_for_bdg_hamiltonian():
    """required_elements_eh's Nambu-reordering index arithmetic must
    reproduce exact diagonalization's dense (2n)x(2n) density matrix for a
    (spinful) BdG/Nambu Hamiltonian, on a frozen Hamiltonian -- isolating
    get_dm_kpm's has_eh path from SCF convergence-path sensitivity."""
    h = geometry.chain().get_hamiltonian()
    h.turn_nambu()
    n = h.intra.shape[0]//2  # electron-sector (v-space) size
    v = dict()
    zero = np.zeros((n, n), dtype=np.complex128)
    zero[0, 1] += 0.6/2.
    zero[1, 0] += 0.6/2.
    v[(0, 0, 0)] = zero

    dm_ed = h.get_density_matrix(ds=list(v.keys()), nk=8)
    dm_kpm = get_dm_kpm(h, v, nk=8, npol=400, scale=None)

    mf_ed = get_mf(v, dm_ed, has_eh=True)
    mf_kpm = get_mf(v, dm_kpm, has_eh=True)
    for d in v:
        diff = np.max(np.abs(mf_ed[d]-mf_kpm[d]))
        assert diff < 1e-2, f"direction {d}: |mf_ed-mf_kpm|={diff}"


def test_get_dm_kpm_temperature_smearing_moves_toward_ed():
    """T is forwarded from the KPM SCF loop into get_dm_kpm for
    finite-temperature (Fermi-Dirac) smearing, matching what
    densitymatrix.py's full_dm(h,T=...) does for the exact-diagonalization
    path -- it must not be silently ignored, so passing an explicit
    (non-tiny) T has to actually change the resulting density matrix
    relative to the effectively-zero-temperature default."""
    h = islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()
    v = {(0, 0, 0): np.eye(h.intra.shape[0], dtype=np.complex128)*0.5}
    dm_cold = get_dm_kpm(h, v, nk=1, npol=300, scale=None, T=1e-7)
    dm_warm = get_dm_kpm(h, v, nk=1, npol=300, scale=None, T=2.0)
    diff = np.max(np.abs(dm_cold[(0, 0, 0)] - dm_warm[(0, 0, 0)]))
    assert diff > 1e-3, "changing T had no effect on get_dm_kpm's output"


# Regression coverage for two bugs found while adding VJinteraction's own
# integration="kpm" mode: generic_densitydensity/generic_densitydensity_kpm's
# shared mf=None default-guess and MF.pkl-caching logic (densitydensity.py)
# had the same non-Hermitian-guess and shape-mismatch issues that
# spinspin.py's copy of the same pattern was fixed for.


def test_random_hermitian_guess_is_hermitian():
    """random_hermitian_guess must produce mf[d] == mf[-d].conj().T for
    every direction pair (not just symmetrize the onsite (0,0,0) term, as
    the old inline construction did) -- required for integration="kpm"'s
    Chebyshev recursion, which assumes real eigenvalues bounded by an
    energy scale estimated from H(k); a non-Hermitian H(k) can violate
    that badly enough to diverge exponentially (see
    test_vinteraction_kpm_default_guess_does_not_blow_up)."""
    v = {(0, 0, 0): np.eye(2, dtype=np.complex128),
         (1, 0, 0): np.eye(2, dtype=np.complex128),
         (-1, 0, 0): np.eye(2, dtype=np.complex128)}
    mf = random_hermitian_guess(v, (2, 2))
    for d, m in mf.items():
        d2 = tuple(-x for x in d)
        diff = np.max(np.abs(m - mf[d2].conj().T))
        assert diff < 1e-14, (d, diff)


def test_vinteraction_kpm_default_guess_does_not_blow_up():
    """Regression test: before random_hermitian_guess, Vinteraction_kpm's
    mf=None default guess was non-Hermitian (independent random matrices
    per direction, only the onsite term symmetrized), which made the
    Chebyshev recursion diverge -- observed mean-field magnitude >1e40
    after a handful of SCF iterations for exactly this system (a plain
    spinless V1 interaction on honeycomb, nothing exotic). With the fix,
    this must converge normally instead."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    scf = meanfield.Vinteraction_kpm(h, V1=1.0, filling=0.3, nk=6, mix=0.3,
            maxerror=1e-2, maxite=60, npol=200, load_mf=False, verbose=0)
    assert scf.converged
    for d, m in scf.mf.items():
        assert np.all(np.isfinite(m)), d
        assert np.max(np.abs(m)) < 10.0, (d, np.max(np.abs(m)))


def test_mf_matches_hamiltonian_rejects_shape_mismatch():
    h = geometry.chain().get_hamiltonian(has_spin=False)  # (1,1) matrices
    ok_mf = {(0, 0, 0): np.zeros((1, 1), dtype=np.complex128)}
    bad_mf = {(0, 0, 0): np.zeros((2, 2), dtype=np.complex128)}
    assert mf_matches_hamiltonian(h, ok_mf)
    assert not mf_matches_hamiltonian(h, bad_mf)


def test_stale_mf_pkl_shape_mismatch_is_discarded(tmp_path, monkeypatch):
    """Regression test: a cached MF.pkl left over from an unrelated,
    differently-shaped Hamiltonian (e.g. a spinful run's (2,2) blocks,
    reused by a spinless (1,1) one) used to be silently accepted --
    MultiHopping(h0.get_dict())+MultiHopping(mf) does not reliably raise
    on a shape mismatch, since numpy broadcasts a (1,1) array against a
    (2,2) one instead of erroring, corrupting h0's own matrix shapes
    downstream (an opaque "inhomogeneous shape" crash far from the actual
    cause). mf_matches_hamiltonian's explicit shape check must catch this
    and fall back to a fresh guess instead."""
    monkeypatch.chdir(tmp_path)  # MF.pkl is always read/written in the cwd
    stale_mf = {(0, 0, 0): np.eye(2, dtype=np.complex128)*0.3,
                (1, 0, 0): np.zeros((2, 2), dtype=np.complex128),
                (-1, 0, 0): np.zeros((2, 2), dtype=np.complex128)}
    inout.save(stale_mf, mf_file)

    h = geometry.chain().get_hamiltonian(has_spin=False)  # (1,1) matrices
    scf = meanfield.Vinteraction_kpm(h, V1=1.0, filling=0.3, nk=6, mix=0.3,
            maxerror=1e-2, maxite=60, npol=200, load_mf=True, verbose=0)
    assert scf.converged
    for d, m in scf.mf.items():
        assert m.shape == (1, 1), (d, m.shape)


def test_dm_kpm_batched_reconstruction_matches_per_pair_reference():
    """_dm_kpm_from_needed's batched profile reconstruction (a precomputed
    Chebyshev basis + matrix multiplies, replacing a per-pair
    kpm.dm_ij_energy+trapz loop -- see that function's docstring) must
    reproduce the original per-pair formula to numerical precision. This
    is a much tighter check than the ~1e-2 KPM-vs-ED tolerance used
    elsewhere in this file: that tolerance is dominated by KPM's own
    Chebyshev-truncation approximation error relative to exact
    diagonalization, a different question from whether this algebraic
    reformulation of the SAME truncated expansion is correct. Isolates
    the latter by reimplementing the old per-pair formula directly
    (kpm.dm_ij_energy is still present/importable -- only
    _dm_kpm_from_needed stopped calling it) and comparing at nk=1 (single
    k-point, so there is no Bloch-phase summation to also reproduce) on a
    small frozen Hamiltonian."""
    h = islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()
    h.add_sublattice_imbalance(0.3)  # nontrivial charge pattern
    n = h.intra.shape[0]
    needed = {((0, 0, 0), i, j) for i in range(n) for j in range(n)
            if (i + j) % 3 == 0}  # an arbitrary, non-trivial subset
    npol, ne, T = 80, 320, 1e-6

    dm = _dm_kpm_from_needed(h, needed, nk=1, npol=npol, ne=ne, scale=None, T=T)

    # Reconstruct exactly what _dm_kpm_from_needed itself would have used
    # for scale/xin/weights, to feed the reference per-pair computation
    # the identical inputs.
    hk_gen = h.get_hk_gen()
    k = list(h.geometry.get_kmesh(nk=1)[0])
    Hk = hk_gen(k)
    used_scale = 1.1*estimate_bandwidth(Hk)
    Tsafe = abs(T) if T != 0. else 1e-15
    upper = min(0.99*used_scale, 30.*Tsafe)
    xin = np.linspace(-0.99*used_scale, upper, ne)
    weights = expit(-xin/Tsafe)

    worst = 0.0
    for (d, i, j) in needed:
        (x, y) = kpm.dm_ij_energy(Hk, i=j, j=i, scale=used_scale, npol=npol,
                ne=ne, x=xin)
        ref = np.trapezoid(y*weights, x=x)/np.pi
        diff = abs(dm[d][i, j] - ref)
        worst = max(worst, diff)
    assert worst < 1e-6, worst


def test_get_total_energy_kpm_matches_exact_diagonalization():
    """get_total_energy_kpm (the sum of occupied eigenvalues, obtained by
    integrating the KPM-reconstructed density of states up to the Fermi
    energy instead of diagonalizing -- see its own docstring) must
    reproduce spectrum.total_energy's exact-diagonalization result on a
    frozen Hamiltonian, isolating this from any SCF convergence-path
    sensitivity. Uses a non-trivial (asymmetric, non-particle-hole-
    symmetric) Hamiltonian -- a random exchange field plus sublattice
    imbalance -- so this cannot pass by accident via some cancellation
    specific to a clean/symmetric spectrum."""
    from pyqula import spectrum
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    h.add_exchange([0.1, 0.05, 0.2])

    nk = 6
    fermi = h.get_fermi4filling(0.3, nk=nk)
    h.shift_fermi(-fermi)  # get_total_energy_kpm/spectrum.total_energy both
                           # assume the Fermi energy is already at 0

    etot_ed = spectrum.total_energy(h, nk=nk)
    etot_kpm = get_total_energy_kpm(h, fermi=0.0, nk=nk, npol=500)
    reldiff = abs(etot_ed - etot_kpm)/abs(etot_ed)
    assert reldiff < 1e-2, (etot_ed, etot_kpm, reldiff)


def test_get_total_energy_kpm_rejects_nambu():
    h = geometry.chain().get_hamiltonian()
    h.turn_nambu()
    with pytest.raises(NotImplementedError):
        get_total_energy_kpm(h, fermi=0.0, nk=4, npol=100)
