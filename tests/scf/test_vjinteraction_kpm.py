import numpy as np
import pytest
import scipy.sparse as sp

from pyqula import geometry
from pyqula import meanfield
from pyqula.multihopping import MultiHopping
from pyqula.selfconsistency import spinspin
from pyqula.selfconsistency.spinspin import _sparse_pairs_to_needed
from pyqula.densitymatrix import full_dm_accumulate_sparse
from pyqula.kpmtk.densitymatrix_kpm import _dm_kpm_from_needed

# Regression coverage for VJinteraction's integration="kpm" density-matrix
# path (selfconsistency.spinspin._run_anisotropic_scf's use_kpm branch).
# It reuses _build_sparse_pairs' needed (direction,row,col) positions --
# the exact same ones the "ed" sparse path (densitymatrix.
# full_dm_accumulate_sparse) reads -- but evaluates them through
# kpmtk.densitymatrix_kpm's per-k Chebyshev-moment engine instead of
# diagonalizing H(k), mirroring Vinteraction_kpm/densitydensity_kpm.py.


def _build_vj_matrices(g):
    h0 = g.get_hamiltonian(has_spin=True)
    h1 = h0.get_multicell().get_dense()
    nd = h1.geometry.neighbor_distances()
    vz = spinspin._build_v(h1, -0.3, 0.1, 0.0, None, nd=nd)
    vd = spinspin._build_density_v(h1, 0.3, 0.0, 0.0, 1.0, None, nd=nd)
    vx = spinspin._build_v(h1, 0.05, 0.1, 0.0, None, nd=nd)
    vy = spinspin._build_v(h1, 0.05, 0.1, 0.0, None, nd=nd)
    vz = (MultiHopping(vz) + MultiHopping(vd)).get_dict()
    v_dirs = {d: None for d in (set(vz) | set(vx) | set(vy))}
    n = vz[(0, 0, 0)].shape[0]
    pairs = spinspin._build_sparse_pairs([vz, vx, vy], v_dirs, n)
    return h1, pairs


def test_kpm_density_matrix_matches_ed_sparse_at_requested_positions():
    """_dm_kpm_from_needed, fed the exact same sparse_pairs positions the
    ED sparse path (full_dm_accumulate_sparse) reads, must reproduce them
    on a frozen Hamiltonian -- isolating the KPM backend's own correctness
    from any SCF convergence-path sensitivity (same reasoning as
    tests/scf/test_densitydensity_kpm.py)."""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h1, pairs = _build_vj_matrices(g)
    h1.add_sublattice_imbalance(0.3)  # seed a nontrivial charge pattern
    nk = 4
    dm_ed = full_dm_accumulate_sparse(h1, pairs, nk=nk, delta=1e-6)
    needed = _sparse_pairs_to_needed(pairs)
    dm_kpm = _dm_kpm_from_needed(h1, needed, nk=nk, npol=300, scale=None, T=1e-6)
    for d, (rows, cols) in pairs.items():
        if len(rows) == 0:
            continue
        diff = np.max(np.abs(dm_ed[d][rows, cols] - dm_kpm[d][rows, cols]))
        assert diff < 1e-2, (d, diff)


def test_vjinteraction_kpm_one_step_matches_ed_fixed_point():
    """Rather than running two independent (and independently noisy) SCF
    trajectories to convergence, seed the SCF loop directly at the already
    -converged ED fixed point and take a single step through the KPM
    backend (maxite=0): the resulting mean field must reproduce that same
    fixed point, isolating the KPM density-matrix backend's own correctness
    from SCF convergence-path sensitivity (same reasoning as
    test_kpm_density_matrix_matches_ed_sparse_at_requested_positions)."""
    g = geometry.chain()
    params = dict(mf="ferroZ", nk=10, mix=0.3, filling=0.2)
    h_ed = g.get_hamiltonian(has_spin=True)
    scf_ed = meanfield.VJinteraction(h_ed, U=5.0, J1z=-1.0,
            maxerror=1e-6, maxite=300, **params)
    assert scf_ed.converged

    h_kpm = g.get_hamiltonian(has_spin=True)
    scf_kpm = meanfield.VJinteraction(h_kpm, U=5.0, J1z=-1.0,
            mf=scf_ed.mf, nk=10, filling=0.2, mix=1.0,
            maxerror=1.0, maxite=0, integration="kpm", npol=400)
    for d in scf_ed.mf:
        diff = np.max(np.abs(scf_ed.mf[d] - scf_kpm.mf[d]))
        assert diff < 1e-2, (d, diff)


def test_vjinteraction_kpm_default_guess_is_hermitian_and_stable():
    """_run_anisotropic_scf's mf=None default guess must be Hermitian
    (mf[d] == mf[-d].conj().T for every direction pair, not just the
    onsite (0,0,0) term): under integration="ed" a mildly non-Hermitian
    starting guess is tolerated (diagonalizing a near-Hermitian H(k) is
    still well-behaved, and the SCF's own mixing washes out the residual
    asymmetry within a few iterations), but under integration="kpm" it is
    not -- the Chebyshev recursion assumes real eigenvalues bounded by
    `scale`, and a non-Hermitian H(k) can violate that badly enough to
    blow up exponentially over npol recursion steps. Regression: before
    this was fixed to mirror each direction's matrix onto its opposite
    (see _run_anisotropic_scf), the mean field reached >1e50 in magnitude
    after a single SCF iteration starting from the default guess."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    scf = meanfield.VJinteraction(h, U=5.0, J1z=-1.0, nk=10, filling=0.2,
            mix=0.3, maxerror=1e-8, maxite=2, integration="kpm", npol=300)
    for d, m in scf.mf.items():
        assert np.all(np.isfinite(m)), d
        assert np.max(np.abs(m)) < 10.0, (d, np.max(np.abs(m)))


def test_vjinteraction_kpm_rejects_nambu():
    """VJinteraction's integration="kpm" path is only implemented for a
    normal-state Hamiltonian -- see _run_anisotropic_scf's docstring for
    why the Nambu case (vd in its own reordered basis) is out of scope --
    and must fail loudly rather than silently falling back to ED or
    misreading the Nambu-doubled density matrix."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    with pytest.raises(NotImplementedError):
        meanfield.VJinteraction(h, U=5.0, mf="ferroZ", nk=10,
                maxerror=1e-6, mix=0.3, filling=0.2, integration="kpm")


def test_vjinteraction_rejects_unknown_integration_value():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError):
        meanfield.VJinteraction(h, U=5.0, mf="ferroZ", nk=10,
                maxerror=1e-6, mix=0.3, filling=0.2, integration="qtci")


def test_get_mean_field_hamiltonian_kpm_matches_ed_for_spinful_hamiltonian():
    """h.get_mean_field_hamiltonian(integration="kpm") on a spinful
    Hamiltonian must route through VJinteraction's new KPM backend (not
    error out, and not silently fall back to Vinteraction, which would
    drop the J1z exchange term) and give a result close to
    integration="ed"."""
    g = geometry.chain()
    params = dict(mf="ferroZ", nk=10, mix=0.3, maxite=300, filling=0.2,
            U=5.0, J1z=-1.0)
    h_ed = g.get_hamiltonian(has_spin=True)
    h_ed, e_ed = h_ed.get_mean_field_hamiltonian(return_total_energy=True,
            maxerror=1e-6, integration="ed", **params)
    assert h_ed is not None

    h_kpm = g.get_hamiltonian(has_spin=True)
    h_kpm, e_kpm = h_kpm.get_mean_field_hamiltonian(return_total_energy=True,
            maxerror=1e-3, integration="kpm", npol=400, **params)
    assert h_kpm is not None
    assert np.isclose(e_ed, e_kpm, atol=5e-2), (e_ed, e_kpm)
    m_ed = h_ed.get_magnetization()
    m_kpm = h_kpm.get_magnetization()
    assert np.mean(np.abs(m_ed - m_kpm)) < 5e-2


def test_get_mean_field_hamiltonian_kpm_still_works_for_spinless_hamiltonian():
    """integration="kpm" for a spinless Hamiltonian must still route to
    Vinteraction_kpm (get_mean_field_hamiltonian_kpm's own engine) instead
    of erroring -- before VJinteraction supported integration="kpm", this
    combination fell through to Vinteraction(integration="kpm"), which
    does not accept "kpm" at all.

    load_mf=False: Vinteraction_kpm's SCF loop otherwise tries to
    warm-start from a cached MF.pkl in the cwd, which may be incompatible
    (different shape) with this Hamiltonian if left over from an unrelated
    run. mf="random": densitydensity_kpm.py's own default (mf=None) guess
    construction (a separate copy of the same pattern VJinteraction's
    _run_anisotropic_scf used to have) only symmetrizes the onsite
    (0,0,0) term, leaving off-diagonal directions independently random and
    the overall guess non-Hermitian -- harmless for exact diagonalization,
    but enough to blow up integration="kpm"'s Chebyshev recursion (see
    _run_anisotropic_scf's now-fixed version of the same issue). Not fixed
    here since densitydensity_kpm.py is shared by Vinteraction_kpm/
    hubbard_kpm generally, out of scope for VJinteraction's own kpm mode;
    mf="random" (meanfield.guess's Hermitian-by-construction random mode)
    sidesteps it for this dispatch-only test."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    h_new = h.get_mean_field_hamiltonian(V1=1.0, filling=0.3, nk=6, mix=0.3,
            maxerror=1e-2, maxite=100, integration="kpm", npol=200,
            load_mf=False, mf="random")
    assert h_new is not None


def test_vjinteraction_kpm_keeps_sparse_hamiltonian_sparse():
    """integration="kpm" must never force a sparse h0 into a dense
    representation anywhere in the SCF loop -- the whole point of picking
    KPM over ED is to support large systems whose unit cell (h.intra) is
    too big to hold as a dense array. Two ways this could silently happen:
    the initial h0.get_multicell().get_dense() call every other
    integration mode (and Jinteraction) always makes, and the per-iteration
    mean-field update, where scipy's sparse+dense addition returns a dense
    matrix unless explicitly re-sparsified (see _run_anisotropic_scf's
    keep_sparse handling)."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    assert h.is_sparse and sp.issparse(h.intra)

    scf = meanfield.VJinteraction(h, U=5.0, J1z=-1.0, mf="ferroZ", nk=6,
            filling=0.2, mix=0.3, maxerror=1.0, maxite=2,
            integration="kpm", npol=200)
    hres = scf.hamiltonian
    assert hres.is_sparse
    assert sp.issparse(hres.intra), type(hres.intra)
    for t in hres.hopping:
        assert sp.issparse(t.m), type(t.m)
