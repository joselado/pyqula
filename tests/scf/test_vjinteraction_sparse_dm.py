import numpy as np

from pyqula import geometry
from pyqula import meanfield
from pyqula.multihopping import MultiHopping
from pyqula.scftk import spinspin
from pyqula.scftk.densitydensity import get_dm
from pyqula.densitymatrix import (full_dm_accumulate, full_dm_accumulate_sparse,
        full_dm_accumulate_sparse_with_fermi)

# Regression coverage for the sparse density-matrix machinery
# (scftk.spinspin._build_sparse_pairs, densitymatrix's
# full_dm_accumulate_sparse/full_dm_accumulate_sparse_with_fermi,
# dmtk.fulldm.full_dm_batch_d_sparse) introduced for VJinteraction/
# Jinteraction. None of the small-geometry tests elsewhere in this file/
# module are large enough for any direction's requested-pair count to
# cross the dense_fraction*n^2 threshold, so those tests always exercise
# the dense-fallback branch only, never the actual sparse gather kernel --
# these tests use a large enough system (98-site honeycomb supercell, 196
# orbitals) that both branches are genuinely exercised, confirmed by
# asserting on the pair counts directly rather than just trusting it.

DENSE_FRACTION = 0.01


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
    return h1, v_dirs, pairs, n


def test_sparsity_thresholds_actually_exercise_both_branches():
    """Sanity check for the tests below: on a 98-site/196-orbital system
    with several active channels, some directions must fall under the
    dense_fraction threshold (sparse kernel) and at least one must fall
    over it (dense fallback) -- otherwise the tests further down would
    silently only cover one branch, exactly the gap this file exists to
    close."""
    g = geometry.honeycomb_lattice().get_supercell(7)
    _, _, pairs, n = _build_vj_matrices(g)
    threshold = DENSE_FRACTION * n * n
    npairs = {d: len(rows) for d, (rows, cols) in pairs.items()}
    assert any(0 < c <= threshold for c in npairs.values()), npairs
    assert any(c > threshold for c in npairs.values()), npairs


def test_sparse_density_matrix_matches_dense_at_requested_positions():
    """full_dm_accumulate_sparse (used for every SCF iteration) must agree
    with the plain dense full_dm_accumulate at every position it actually
    populates, for both the genuinely-sparse and the dense-fallback
    directions."""
    g = geometry.honeycomb_lattice().get_supercell(7)
    h1, v_dirs, pairs, n = _build_vj_matrices(g)
    nk = 4
    dense = full_dm_accumulate(h1, ds=list(v_dirs), nk=nk, delta=1e-6)
    sparse = full_dm_accumulate_sparse(h1, pairs, nk=nk, delta=1e-6)
    for d in v_dirs:
        rows, cols = pairs[d]
        if len(rows) == 0:
            continue
        diff = np.max(np.abs(dense[d][rows, cols] - sparse[d][rows, cols]))
        assert diff < 1e-9, (d, diff)


def test_sparse_fermi_dedup_matches_two_diagonalization_reference():
    """full_dm_accumulate_sparse_with_fermi (diagonalizes once) must match
    the old get_fermi4filling + full_dm_accumulate_sparse sequence
    (diagonalizes twice) to numerical precision, in both its normal
    (hold-the-whole-mesh) mode and its max_memory_gb fallback mode."""
    g = geometry.honeycomb_lattice().get_supercell(7)
    h1, v_dirs, pairs, n = _build_vj_matrices(g)
    filling, nk = 0.3, 4

    fermi_ref = h1.get_fermi4filling(filling, nk=nk)
    h_shifted = h1.copy()
    h_shifted.shift_fermi(-fermi_ref)
    dm_ref = full_dm_accumulate_sparse(h_shifted, pairs, nk=nk, delta=1e-6)

    dm_combined, fermi_combined = full_dm_accumulate_sparse_with_fermi(
            h1, pairs, filling, nk=nk, delta=1e-6)
    dm_fallback, fermi_fallback = full_dm_accumulate_sparse_with_fermi(
            h1, pairs, filling, nk=nk, delta=1e-6, max_memory_gb=1e-12)

    assert abs(fermi_ref - fermi_combined) < 1e-9
    assert abs(fermi_ref - fermi_fallback) < 1e-9
    for d in v_dirs:
        assert np.max(np.abs(dm_ref[d] - dm_combined[d])) < 1e-8, d
        assert np.max(np.abs(dm_ref[d] - dm_fallback[d])) < 1e-8, d


def test_vjinteraction_scf_dm_is_a_complete_density_matrix():
    """scf.dm is a public field (Vinteraction/SzSz/SxSx/SySy's SCF objects
    always expose a fully dense one); VJinteraction/Jinteraction's must be
    too, even though the SCF loop internally only ever populates the
    sparse subset it needs for its own use. Uses a fixed, small maxite
    (not full convergence) purely to keep this fast -- the point is
    checking the shape/completeness of the returned dm, not physical
    convergence, which is already covered elsewhere."""
    g = geometry.honeycomb_lattice().get_supercell(7)
    h = g.get_hamiltonian(has_spin=True)
    scf = meanfield.VJinteraction(h, V1=0.3, U=1.0, J1=-0.3, J2=0.1,
            J1x=0.05, J1y=0.05, mf="ferroZ", nk=4, maxerror=1e-6, mix=0.3,
            maxite=3, filling=0.3, verbose=0)

    _, v_dirs, _, n = _build_vj_matrices(g)
    dm_ref = get_dm(scf.hamiltonian, v_dirs, nk=4, T=1e-7, integration="ed")

    for d in v_dirs:
        assert np.max(np.abs(scf.dm[d] - dm_ref[d])) < 1e-8, d
    nnz_scf = sum(np.count_nonzero(np.abs(scf.dm[d]) > 1e-14) for d in v_dirs)
    nnz_ref = sum(np.count_nonzero(np.abs(dm_ref[d]) > 1e-14) for d in v_dirs)
    # not just the handful of entries the sparse SCF loop itself needed
    assert nnz_scf > 0.5 * nnz_ref, (nnz_scf, nnz_ref)
