"""Parity test for the pure-Python disentanglement engine
(``wannier90._engine.disentangle.dis_main``) against the Fortran extension,
on the GaAs reference case (which does disentangle: 12 bands -> 8 Wannier
functions with an outer window at 24 eV and frozen window at 14 eV).

``dis_main``'s output subspace is only defined up to an arbitrary unitary
rotation among its num_wann columns (the eigenvectors of the Z-matrix/QPQ
matrix can come out in a different internal gauge from a different
eigensolver, or even a different LAPACK build, without the algorithm being
wrong) -- see the module docstring in ``disentangle.py``. So the comparison
here is against the gauge-invariant projector U_opt @ U_opt^H, not the raw
U_matrix_opt array; ``lwindow`` (which band indices fall in the outer
window) has no such ambiguity and is compared exactly.
"""
import shutil

import numpy as np

from conftest import UPSTREAM_TESTDIR


def test_dis_main_subspace_matches_fortran(gaas_case, tmp_path):
    import wannier90
    from wannier90 import io_helpers
    from wannier90._engine.disentangle import dis_main
    from wannier90._engine.kmesh import kmesh_get

    for name in ["gaas.win", "PARAMS", "CELL", "KPOINTS", "POSITIONS", "EIG", "gaas.mmn", "gaas.amn"]:
        shutil.copy(UPSTREAM_TESTDIR / name, tmp_path / name)

    setup_result = wannier90.setup(
        "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
        gaas_case.symbols, gaas_case.atoms_cart, gamma_only=gaas_case.gamma_only,
        spinors=gaas_case.spinors, cwd=str(tmp_path), backend="fortran",
    )
    num_kpts = int(np.prod(gaas_case.mp_grid))
    M_matrix = io_helpers.read_mmn(
        tmp_path / "gaas.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
        setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(tmp_path / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann)
    eigenvalues = io_helpers.read_eig(tmp_path / "EIG", setup_result.num_bands, num_kpts)

    run_result = wannier90.run(
        "gaas", setup_result, gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice,
        gaas_case.symbols, gaas_case.atoms_cart, M_matrix, A_matrix, eigenvalues,
        gamma_only=gaas_case.gamma_only, cwd=str(tmp_path), backend="fortran",
    )

    recip_lattice = io_helpers.reciprocal_lattice(gaas_case.real_lattice)
    kmesh = kmesh_get(gaas_case.kpt_latt, recip_lattice)
    dis_win_min = float(eigenvalues.min())

    u_opt, u_matrix, lwindow, M_matrix, converged = dis_main(
        A_matrix, M_matrix, eigenvalues, kmesh.nnlist, kmesh.wb, num_wann=setup_result.num_wann,
        dis_win_min=dis_win_min, dis_win_max=24.0, frozen_states=True,
        dis_froz_min=dis_win_min, dis_froz_max=14.0,
        dis_num_iter=1200, dis_mix_ratio=1.0, dis_conv_tol=1e-10, dis_conv_window=3,
    )

    assert converged
    np.testing.assert_array_equal(lwindow, run_result.lwindow)

    proj_mine = np.einsum("ilk,jlk->ijk", u_opt, u_opt.conj())
    proj_ref = np.einsum("ilk,jlk->ijk", run_result.U_matrix_opt, run_result.U_matrix_opt.conj())
    np.testing.assert_allclose(proj_mine, proj_ref, atol=1e-8)

    # u_matrix/M_matrix here are dis_main's *initial guess* for Wannierisation
    # (phase 3) -- no Fortran reference is exposed for that intermediate state
    # (RunResult.U_matrix is wann_main's final, further-optimized rotation), so
    # just check the closed-form invariants internal_find_u/_rotate_m must satisfy.
    num_kpts = u_matrix.shape[2]
    for k in range(num_kpts):
        np.testing.assert_allclose(u_matrix[:, :, k].conj().T @ u_matrix[:, :, k], np.eye(setup_result.num_wann),
                                    atol=1e-8)
