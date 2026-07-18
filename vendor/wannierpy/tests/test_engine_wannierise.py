"""End-to-end parity test for the full pure-Python pipeline
(kmesh -> disentangle -> wannierise) against upstream Wannier90's own golden
reference for the GaAs case (``ref/results_ref.dat``, the same file
``test_gaas.py`` checks the fortran backend against) and against the
compiled extension's ``RunResult``.

As in ``test_engine_disentangle.py``, the final ``U_matrix`` is only
defined up to a gauge choice (here inherited from dis_main's internal
disentanglement gauge, which already differs from the Fortran backend's --
see that module) -- so only the gauge-invariant physical outputs (Wannier
centres, spreads, spread_total/invariant/tilde) are compared, not the raw
U_matrix.
"""
import shutil

import numpy as np

from conftest import UPSTREAM_TESTDIR


def test_wann_main_matches_upstream_reference(gaas_case, tmp_path):
    import wannier90
    from wannier90 import io_helpers
    from wannier90._engine.disentangle import dis_main
    from wannier90._engine.kmesh import kmesh_get
    from wannier90._engine.wannierise import wann_main

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

    _, u_matrix, _, M_wann, dis_converged = dis_main(
        A_matrix, M_matrix, eigenvalues, kmesh.nnlist, kmesh.wb, num_wann=setup_result.num_wann,
        dis_win_min=dis_win_min, dis_win_max=24.0, frozen_states=True,
        dis_froz_min=dis_win_min, dis_froz_max=14.0,
        dis_num_iter=1200, dis_mix_ratio=1.0, dis_conv_tol=1e-10, dis_conv_window=3,
    )
    assert dis_converged

    U_final, wann_centres, wann_spreads, om_tot, om_i, om_tilde, wann_converged = wann_main(
        u_matrix, M_wann, kmesh.nnlist, kmesh.bk, kmesh.wb,
        num_iter=1000, num_cg_steps=5, conv_tol=1e-10, conv_window=3, trial_step=2.0,
    )
    assert wann_converged

    num_wann = setup_result.num_wann
    for k in range(num_kpts):
        np.testing.assert_allclose(U_final[:, :, k].conj().T @ U_final[:, :, k], np.eye(num_wann), atol=1e-8)

    ref = np.loadtxt(UPSTREAM_TESTDIR / "ref" / "results_ref.dat")
    got = np.column_stack([wann_centres.T, wann_spreads])
    np.testing.assert_allclose(got, ref, atol=1e-5)

    np.testing.assert_allclose(om_tot, run_result.spread_total, atol=1e-5)
    np.testing.assert_allclose(om_i, run_result.spread_invariant, atol=1e-5)
    np.testing.assert_allclose(om_tilde, run_result.spread_tilde, atol=1e-5)
