"""Parity test for ``fixed_step`` (skip the parabolic line search, always
take a fixed-length step along the CG search direction) against the
fortran backend on GaAs. Deliberately capped at a low iteration count
(fixed-step CG converges much slower than the line-searched default) --
this only needs the two backends to walk the *same* trajectory, not to
reach convergence.
"""
import shutil

import numpy as np

from conftest import GAAS_EXCLUDE_BANDS, GAAS_PROJECTIONS, GAAS_WIN_KEYWORDS, UPSTREAM_TESTDIR


def test_fixed_step_matches_fortran(gaas_case, tmp_path):
    import wannier90
    from wannier90 import io_helpers

    for name in ["gaas.win", "PARAMS", "CELL", "KPOINTS", "POSITIONS", "EIG", "gaas.mmn", "gaas.amn"]:
        shutil.copy(UPSTREAM_TESTDIR / name, tmp_path / name)

    win_keywords = dict(GAAS_WIN_KEYWORDS, fixed_step=1.5, num_iter=25)
    num_kpts = int(np.prod(gaas_case.mp_grid))

    results = {}
    for backend in ("fortran", "python"):
        cwd = str(tmp_path) if backend == "fortran" else None
        setup_result = wannier90.setup(
            "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
            gaas_case.symbols, gaas_case.atoms_cart, win_keywords=win_keywords,
            exclude_bands=GAAS_EXCLUDE_BANDS, projections=GAAS_PROJECTIONS,
            gamma_only=gaas_case.gamma_only, spinors=gaas_case.spinors, backend=backend, cwd=cwd,
        )
        M_matrix = io_helpers.read_mmn(
            tmp_path / "gaas.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
            setup_result.nnlist, setup_result.nncell,
        )
        A_matrix = io_helpers.read_amn(
            tmp_path / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann,
        )
        eigenvalues = io_helpers.read_eig(tmp_path / "EIG", setup_result.num_bands, num_kpts)
        run_kwargs = dict(cwd=str(tmp_path)) if backend == "fortran" else {}
        results[backend] = wannier90.run(
            "gaas", setup_result, gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice,
            gaas_case.symbols, gaas_case.atoms_cart, M_matrix, A_matrix, eigenvalues,
            gamma_only=gaas_case.gamma_only, backend=backend, **run_kwargs,
        )

    fortran, python = results["fortran"], results["python"]
    np.testing.assert_allclose(python.spread_total, fortran.spread_total, atol=1e-6)
    np.testing.assert_allclose(python.wann_centres, fortran.wann_centres, atol=1e-5)
    np.testing.assert_allclose(python.wann_spreads, fortran.wann_spreads, atol=1e-5)
