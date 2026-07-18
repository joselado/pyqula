"""Parity test for the ``precond`` preconditioned CG search direction
against the fortran backend, on real DFT data referenced in place from
``test-suite/tests/testw90_precond_1`` (GaAs, 4 bands = 4 Wannier
functions, no disentanglement).

Exercises new machinery with a real correctness question that had no a
priori answer: ``_engine.ws_vectors.wigner_seitz_vectors`` (real-space
R-vector search for the Fourier filter) and, in particular, the exact
matrix convention of Fortran's ``rvec_cart = matmul(real_lattice, irvec)``
(literally transcribed in ``_precond_direction`` -- worth flagging because
it doesn't match the "cart = frac @ real_lattice" convention used
everywhere else in this port, e.g. ``utility_frac_to_cart``; it's
transcribed as-is rather than "corrected" to that convention, and validated
here rather than derived by hand).
"""
import shutil

import numpy as np

from conftest import UPSTREAM_TESTDIR

PRECOND_DIR = UPSTREAM_TESTDIR.parents[1] / "test-suite" / "tests" / "testw90_precond_1"


def test_precond_matches_fortran(tmp_path):
    import wannier90
    from wannier90 import io_helpers

    if not PRECOND_DIR.exists():
        import pytest
        pytest.skip(f"upstream test fixtures not found at {PRECOND_DIR}")

    real_lattice = np.array([
        [-5.367, 0.0, 5.367], [0.0, 5.367, 5.367], [-5.367, 5.367, 0.0],
    ]) * 0.5291772109  # bohr -> Angstrom
    symbols = ["Ga", "As"]
    atoms_frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    atoms_cart = (atoms_frac @ real_lattice).T
    mp_grid = np.array([2, 2, 2], dtype=np.int32)
    kpt_latt = np.array([
        [0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5],
        [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 0.5],
    ]).T
    num_kpts = 8
    win_keywords = {"num_wann": 4, "num_iter": 40, "precond": True, "search_shells": 12}
    projections = ["f=0.25,0.25,0.25:sp3"]

    shutil.copy(PRECOND_DIR / "gaas1.win", tmp_path / "gaas1.win")

    results = {}
    for backend in ("fortran", "python"):
        cwd = str(tmp_path) if backend == "fortran" else None
        setup_result = wannier90.setup(
            "gaas1", mp_grid, kpt_latt, real_lattice, 4, symbols, atoms_cart,
            win_keywords=win_keywords, projections=projections, backend=backend, cwd=cwd,
        )
        assert setup_result.num_bands == setup_result.num_wann == 4

        M_matrix = io_helpers.read_mmn(
            PRECOND_DIR / "gaas1.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
            setup_result.nnlist, setup_result.nncell,
        )
        A_matrix = io_helpers.read_amn(
            PRECOND_DIR / "gaas1.amn", setup_result.num_bands, num_kpts, setup_result.num_wann,
        )
        eigenvalues = np.zeros((setup_result.num_bands, num_kpts))  # never read: no disentanglement here

        run_kwargs = dict(cwd=str(tmp_path)) if backend == "fortran" else {}
        results[backend] = wannier90.run(
            "gaas1", setup_result, mp_grid, kpt_latt, real_lattice, symbols, atoms_cart,
            M_matrix, A_matrix, eigenvalues, backend=backend, **run_kwargs,
        )

    fortran, python = results["fortran"], results["python"]
    np.testing.assert_allclose(python.spread_total, fortran.spread_total, atol=1e-6)
    np.testing.assert_allclose(python.spread_total, 4.466880976, atol=1e-4)  # benchmark.out's own value
    np.testing.assert_allclose(python.wann_centres, fortran.wann_centres, atol=1e-4)
    np.testing.assert_allclose(python.wann_spreads, fortran.wann_spreads, atol=1e-4)
