"""Full end-to-end golden test for ``backend="python"`` through the public
``wannier90.setup()``/``run()`` API -- the python-backend equivalent of
``test_gaas.py``'s ``test_gaas_fully_in_memory``. Everything below this call
site (kmesh, projections, disentangle, wannierise) is unit-tested more
narrowly in the other ``test_engine_*.py`` files; this test exists to catch
wiring bugs in ``api.py``'s backend dispatch itself.
"""
import numpy as np

from conftest import GAAS_EXCLUDE_BANDS, GAAS_PROJECTIONS, GAAS_WIN_KEYWORDS, UPSTREAM_TESTDIR


def test_run_python_backend_end_to_end_matches_upstream_reference(gaas_case):
    import wannier90
    from wannier90 import io_helpers

    setup_result = wannier90.setup(
        "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
        gaas_case.symbols, gaas_case.atoms_cart, win_keywords=GAAS_WIN_KEYWORDS,
        exclude_bands=GAAS_EXCLUDE_BANDS, projections=GAAS_PROJECTIONS,
        gamma_only=gaas_case.gamma_only, spinors=gaas_case.spinors, backend="python",
    )
    num_kpts = int(np.prod(gaas_case.mp_grid))
    M_matrix = io_helpers.read_mmn(
        UPSTREAM_TESTDIR / "gaas.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
        setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(
        UPSTREAM_TESTDIR / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann,
    )
    eigenvalues = io_helpers.read_eig(UPSTREAM_TESTDIR / "EIG", setup_result.num_bands, num_kpts)

    run_result = wannier90.run(
        "gaas", setup_result, gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice,
        gaas_case.symbols, gaas_case.atoms_cart, M_matrix, A_matrix, eigenvalues,
        gamma_only=gaas_case.gamma_only,
    )

    ref = np.loadtxt(UPSTREAM_TESTDIR / "ref" / "results_ref.dat")
    got = np.column_stack([run_result.wann_centres.T, run_result.wann_spreads])
    np.testing.assert_allclose(got, ref, atol=1e-5)
