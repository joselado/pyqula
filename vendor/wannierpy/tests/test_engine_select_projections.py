"""Parity test for ``select_projections`` (choosing/reordering a subset of
a larger projections block) against the fortran backend on GaAs.

Regression coverage for a real bug found while implementing this: the
shared range-vector parser (``parse_range_vector``) sorts and dedupes its
output, which is correct for ``exclude_bands``/``shell_list`` (unordered
sets) but wrong for ``select_projections`` -- entry *j* there names which
projection Wannier function *j* gets, so parsing "8,1,2,..." must preserve
that order, not return "1,2,...,8". See ``parse_range_vector_ordered`` in
``_engine/params.py``.
"""
import shutil

import numpy as np

from conftest import GAAS_EXCLUDE_BANDS, GAAS_WIN_KEYWORDS, UPSTREAM_TESTDIR

# Same 8 projections as GAAS_PROJECTIONS, just reordered/relabelled 1-8 so a
# non-trivial (non-identity) select_projections permutation is meaningful.
EXTRA_PROJECTIONS = [
    "f=0.25,0.25,0.25 : s",  # 1
    "f=0.25,0.25,0.25 : p",  # 2,3,4
    "f= 0.0, 0.0, 0.0 : p",  # 5,6,7
    "f= 0.0, 0.0, 0.0 : s",  # 8
]
SELECT_PROJECTIONS = "8,1,2,3,4,5,6,7"


def test_select_projections_order_preserved(gaas_case):
    import wannier90

    win_keywords = dict(GAAS_WIN_KEYWORDS, select_projections=SELECT_PROJECTIONS)
    setup_result = wannier90.setup(
        "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
        gaas_case.symbols, gaas_case.atoms_cart, win_keywords=win_keywords,
        exclude_bands=GAAS_EXCLUDE_BANDS, projections=EXTRA_PROJECTIONS,
        gamma_only=gaas_case.gamma_only, spinors=gaas_case.spinors, backend="python",
    )
    # slot 0 should get projection #8 (f=0,0,0 : s -> site (0,0,0), l=0), not
    # projection #1 (f=0.25,0.25,0.25 : s) -- the bug this test guards against
    # would silently sort select_projections back to "1,2,...,8".
    np.testing.assert_allclose(setup_result.proj_site[:, 0], [0.0, 0.0, 0.0])
    assert setup_result.proj_l[0] == 0


def test_select_projections_matches_fortran(gaas_case, tmp_path):
    import wannier90
    from wannier90 import io_helpers

    for name in ["gaas.win", "PARAMS", "CELL", "KPOINTS", "POSITIONS", "EIG", "gaas.mmn", "gaas.amn"]:
        shutil.copy(UPSTREAM_TESTDIR / name, tmp_path / name)

    win_keywords = dict(GAAS_WIN_KEYWORDS, select_projections=SELECT_PROJECTIONS)
    num_kpts = int(np.prod(gaas_case.mp_grid))

    results = {}
    for backend in ("fortran", "python"):
        cwd = str(tmp_path) if backend == "fortran" else None
        setup_result = wannier90.setup(
            "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
            gaas_case.symbols, gaas_case.atoms_cart, win_keywords=win_keywords,
            exclude_bands=GAAS_EXCLUDE_BANDS, projections=EXTRA_PROJECTIONS,
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
    np.testing.assert_allclose(python.spread_total, fortran.spread_total, atol=1e-5)
    np.testing.assert_allclose(python.wann_centres, fortran.wann_centres, atol=1e-4)
    np.testing.assert_allclose(python.wann_spreads, fortran.wann_spreads, atol=1e-4)
