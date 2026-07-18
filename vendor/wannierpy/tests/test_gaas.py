"""Golden integration test: reproduces the GaAs reference case shipped with
upstream Wannier90 at test-suite/library-mode-test/, using this package's
public API instead of the reference Fortran caller (test_library.F90).
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import wannier90
from wannier90 import io_helpers

UPSTREAM_TESTDIR = (
    Path(__file__).resolve().parents[2]
    / "wannier90-3.1.0" / "test-suite" / "library-mode-test"
)

FIXTURE_FILES = [
    "gaas.win", "PARAMS", "CELL", "KPOINTS", "POSITIONS",
    "EIG", "gaas.mmn", "gaas.amn",
]


def _read_params(path):
    text = path.read_text()
    mp_grid = [int(re.search(rf"mp_grid_loc\({i}\)\s*=\s*(\d+)", text).group(1)) for i in (1, 2, 3)]
    num_bands_tot = int(re.search(r"num_bands_tot\s*=\s*(\d+)", text).group(1))
    gamma_only = "true" in re.search(r"gamma_only_loc\s*=\s*\.(\w+)\.", text).group(1).lower()
    spinors = "true" in re.search(r"spinors_loc\s*=\s*\.(\w+)\.", text).group(1).lower()
    return np.array(mp_grid, dtype=np.int32), num_bands_tot, gamma_only, spinors


def _read_cell(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=np.float64)


def _read_kpoints(path, num_kpts):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.split()]
    assert len(rows) == num_kpts
    return np.array(rows, dtype=np.float64).T


def _read_positions(path):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    num_atoms = int(lines[0])
    symbols = lines[1:1 + num_atoms]
    coords = [list(map(float, l.split())) for l in lines[1 + num_atoms:1 + 2 * num_atoms]]
    return symbols, np.array(coords, dtype=np.float64).T


@pytest.fixture(scope="module")
def gaas_dir(tmp_path_factory):
    if not UPSTREAM_TESTDIR.exists():
        pytest.skip(f"upstream test fixtures not found at {UPSTREAM_TESTDIR}")
    d = tmp_path_factory.mktemp("gaas")
    for name in FIXTURE_FILES:
        shutil.copy(UPSTREAM_TESTDIR / name, d / name)
    return d


@pytest.mark.parametrize("in_process", [True, False], ids=["in_process", "subprocess"])
def test_gaas_matches_reference(gaas_dir, in_process):
    mp_grid, num_bands_tot, gamma_only, spinors = _read_params(gaas_dir / "PARAMS")
    real_lattice = _read_cell(gaas_dir / "CELL")
    num_kpts = int(np.prod(mp_grid))
    kpt_latt = _read_kpoints(gaas_dir / "KPOINTS", num_kpts)
    symbols, atoms_cart = _read_positions(gaas_dir / "POSITIONS")

    setup_result = wannier90.setup(
        "gaas", mp_grid, kpt_latt, real_lattice, num_bands_tot, symbols, atoms_cart,
        gamma_only=gamma_only, spinors=spinors, cwd=str(gaas_dir), in_process=in_process,
    )
    assert setup_result.num_bands == 12
    assert setup_result.num_wann == 8

    M_matrix = io_helpers.read_mmn(
        gaas_dir / "gaas.mmn", setup_result.num_bands, num_kpts,
        setup_result.nntot, setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(
        gaas_dir / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann,
    )
    eigenvalues = io_helpers.read_eig(gaas_dir / "EIG", setup_result.num_bands, num_kpts)

    run_result = wannier90.run(
        "gaas", setup_result, mp_grid, kpt_latt, real_lattice, symbols, atoms_cart,
        M_matrix, A_matrix, eigenvalues,
        gamma_only=gamma_only, cwd=str(gaas_dir), in_process=in_process,
    )

    ref = np.loadtxt(UPSTREAM_TESTDIR / "ref" / "results_ref.dat")
    got = np.column_stack([run_result.wann_centres.T, run_result.wann_spreads])
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_gaas_fully_in_memory():
    """Same GaAs case as above, but with none of gaas.win/a pre-made run
    directory involved: the .win content comes from Python data structures
    (win_keywords/exclude_bands/projections), and cwd is left unset so
    setup() creates its own scratch directory. Only the .mmn/.amn/.eig
    (upstream DFT-interface output -- not something this package generates)
    are still read from disk, via io_helpers, straight into arrays."""
    if not UPSTREAM_TESTDIR.exists():
        pytest.skip(f"upstream test fixtures not found at {UPSTREAM_TESTDIR}")

    mp_grid, num_bands_tot, gamma_only, spinors = _read_params(UPSTREAM_TESTDIR / "PARAMS")
    real_lattice = _read_cell(UPSTREAM_TESTDIR / "CELL")
    num_kpts = int(np.prod(mp_grid))
    kpt_latt = _read_kpoints(UPSTREAM_TESTDIR / "KPOINTS", num_kpts)
    symbols, atoms_cart = _read_positions(UPSTREAM_TESTDIR / "POSITIONS")

    # Equivalent to gaas.win's content -- mp_grid/num_bands/atoms_frac/kpoints
    # blocks are deliberately omitted: wannier90 reads and ignores them in
    # library mode (see write_win's docstring), since those come from the
    # setup() arguments above instead.
    win_keywords = {
        "num_wann": 8,
        "num_iter": 1000,
        "num_print_cycles": 40,
        "conv_tol": 1e-10,
        "conv_window": 3,
        "dis_win_max": 24.0,
        "dis_froz_max": 14.0,
        "dis_num_iter": 1200,
        "dis_mix_ratio": 1.0,
    }
    exclude_bands = range(1, 6)  # bands 1-5
    projections = [
        "f=0.25,0.25,0.25 : s",
        "f=0.25,0.25,0.25 : p",
        "f= 0.0, 0.0, 0.0 : p",
        "f= 0.0, 0.0, 0.0 : s",
    ]

    setup_result = wannier90.setup(
        "gaas", mp_grid, kpt_latt, real_lattice, num_bands_tot, symbols, atoms_cart,
        win_keywords=win_keywords, exclude_bands=exclude_bands, projections=projections,
        gamma_only=gamma_only, spinors=spinors,
    )
    assert setup_result.num_bands == 12
    assert setup_result.num_wann == 8

    M_matrix = io_helpers.read_mmn(
        UPSTREAM_TESTDIR / "gaas.mmn", setup_result.num_bands, num_kpts,
        setup_result.nntot, setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(
        UPSTREAM_TESTDIR / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann,
    )
    eigenvalues = io_helpers.read_eig(UPSTREAM_TESTDIR / "EIG", setup_result.num_bands, num_kpts)

    run_result = wannier90.run(
        "gaas", setup_result, mp_grid, kpt_latt, real_lattice, symbols, atoms_cart,
        M_matrix, A_matrix, eigenvalues, gamma_only=gamma_only,
    )

    ref = np.loadtxt(UPSTREAM_TESTDIR / "ref" / "results_ref.dat")
    got = np.column_stack([run_result.wann_centres.T, run_result.wann_spreads])
    np.testing.assert_allclose(got, ref, atol=1e-5)
