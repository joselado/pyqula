"""Parity test for selective localization with centre constraints
(``slwf_num``/``slwf_constrain``/``slwf_lambda``/``slwf_centres`` -- the
"SLWF+C" method, Vitale et al., PRB 90, 165125 (2018)), against both
upstream's own benchmark and the fortran backend, on real DFT data
referenced in place from ``test-suite/tests/testw90_example26`` (GaAs,
4 bands = 4 Wannier functions, no disentanglement, only the first Wannier
function is "objective"/selectively localized with a centre constraint).

This is the highest-risk hand-transcribed piece of the port so far
(``wann_domega``'s selective_loc branch has several terms that
algebraically cancel -- see the comment in ``wannierise.py`` -- and getting
the cancellation "by hand" while porting would have been easy to get
subtly wrong), so it's checked against three independent things: the raw
benchmark text, the fortran backend run through this same harness, and
internal cross-consistency between the two.

That test-suite directory is a ``wannier90.x``-style one (structural data
in the ``.win`` file's blocks, no ``.eig`` file since it's never read
without disentanglement) -- see ``test_engine_diamond.py``'s module
docstring for why this file parses the ``.win`` blocks itself rather than
using library-mode-test's PARAMS/CELL/KPOINTS/POSITIONS convention.
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

EXAMPLE26_DIR = (
    Path(__file__).resolve().parents[2]
    / "wannier90-3.1.0" / "test-suite" / "tests" / "testw90_example26"
)

WIN_KEYWORDS = {
    "num_wann": 4, "num_iter": 100, "search_shells": 12,
    "slwf_num": 1, "slwf_constrain": True, "slwf_lambda": 1,
}
EXCLUDE_BANDS = "1-5,10-18"
NUM_BANDS_TOT = 18  # gaas.amn/.mmn are pre-filtered to 4 bands; exclude_bands must account for the other 14
PROJECTIONS = [
    "f= 0.125, 0.125, 0.125: s", "f= 0.125, 0.125, -.375: s",
    "f= -.375, 0.125, 0.125: s", "f= 0.125, -.375, 0.125: s",
]
SLWF_CENTRES = ["1 0.25 0.25 0.25"]

BENCHMARK_SPREAD_TOTAL = 1.634087565
BENCHMARK_CENTRES_SPREADS = np.array([
    [-1.412902, 1.412902, 1.412902, 1.63408756],
    [1.239663, -1.239663, 1.074049, 2.74772657],
    [1.239663, 1.074049, -1.239663, 2.74772657],
    [-1.074049, -1.239663, -1.239663, 2.74772657],
])


def _parse_win_block(text: str, name: str) -> list[str]:
    m = re.search(rf"begin\s+{name}\s*\n(.*?)\nend\s+{name}", text, re.IGNORECASE | re.DOTALL)
    return [line for line in m.group(1).splitlines() if line.strip()]


@pytest.fixture(scope="module")
def example26_case():
    if not EXAMPLE26_DIR.exists():
        pytest.skip(f"upstream test fixtures not found at {EXAMPLE26_DIR}")
    text = (EXAMPLE26_DIR / "gaas.win").read_text()

    cell_lines = _parse_win_block(text, "unit_cell_cart")
    unit = cell_lines[0].strip().lower()
    if unit in ("ang", "bohr"):
        cell_lines = cell_lines[1:]
    real_lattice = np.array([list(map(float, line.split())) for line in cell_lines])
    if unit == "bohr":
        real_lattice *= 0.5291772109

    atom_lines = _parse_win_block(text, "atoms_frac")
    symbols = [line.split()[0] for line in atom_lines]
    atoms_frac = np.array([list(map(float, line.split()[1:4])) for line in atom_lines]).T
    atoms_cart = (atoms_frac.T @ real_lattice).T

    kpt_lines = _parse_win_block(text, "kpoints")
    kpt_latt = np.array([list(map(float, line.split()[:3])) for line in kpt_lines]).T
    mp_grid = np.array(
        [int(x) for x in re.search(r"mp_grid\s*[:=]\s*(\d+)\s+(\d+)\s+(\d+)", text).groups()], dtype=np.int32
    )
    return real_lattice, symbols, atoms_cart, kpt_latt, mp_grid


def _run(backend, cwd, example26_case):
    import wannier90
    from wannier90 import io_helpers

    real_lattice, symbols, atoms_cart, kpt_latt, mp_grid = example26_case
    num_kpts = int(np.prod(mp_grid))

    setup_kwargs = dict(cwd=str(cwd)) if backend == "fortran" else {}
    setup_result = wannier90.setup(
        "gaas", mp_grid, kpt_latt, real_lattice, NUM_BANDS_TOT, symbols, atoms_cart,
        win_keywords=WIN_KEYWORDS, exclude_bands=EXCLUDE_BANDS, projections=PROJECTIONS,
        slwf_centres=SLWF_CENTRES, backend=backend, **setup_kwargs,
    )
    assert setup_result.num_bands == setup_result.num_wann == 4

    M_matrix = io_helpers.read_mmn(
        EXAMPLE26_DIR / "gaas.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
        setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(EXAMPLE26_DIR / "gaas.amn", setup_result.num_bands, num_kpts, setup_result.num_wann)
    eigenvalues = np.zeros((setup_result.num_bands, num_kpts))  # never read: no disentanglement here

    run_kwargs = dict(cwd=str(cwd)) if backend == "fortran" else {}
    return wannier90.run(
        "gaas", setup_result, mp_grid, kpt_latt, real_lattice, symbols, atoms_cart,
        M_matrix, A_matrix, eigenvalues, backend=backend, **run_kwargs,
    )


def test_selective_localization_matches_upstream_and_fortran(example26_case, tmp_path):
    shutil.copy(EXAMPLE26_DIR / "gaas.win", tmp_path / "gaas.win")

    fortran = _run("fortran", tmp_path, example26_case)
    python = _run("python", tmp_path, example26_case)

    for label, result in (("fortran", fortran), ("python", python)):
        got = np.column_stack([result.wann_centres.T, result.wann_spreads])
        # WF 2-4 are symmetry-equivalent (only their sum is pinned by the single
        # constraint on WF 1), so match by nearest centre rather than assuming order.
        used = set()
        for row in got:
            dists = np.linalg.norm(BENCHMARK_CENTRES_SPREADS[:, :3] - row[:3], axis=1)
            for i in used:
                dists[i] = np.inf
            best = int(np.argmin(dists))
            used.add(best)
            np.testing.assert_allclose(row, BENCHMARK_CENTRES_SPREADS[best], atol=1e-4, err_msg=label)
        np.testing.assert_allclose(result.spread_total, BENCHMARK_SPREAD_TOTAL, atol=1e-5, err_msg=label)

    np.testing.assert_allclose(python.spread_total, fortran.spread_total, atol=1e-5)
