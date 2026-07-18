"""Second real-data validation case for the python backend, complementing
GaAs: diamond, from upstream Wannier90's own test-suite
(``test-suite/tests/testw90_example05``, referenced in place -- not
vendored into this repo, matching ``conftest.py``'s ``UPSTREAM_TESTDIR``
convention).

Diamond exercises paths GaAs doesn't:
* ``num_bands == num_wann`` (no disentanglement) with *real* DFT overlaps --
  closes the gap ``test_engine_run_no_disentangle.py`` could only cover with
  synthetic data.
* A different (FCC-diamond) lattice/k-mesh, cross-checking ``kmesh_get``'s
  shell search on a second geometry.

That test-suite directory is built for the ``wannier90.x`` CLI (structural
data lives in the ``.win`` file's ``unit_cell_cart``/``atoms_frac``/
``kpoints`` blocks), not library mode (where it's the caller's job to
supply that data as arguments, and library mode reads-but-ignores those
blocks) -- so this file's ``_parse_win_block`` extracts just those three
blocks itself, standing in for what a real library-mode caller (e.g. a DFT
code's Wannier90 interface) would already have from its own SCF run.
``benchmark.out`` is that test's shipped ``wannier90.x`` reference log,
checked into upstream -- ``_parse_final_state`` reads its "Final State"
block the same way a human would.
"""
import re
from pathlib import Path

import numpy as np
import pytest

DIAMOND_DIR = (
    Path(__file__).resolve().parents[2]
    / "wannier90-3.1.0" / "test-suite" / "tests" / "testw90_example05"
)


def _parse_win_block(text: str, name: str) -> list[str]:
    m = re.search(rf"begin\s+{name}\s*\n(.*?)\nend\s+{name}", text, re.IGNORECASE | re.DOTALL)
    return [line for line in m.group(1).splitlines() if line.strip()]


def _load_diamond():
    text = (DIAMOND_DIR / "diamond.win").read_text()

    cell_lines = _parse_win_block(text, "unit_cell_cart")
    real_lattice = np.array([list(map(float, line.split())) for line in cell_lines])

    atom_lines = _parse_win_block(text, "atoms_frac")
    symbols = [line.split()[0] for line in atom_lines]
    atoms_frac = np.array([list(map(float, line.split()[1:4])) for line in atom_lines]).T
    atoms_cart = (atoms_frac.T @ real_lattice).T

    kpt_lines = _parse_win_block(text, "kpoints")
    kpt_latt = np.array([list(map(float, line.split())) for line in kpt_lines]).T

    mp_grid = np.array(
        [int(x) for x in re.search(r"mp_grid\s*[:=]\s*(\d+)\s+(\d+)\s+(\d+)", text).groups()],
        dtype=np.int32,
    )
    num_wann = int(re.search(r"num_wann\s*=\s*(\d+)", text).group(1))

    return real_lattice, symbols, atoms_cart, kpt_latt, mp_grid, num_wann


def _parse_final_state(text: str):
    """Rows of (x, y, z, spread) parsed straight from ``benchmark.out``'s
    "Final State" block -- not hand-transcribed."""
    block = re.search(r"Final State\n(.*?)Sum of centres", text, re.DOTALL).group(1)
    rows = [list(map(float, re.findall(r"[-\d.]+", line)))[1:] for line in block.strip().splitlines()]
    return np.array(rows)


@pytest.fixture(scope="module")
def diamond_case():
    if not DIAMOND_DIR.exists():
        pytest.skip(f"upstream test fixtures not found at {DIAMOND_DIR}")
    return _load_diamond()


def test_diamond_no_disentanglement_matches_upstream_benchmark(diamond_case):
    import wannier90
    from wannier90 import io_helpers

    real_lattice, symbols, atoms_cart, kpt_latt, mp_grid, num_wann = diamond_case
    num_kpts = int(np.prod(mp_grid))

    setup_result = wannier90.setup(
        "diamond", mp_grid, kpt_latt, real_lattice, num_wann, symbols, atoms_cart,
        win_keywords={"num_wann": num_wann, "num_iter": 20, "search_shells": 12},
        projections=["f=0.0,0.0,0.0:s", "f=0.0,0.0,0.5:s", "f=0.0,0.5,0.0:s", "f=0.5,0.0,0.0:s"],
        backend="python",
    )
    assert setup_result.num_bands == setup_result.num_wann == num_wann == 4

    M_matrix = io_helpers.read_mmn(
        DIAMOND_DIR / "diamond.mmn", setup_result.num_bands, num_kpts, setup_result.nntot,
        setup_result.nnlist, setup_result.nncell,
    )
    A_matrix = io_helpers.read_amn(DIAMOND_DIR / "diamond.amn", setup_result.num_bands, num_kpts, num_wann)
    eigenvalues = io_helpers.read_eig(DIAMOND_DIR / "diamond.eig", setup_result.num_bands, num_kpts)

    run_result = wannier90.run(
        "diamond", setup_result, mp_grid, kpt_latt, real_lattice, symbols, atoms_cart,
        M_matrix, A_matrix, eigenvalues,
    )

    ref = _parse_final_state((DIAMOND_DIR / "benchmark.out.default.inp=diamond.win").read_text())
    got = np.column_stack([run_result.wann_centres.T, run_result.wann_spreads])

    # Wannier centres/spreads for a set of symmetry-equivalent bonds are only
    # defined as a set (the four sp3 bonds are physically interchangeable,
    # and nothing pins a particular column of U_matrix to a particular one) --
    # so match rows by nearest-centre rather than assuming benchmark order.
    used = set()
    for row in got:
        dists = np.linalg.norm(ref[:, :3] - row[:3], axis=1)
        for i in used:
            dists[i] = np.inf
        best = int(np.argmin(dists))
        used.add(best)
        np.testing.assert_allclose(row, ref[best], atol=1e-4)

    np.testing.assert_allclose(run_result.spread_total, 2.320904915, atol=1e-5)
    np.testing.assert_allclose(run_result.spread_invariant, 1.954619860, atol=1e-5)
    np.testing.assert_allclose(run_result.spread_tilde, 0.366285055, atol=1e-5)
