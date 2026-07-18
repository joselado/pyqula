"""Shared fixtures for the python-backend engine tests
(``test_engine_kmesh.py``, ``test_engine_setup.py``). ``test_gaas.py`` has
its own independent readers -- it's the pre-existing golden/end-to-end test
for the fortran backend and deliberately left untouched.
"""
import re
from pathlib import Path

import numpy as np
import pytest

UPSTREAM_TESTDIR = (
    Path(__file__).resolve().parents[2]
    / "wannier90-3.1.0" / "test-suite" / "library-mode-test"
)

GAAS_WIN_KEYWORDS = {
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
GAAS_EXCLUDE_BANDS = range(1, 6)
GAAS_PROJECTIONS = [
    "f=0.25,0.25,0.25 : s",
    "f=0.25,0.25,0.25 : p",
    "f= 0.0, 0.0, 0.0 : p",
    "f= 0.0, 0.0, 0.0 : s",
]


class GaasCase:
    def __init__(self):
        d = UPSTREAM_TESTDIR
        text = (d / "PARAMS").read_text()
        self.mp_grid = np.array(
            [int(re.search(rf"mp_grid_loc\({i}\)\s*=\s*(\d+)", text).group(1)) for i in (1, 2, 3)],
            dtype=np.int32,
        )
        self.num_bands_tot = int(re.search(r"num_bands_tot\s*=\s*(\d+)", text).group(1))
        self.gamma_only = "true" in re.search(r"gamma_only_loc\s*=\s*\.(\w+)\.", text).group(1).lower()
        self.spinors = "true" in re.search(r"spinors_loc\s*=\s*\.(\w+)\.", text).group(1).lower()
        self.real_lattice = np.array(
            [list(map(float, line.split())) for line in (d / "CELL").read_text().splitlines() if line.strip()]
        )
        num_kpts = int(np.prod(self.mp_grid))
        self.kpt_latt = np.array(
            [list(map(float, line.split())) for line in (d / "KPOINTS").read_text().splitlines() if line.split()]
        ).T
        assert self.kpt_latt.shape == (3, num_kpts)
        lines = [line for line in (d / "POSITIONS").read_text().splitlines() if line.strip()]
        num_atoms = int(lines[0])
        self.symbols = lines[1:1 + num_atoms]
        self.atoms_cart = np.array(
            [list(map(float, line.split())) for line in lines[1 + num_atoms:1 + 2 * num_atoms]]
        ).T


@pytest.fixture(scope="module")
def gaas_case():
    if not UPSTREAM_TESTDIR.exists():
        pytest.skip(f"upstream test fixtures not found at {UPSTREAM_TESTDIR}")
    return GaasCase()
