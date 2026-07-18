"""Parity test for spinor projections (``(u)``/``(d)`` spin channel syntax,
``[qaxis]`` quantisation direction) against the fortran backend's
``wannier_setup`` on GaAs's geometry.

Only ``setup()`` is exercised (not a full disentangle+wannierise run):
``spinors`` affects projection parsing/output-array shapes at the
``wannier_setup`` level, but validating the actual spinor *physics* through
``wannier_run`` would need genuinely spin-orbit-coupled DFT overlap data,
which isn't available for GaAs in this repo's fixtures.
"""
import shutil

import numpy as np

from conftest import UPSTREAM_TESTDIR

SPINOR_PROJECTIONS = [
    "f=0.25,0.25,0.25 : s (u)[0,0,1]",
    "f=0.25,0.25,0.25 : s (d)[0,0,1]",
    "f= 0.0, 0.0, 0.0 : s",  # no (u)/(d) -> both channels, expands to 2
]


def test_spinor_projections_match_fortran(gaas_case, tmp_path):
    import wannier90

    shutil.copy(UPSTREAM_TESTDIR / "gaas.win", tmp_path / "gaas.win")
    win_keywords = {"num_wann": 4}

    results = {}
    for backend in ("fortran", "python"):
        cwd = str(tmp_path) if backend == "fortran" else None
        results[backend] = wannier90.setup(
            "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
            gaas_case.symbols, gaas_case.atoms_cart, win_keywords=win_keywords,
            projections=SPINOR_PROJECTIONS, gamma_only=False, spinors=True, backend=backend, cwd=cwd,
        )

    fortran, python = results["fortran"], results["python"]
    assert python.num_wann == fortran.num_wann == 4
    n = python.num_wann  # fortran pads proj_* arrays to num_bands_tot; only the first n entries are meaningful

    np.testing.assert_allclose(python.proj_site, fortran.proj_site[:, :n])
    np.testing.assert_array_equal(python.proj_l, fortran.proj_l[:n])
    np.testing.assert_array_equal(python.proj_m, fortran.proj_m[:n])
    np.testing.assert_array_equal(python.proj_s, fortran.proj_s[:n])
    np.testing.assert_allclose(python.proj_s_qaxis, fortran.proj_s_qaxis[:, :n])

    # explicit (u)/(d) on the first site; default (both channels) on the second
    np.testing.assert_array_equal(python.proj_s, [1, -1, 1, -1])
