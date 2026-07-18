"""Parity test for the pure-Python engine's k-mesh determination
(``wannier90._engine.kmesh.kmesh_get``) against the compiled Fortran
extension's ``wannier_setup``, on the GaAs reference case.

This is the first slice of the python backend (see the porting plan) --
narrower than ``test_gaas.py``'s end-to-end golden test, since
``wannier_run`` isn't ported yet. It calls the raw ``_wannier90`` extension
directly (not ``wannier90.api``) purely to get reference ``nnlist``/
``nncell`` arrays to diff against.
"""
import os

import numpy as np

from wannier90._engine.kmesh import kmesh_get
from wannier90.io_helpers import reciprocal_lattice

from conftest import UPSTREAM_TESTDIR


def test_kmesh_get_matches_fortran_wannier_setup(gaas_case):
    from wannier90 import _wannier90

    recip_lattice = reciprocal_lattice(gaas_case.real_lattice)
    symbols_arr = np.array([s.encode() for s in gaas_case.symbols], dtype="S20")

    prev_cwd = os.getcwd()
    os.chdir(UPSTREAM_TESTDIR)
    try:
        ref_nntot, ref_nnlist, ref_nncell, *_ = _wannier90.wannier_setup(
            "gaas", gaas_case.mp_grid, gaas_case.real_lattice, recip_lattice, gaas_case.kpt_latt,
            gaas_case.num_bands_tot, symbols_arr, gaas_case.atoms_cart, False, False,
        )
    finally:
        os.chdir(prev_cwd)

    result = kmesh_get(gaas_case.kpt_latt, recip_lattice)

    assert result.nntot == ref_nntot
    np.testing.assert_array_equal(result.nnlist, ref_nnlist[:, :result.nntot])
    np.testing.assert_array_equal(result.nncell, ref_nncell[:, :, :result.nntot])

    # The B1 (finite-difference completeness) relation is kmesh_get's actual
    # defining correctness property -- checked internally already (kmesh_get
    # raises if it fails), re-asserted here so a future refactor that
    # silences that internal check still gets caught.
    acc = np.einsum("n,in,jn->ij", result.wb, result.bk[:, :, 0], result.bk[:, :, 0])
    np.testing.assert_allclose(acc, np.eye(3), atol=1e-6)

    # kmesh_shell_fixed (explicit shell_list): GaAs's automatic search only
    # ever needs shell 1, so pinning it explicitly should reproduce exactly
    # the same result as the automatic path (and the fortran reference).
    fixed = kmesh_get(gaas_case.kpt_latt, recip_lattice, num_shells=1, shell_list=[1])
    assert fixed.nntot == ref_nntot
    np.testing.assert_array_equal(fixed.nnlist, ref_nnlist[:, :fixed.nntot])
    np.testing.assert_array_equal(fixed.nncell, ref_nncell[:, :, :fixed.nntot])
