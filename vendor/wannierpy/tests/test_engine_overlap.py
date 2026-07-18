"""Correctness tests for ``wannier90._engine.overlap.overlap_project``.

No Fortran reference is available for this specific routine in isolation
(the upstream test fixtures shipped in this repo all disentangle, which
takes a different code path -- ``dis_main``, phase 2 -- not
``overlap_project``). Validated instead against an independently-computed
closed-form Lowdin transformation (eigendecomposition of A^dagger A, rather
than the SVD ``overlap_project`` itself uses) and a direct check of the
M-matrix rotation formula.
"""
import numpy as np

from wannier90._engine.overlap import overlap_project


def test_overlap_project_matches_eigendecomposition_lowdin():
    rng = np.random.default_rng(0)
    num_wann, num_kpts, nntot = 3, 4, 2
    A = rng.normal(size=(num_wann, num_wann, num_kpts)) + 1j * rng.normal(size=(num_wann, num_wann, num_kpts))
    M = rng.normal(size=(num_wann, num_wann, nntot, num_kpts)) + \
        1j * rng.normal(size=(num_wann, num_wann, nntot, num_kpts))
    nnlist = np.array([[2, 3], [3, 4], [4, 1], [1, 2]])  # 1-indexed

    U, M_rotated = overlap_project(A, M, nnlist)

    for k in range(num_kpts):
        Ak = A[:, :, k]
        AhA = Ak.conj().T @ Ak
        w, v = np.linalg.eigh(AhA)
        inv_sqrt = v @ np.diag(w ** -0.5) @ v.conj().T
        U_ref = Ak @ inv_sqrt
        np.testing.assert_allclose(U[:, :, k], U_ref, atol=1e-8)

        unitarity = U[:, :, k].conj().T @ U[:, :, k]
        np.testing.assert_allclose(unitarity, np.eye(num_wann), atol=1e-10)

    for k in range(num_kpts):
        for nn in range(nntot):
            nkp2 = nnlist[k, nn] - 1
            expected = U[:, :, k].conj().T @ M[:, :, nn, k] @ U[:, :, nkp2]
            np.testing.assert_allclose(M_rotated[:, :, nn, k], expected)
