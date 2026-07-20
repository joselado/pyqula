"""Pure-Python port of ``overlap_project`` (src/overlap.F90): builds the
initial guess for the Wannierisation unitary matrices from the projection
overlaps ``A_matrix`` via a Lowdin transformation (Marzari-Vanderbilt-style,
see Sec. 3 of the wannier90 CPC 2008 paper), and rotates ``M_matrix`` into
that gauge.

Only used on the *no-disentanglement* path (``num_bands == num_wann``) --
when disentangling, the equivalent initial guess comes out of ``dis_main``
(``dis_project``/``dis_extract``, phase 2) instead. ``overlap_project_gamma``
(the real-valued gamma-point variant) is not yet ported.

``sym`` (a ``sitesym.SymmetryData``, ``lsitesymmetry``) is supported here
too: ``wann_main`` requires its *initial* U/M already be symmetry-consistent
across each star (not just each iteration's update), so this seed needs the
same ``symmetrize_u_matrix`` treatment as ``disentangle.py``'s
``internal_find_u`` (src/overlap.F90's own call to
``sitesym_symmetrize_u_matrix`` right after building ``u_matrix``).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import sitesym

EPS5 = 1.0e-5


def overlap_project(A_matrix: np.ndarray, M_matrix: np.ndarray, nnlist: np.ndarray, sym=None):
    """
    Parameters
    ----------
    A_matrix : (num_wann, num_wann, num_kpts) complex
        Projection overlaps (square: this path requires num_bands == num_wann).
    M_matrix : (num_wann, num_wann, nntot, num_kpts) complex
        Overlap matrices ``M_mn(k, b)``.
    nnlist : (num_kpts, nntot) int, 1-indexed
        k-point neighbour table (as returned by ``kmesh_get``).
    sym : sitesym.SymmetryData, optional
        Enables ``lsitesymmetry``.

    Returns
    -------
    U_matrix : (num_wann, num_wann, num_kpts) complex, unitary at each k-point.
    M_matrix_rotated : (num_wann, num_wann, nntot, num_kpts) complex.
    """
    num_bands, num_wann, num_kpts = A_matrix.shape
    if num_bands != num_wann:
        raise ValueError("overlap_project requires num_bands == num_wann (no disentanglement)")

    # Lowdin orthogonalization: A = Z @ diag(s) @ Vh  =>  U = Z @ Vh
    # (the SVD-based equivalent of U = A (A^dagger A)^{-1/2}).
    Z, _, Vh = np.linalg.svd(np.moveaxis(A_matrix, -1, 0))  # batched over k-points
    U_matrix = np.moveaxis(Z @ Vh, 0, -1)

    if sym is not None:
        sym_wann = replace(sym, d_matrix_band=sym.d_matrix_wann.copy())
        U_matrix = sitesym.symmetrize_u_matrix(U_matrix, sym_wann, lwindow=None)

    unitarity = np.einsum("mik,mjk->ijk", U_matrix.conj(), U_matrix)
    if not np.allclose(unitarity, np.eye(num_wann)[:, :, None], atol=EPS5):
        raise ValueError("overlap_project: initial U_matrix is not unitary")

    nntot = M_matrix.shape[2]
    M_rotated = np.empty_like(M_matrix)
    for nkp in range(num_kpts):
        for nn in range(nntot):
            nkp2 = int(nnlist[nkp, nn]) - 1
            M_rotated[:, :, nn, nkp] = (
                U_matrix[:, :, nkp].conj().T @ M_matrix[:, :, nn, nkp] @ U_matrix[:, :, nkp2]
            )

    return U_matrix, M_rotated
