"""Pure-Python port of ``hamiltonian_wigner_seitz`` (src/hamiltonian.F90):
finds the lattice points of the real-space supercell (mp_grid unit cells)
that fall inside its Wigner-Seitz cell, with their degeneracies -- the
standard construction for Fourier-interpolating a k-space quantity to real
space on a regular mesh. Used here only for ``precond``'s real-space
gradient filter (``wannierise.py``); not the full ``hamiltonian.F90`` module
(Wannier-interpolated band structure, ``write_hr``, etc. -- unported).
"""
from __future__ import annotations

import numpy as np


def wigner_seitz_vectors(mp_grid: np.ndarray, real_lattice: np.ndarray, ws_search_size=(2, 2, 2),
                          ws_distance_tol: float = 1.0e-5):
    """Returns (irvec, ndegen, rpt_origin): ``irvec`` (nrpts, 3) integer
    lattice points (in the real_lattice basis), ``ndegen`` (nrpts,) their
    degeneracies, ``rpt_origin`` the index of R=0."""
    mp_grid = np.asarray(mp_grid)
    search = np.asarray(ws_search_size)
    real_metric = real_lattice @ real_lattice.T

    i_ranges = [np.arange(-search[k] - 1, search[k] + 2) for k in range(3)]
    I = np.stack(np.meshgrid(*i_ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    R = I * mp_grid[None, :]  # Born-von Karman supercell translations
    center_idx = int(np.nonzero((I == 0).all(axis=1))[0][0])

    n_ranges = [np.arange(-search[k] * mp_grid[k], search[k] * mp_grid[k] + 1) for k in range(3)]
    N = np.stack(np.meshgrid(*n_ranges, indexing="ij"), axis=-1).reshape(-1, 3)

    ndiff = N[:, None, :] - R[None, :, :]  # (num_n, dist_dim, 3)
    dist = np.einsum("pqi,ij,pqj->pq", ndiff, real_metric, ndiff)
    dist_min = dist.min(axis=1)
    tol2 = ws_distance_tol ** 2
    is_ws = np.abs(dist[:, center_idx] - dist_min) < tol2
    ndegen_all = np.sum(np.abs(dist - dist_min[:, None]) < tol2, axis=1)

    irvec = N[is_ws]
    ndegen = ndegen_all[is_ws]
    rpt_origin = int(np.nonzero((irvec == 0).all(axis=1))[0][0])

    tot = float(np.sum(1.0 / ndegen))
    expected = float(np.prod(mp_grid))
    if abs(tot - expected) > 1.0e-8:
        raise ValueError(
            f"wigner_seitz_vectors: sum(1/ndegen)={tot} != prod(mp_grid)={expected} "
            "(failed to find a consistent set of Wigner-Seitz points)"
        )

    return irvec, ndegen, rpt_origin
