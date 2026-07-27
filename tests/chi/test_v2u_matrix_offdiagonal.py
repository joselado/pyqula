import numpy as np

from pyqula.chitk.spinchi import V2U_matrix


def test_v2u_matrix_extracts_cross_orbital_terms():
    """V2U_matrix used to only read the DIAGONAL (i==i) up-down/down-up
    cross terms of a 2N spin-orbital matrix (U[i,i] = V[2i,2i+1] +
    V[2i+1,2i]), which is exactly right for an onsite-only interaction
    (where different orbitals never couple within the same (0,0,0) key)
    but silently drops any i!=j (cross-orbital) coupling -- exactly what a
    bond direction connecting two DIFFERENT sublattices of a multi-orbital
    unit cell needs (e.g. a bichain's nearest-neighbor SzSz exchange,
    which couples sublattice A to sublattice B, not A to A). This is a
    direct, deterministic check of the fixed (full N x N, not
    diagonal-only) extraction."""
    N = 2  # two orbitals (e.g. a bichain's A/B sublattices)
    V = np.zeros((2*N, 2*N), dtype=np.complex128)
    A, B, C = 1.0, 2.0, 3.0
    V[0, 1] = A; V[1, 0] = A  # orbital 0 <-> orbital 0, up-down cross term
    V[2, 3] = B; V[3, 2] = B  # orbital 1 <-> orbital 1, up-down cross term
    V[0, 3] = C; V[3, 0] = C  # orbital 0 <-> orbital 1, cross-orbital term

    U = V2U_matrix(V)

    expected = np.array([[2*A, C], [C, 2*B]], dtype=np.complex128)
    assert np.allclose(U, expected)


def test_v2u_matrix_reduces_to_diagonal_for_onsite_only_matrix():
    """When the input only ever has i==j (same-orbital) up-down/down-up
    entries populated (the onsite Hubbard U case), V2U_matrix must give
    exactly the old diagonal-only result -- the generalization must not
    change behavior for the case it already worked correctly for."""
    N = 3
    V = np.zeros((2*N, 2*N), dtype=np.complex128)
    diag_vals = [0.5, 1.5, -0.7]
    for i, u in enumerate(diag_vals):
        V[2*i, 2*i+1] = u/2.
        V[2*i+1, 2*i] = u/2.

    U = V2U_matrix(V)
    assert np.allclose(U, np.diag(diag_vals))
