"""Port of matrixluci.jl: MatrixLUCI, a thin AbstractMatrixCI-style wrapper
around rrLU reconstructing CI-style left/right factors via triangular solves.

Note: like the Julia original, this does *not* implement the full
AbstractMatrixCI contract (submatrix/evaluate) -- it's a read-only
view/reconstruction layer over rrLU exposing only the accessors TCI2 needs.

Gotcha (see CLAUDE.md): Julia's `result[perm, :] = result` self-referential
scatter is NOT translated literally -- that risks aliasing/incorrect results
under numpy fancy-index assignment. Ported as `result[argsort(perm)]`
instead, which is the correct un-permute operation.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from pyqula.qutecipytk.matrix.rrlu import rrLU, rrlu, rrlu_from_function


class MatrixLUCI:
    def __init__(self, lu: rrLU):
        self.lu = lu

    @classmethod
    def from_matrix(cls, A: np.ndarray, **kwargs) -> "MatrixLUCI":
        return cls(rrlu(A, **kwargs))

    @classmethod
    def from_function(cls, dtype, f, matrixsize, I0=None, J0=None, **kwargs) -> "MatrixLUCI":
        return cls(rrlu_from_function(dtype, f, matrixsize, I0, J0, **kwargs))

    def size(self) -> tuple[int, int]:
        return self.lu.size()

    def npivots(self) -> int:
        return self.lu.npivots()

    def rowindices(self) -> list[int]:
        return self.lu.rowindices()

    def colindices(self) -> list[int]:
        return self.lu.colindices()

    def colmatrix(self) -> np.ndarray:
        n = self.npivots()
        return self.lu.left() @ self.lu.right(permute=False)[:, :n]

    def rowmatrix(self) -> np.ndarray:
        n = self.npivots()
        return self.lu.left(permute=False)[:n, :] @ self.lu.right()

    def colstimespivotinv(self) -> np.ndarray:
        """A(:,J) @ A(I,J)^-1, without ever forming A(I,J)^-1 explicitly.

        Note: this reads L's raw (possibly non-unit) diagonal regardless of
        `leftorthogonal` -- only correct to assume a unit diagonal when this
        rrLU was actually built with leftorthogonal=True.
        """
        n = self.npivots()
        m = self.size()[0]
        result = np.eye(m, n, dtype=self.lu.L.dtype)
        if n < m:
            L = self.lu.left(permute=False)
            B = L[:n, :]
            rhs = L[n:, :]
            x = scipy.linalg.solve_triangular(
                B.T, rhs.T, lower=False, unit_diagonal=self.lu.leftorthogonal
            ).T
            result[n:, :] = x
        return result[np.argsort(self.lu.rowpermutation)]

    def pivotinvtimesrows(self) -> np.ndarray:
        """A(I,J)^-1 @ A(I,:), without ever forming A(I,J)^-1 explicitly.

        Note: this reads U's raw (possibly non-unit) diagonal regardless of
        `leftorthogonal` -- only correct to assume a unit diagonal when this
        rrLU was actually built with leftorthogonal=False.
        """
        n = self.npivots()
        ncols = self.size()[1]
        result = np.eye(n, ncols, dtype=self.lu.U.dtype)
        if n < ncols:
            U = self.lu.right(permute=False)
            B = U[:, :n]
            rhs = U[:, n:]
            x = scipy.linalg.solve_triangular(
                B, rhs, lower=False, unit_diagonal=not self.lu.leftorthogonal
            )
            result[:, n:] = x
        return result[:, np.argsort(self.lu.colpermutation)]

    def left(self) -> np.ndarray:
        return self.colstimespivotinv() if self.lu.leftorthogonal else self.colmatrix()

    def right(self) -> np.ndarray:
        return self.rowmatrix() if self.lu.leftorthogonal else self.pivotinvtimesrows()

    def pivoterrors(self) -> np.ndarray:
        return self.lu.pivoterrors()

    def lastpivoterror(self) -> float:
        return self.lu.lastpivoterror()
