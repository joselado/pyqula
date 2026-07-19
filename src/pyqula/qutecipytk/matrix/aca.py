"""Port of matrixci.jl (MatrixCI) and matrixaca.jl (MatrixACA).

0-based indexing throughout (Julia is 1-based). ``firstpivot``/pivot indices
and row/col index lists are plain Python ints/lists in ``[0, n)``.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from pyqula.qutecipytk.matrix.base import AbstractMatrixCI


def a_times_binv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A @ B^-1 for rectangular A, square B, via QR (stable if B^-1 ill-conditioned)."""
    m = a.shape[0]
    ab = np.vstack([a, b])
    q, _ = scipy.linalg.qr(ab, mode="economic")
    qa = q[:m, :]
    qb = q[m:, :]
    return np.linalg.solve(qb.T, qa.T).T


def a_inv_times_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A^-1 @ B, via the same QR trick applied to the transposed problem."""
    return a_times_binv(b.T, a.T).T


class MatrixCI(AbstractMatrixCI):
    """Classic matrix cross interpolation with lazy QR-based pivot inversion."""

    def __init__(self, rowindices, colindices, pivotcols: np.ndarray, pivotrows: np.ndarray):
        self.rowindices: list[int] = list(rowindices)
        self.colindices: list[int] = list(colindices)
        self.pivotcols = np.asarray(pivotcols)
        self.pivotrows = np.asarray(pivotrows)

    @classmethod
    def empty(cls, dtype, nrows: int, ncols: int) -> "MatrixCI":
        return cls([], [], np.zeros((nrows, 0), dtype=dtype), np.zeros((0, ncols), dtype=dtype))

    @classmethod
    def from_matrix(cls, a: np.ndarray, firstpivot: tuple[int, int]) -> "MatrixCI":
        i, j = firstpivot
        return cls([i], [j], a[:, [j]], a[[i], :])

    def Iset(self) -> list[int]:
        return self.rowindices

    def Jset(self) -> list[int]:
        return self.colindices

    def nrows(self) -> int:
        return self.pivotcols.shape[0]

    def ncols(self) -> int:
        return self.pivotrows.shape[1]

    def pivotmatrix(self) -> np.ndarray:
        return self.pivotcols[self.rowindices, :]

    def leftmatrix(self) -> np.ndarray:
        return a_times_binv(self.pivotcols, self.pivotmatrix())

    def rightmatrix(self) -> np.ndarray:
        return a_inv_times_b(self.pivotmatrix(), self.pivotrows)

    def available_rows(self) -> list[int]:
        used = set(self.rowindices)
        return [i for i in range(self.nrows()) if i not in used]

    def available_cols(self) -> list[int]:
        used = set(self.colindices)
        return [j for j in range(self.ncols()) if j not in used]

    def rank(self) -> int:
        return len(self.rowindices)

    def is_empty(self) -> bool:
        return len(self.colindices) == 0

    def firstpivotvalue(self):
        return 1.0 if self.is_empty() else self.pivotcols[self.rowindices[0], 0]

    def evaluate(self, i: int, j: int):
        if self.is_empty():
            return self.pivotcols.dtype.type(0)
        return np.dot(self.leftmatrix()[i, :], self.pivotrows[:, j])

    def submatrix(self, rows, cols) -> np.ndarray:
        n_rows = self.nrows() if rows is None else len(rows)
        n_cols = self.ncols() if cols is None else len(cols)
        if self.is_empty():
            return np.zeros((n_rows, n_cols), dtype=self.pivotcols.dtype)
        rows_idx = list(range(self.nrows())) if rows is None else rows
        cols_idx = list(range(self.ncols())) if cols is None else cols
        return self.leftmatrix()[rows_idx, :] @ self.pivotrows[:, cols_idx]

    def to_matrix(self) -> np.ndarray:
        return self.leftmatrix() @ self.pivotrows

    def add_pivot_row(self, a: np.ndarray, rowindex: int) -> None:
        if a.shape != self.size():
            raise ValueError(f"Matrix shape mismatch: {a.shape} != {self.size()}")
        if not (0 <= rowindex < self.nrows()):
            raise IndexError(f"Row index {rowindex} out of bounds")
        if rowindex in self.rowindices:
            raise ValueError(f"Row {rowindex} already has a pivot.")
        self.pivotrows = np.vstack([self.pivotrows, a[rowindex, :]])
        self.rowindices.append(rowindex)

    def add_pivot_col(self, a: np.ndarray, colindex: int) -> None:
        if a.shape != self.size():
            raise ValueError(f"Matrix shape mismatch: {a.shape} != {self.size()}")
        if not (0 <= colindex < self.ncols()):
            raise IndexError(f"Col index {colindex} out of bounds")
        if colindex in self.colindices:
            raise ValueError(f"Column {colindex} already has a pivot.")
        self.pivotcols = np.hstack([self.pivotcols, a[:, [colindex]]])
        self.colindices.append(colindex)

    def add_pivot(self, a: np.ndarray, pivotindices: tuple[int, int] | None = None) -> None:
        if pivotindices is None:
            pivotindices = self.find_new_pivot(a)[0]
        i, j = pivotindices
        if a.shape != self.size():
            raise ValueError(f"Matrix shape mismatch: {a.shape} != {self.size()}")
        if not (0 <= i < self.nrows()) or not (0 <= j < self.ncols()):
            raise IndexError(
                f"Cannot add a pivot at indices ({i}, {j}): out of bounds for a "
                f"{self.nrows()} x {self.ncols()} matrix."
            )
        if i in self.rowindices:
            raise ValueError(f"Row {i} already has a pivot.")
        if j in self.colindices:
            raise ValueError(f"Column {j} already has a pivot.")
        self.add_pivot_row(a, i)
        self.add_pivot_col(a, j)


def crossinterpolate_matrix(
    a: np.ndarray, tolerance: float = 1e-6, maxiter: int = 200, firstpivot=None
) -> MatrixCI:
    """Rank-revealing matrix CI/ACA driver (greedy global-argmax pivoting)."""
    if firstpivot is None:
        firstpivot = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    ci = MatrixCI.from_matrix(a, firstpivot)
    for _ in range(maxiter):
        localerrors = ci.local_error(a)
        newpivot_raw = np.unravel_index(np.argmax(localerrors), localerrors.shape)
        pivoterror = localerrors[newpivot_raw]
        if pivoterror < tolerance:
            return ci
        ci.add_pivot(a, newpivot_raw)
    return ci


class MatrixACA(AbstractMatrixCI):
    """Incremental Adaptive Cross Approximation (Kumar 2016): rank-1 deflation,
    no explicit pivot-matrix inversion."""

    def __init__(self, rowindices, colindices, u: np.ndarray, v: np.ndarray, alpha):
        self.rowindices: list[int] = list(rowindices)
        self.colindices: list[int] = list(colindices)
        self.u = np.asarray(u)
        self.v = np.asarray(v)
        self.alpha: list = list(alpha)

    @classmethod
    def empty(cls, dtype, nrows: int, ncols: int) -> "MatrixACA":
        return cls([], [], np.zeros((nrows, 0), dtype=dtype), np.zeros((0, ncols), dtype=dtype), [])

    @classmethod
    def from_matrix(cls, a: np.ndarray, firstpivot: tuple[int, int]) -> "MatrixACA":
        i, j = firstpivot
        return cls([i], [j], a[:, [j]], a[[i], :], [1.0 / a[i, j]])

    def nrows(self) -> int:
        return self.u.shape[0]

    def ncols(self) -> int:
        return self.v.shape[1]

    def npivots(self) -> int:
        return self.u.shape[1]

    def rank(self) -> int:
        return len(self.rowindices)

    def is_empty(self) -> bool:
        return len(self.colindices) == 0

    def available_rows(self) -> list[int]:
        used = set(self.rowindices)
        return [i for i in range(self.nrows()) if i not in used]

    def available_cols(self) -> list[int]:
        used = set(self.colindices)
        return [j for j in range(self.ncols()) if j not in used]

    def _uk(self, a: np.ndarray) -> np.ndarray:
        """Residual column u_k(x): the new column deflated against previous pivots."""
        k = len(self.colindices)
        yk = self.colindices[-1]
        result = a[:, yk].copy()
        for l in range(k - 1):
            xl = self.rowindices[l]
            result = result - (self.v[l, yk] / self.u[xl, l]) * self.u[:, l]
        return result

    def add_pivot_col(self, a: np.ndarray, yk: int) -> None:
        self.colindices.append(yk)
        self.u = np.hstack([self.u, self._uk(a).reshape(-1, 1)])

    def _vk(self, a: np.ndarray) -> np.ndarray:
        """Residual row v_k(y): the new row deflated against previous pivots."""
        k = len(self.rowindices)
        xk = self.rowindices[-1]
        result = a[xk, :].copy()
        for l in range(k - 1):
            xl = self.rowindices[l]
            result = result - (self.u[xk, l] / self.u[xl, l]) * self.v[l, :]
        return result

    def add_pivot_row(self, a: np.ndarray, xk: int) -> None:
        self.rowindices.append(xk)
        self.v = np.vstack([self.v, self._vk(a).reshape(1, -1)])
        self.alpha.append(1.0 / self.u[xk, -1])

    def add_pivot(self, a: np.ndarray, pivotindices: tuple[int, int] | None = None) -> None:
        if pivotindices is not None:
            self.add_pivot_col(a, pivotindices[1])
            self.add_pivot_row(a, pivotindices[0])
            return
        # ACA local-pivoting heuristic: argmax of the *last* residual row/col only.
        availcols = self.available_cols()
        yk = availcols[int(np.argmax(np.abs(self.v[-1, availcols])))]
        self.add_pivot_col(a, yk)

        availrows = self.available_rows()
        xk = availrows[int(np.argmax(np.abs(self.u[availrows, -1])))]
        self.add_pivot_row(a, xk)

    def submatrix(self, rows, cols) -> np.ndarray:
        n_rows = self.nrows() if rows is None else len(rows)
        n_cols = self.ncols() if cols is None else len(cols)
        if self.is_empty():
            return np.zeros((n_rows, n_cols), dtype=self.u.dtype)
        r = self.rank()
        rows_idx = list(range(self.nrows())) if rows is None else rows
        cols_idx = list(range(self.ncols())) if cols is None else cols
        alpha_r = np.asarray(self.alpha[:r])
        return self.u[np.ix_(rows_idx, range(r))] @ (alpha_r[:, None] * self.v[:r][:, cols_idx])

    def to_matrix(self) -> np.ndarray:
        return self.submatrix(None, None)

    def evaluate(self, i: int, j: int):
        return np.sum(self.u[i, :] * np.asarray(self.alpha) * self.v[:, j])

    def set_cols(self, newpivotrows: np.ndarray, permutation: np.ndarray) -> None:
        """Re-base columns to a new/reordered universe (permutation[old] = new)."""
        permutation = np.asarray(permutation)
        self.colindices = [int(permutation[c]) for c in self.colindices]

        tempv = np.empty_like(newpivotrows)
        tempv[:, permutation] = self.v
        self.v = tempv

        all_cols = set(range(newpivotrows.shape[1]))
        newindices = sorted(all_cols - set(permutation.tolist()))
        for k in range(newpivotrows.shape[0]):
            self.v[k, newindices] = newpivotrows[k, newindices]
            for l in range(k):
                self.v[k, newindices] = self.v[k, newindices] - self.v[l, newindices] * (
                    self.u[self.rowindices[k], l] * self.alpha[l]
                )

    def set_rows(self, newpivotcols: np.ndarray, permutation: np.ndarray) -> None:
        """Re-base rows to a new/reordered universe (permutation[old] = new)."""
        permutation = np.asarray(permutation)
        self.rowindices = [int(permutation[r]) for r in self.rowindices]

        tempu = np.empty_like(newpivotcols)
        tempu[permutation, :] = self.u
        self.u = tempu

        all_rows = set(range(newpivotcols.shape[0]))
        newindices = sorted(all_rows - set(permutation.tolist()))
        for k in range(newpivotcols.shape[1]):
            self.u[newindices, k] = newpivotcols[newindices, k]
            for l in range(k):
                self.u[newindices, k] = self.u[newindices, k] - self.u[newindices, l] * (
                    self.v[l, self.colindices[k]] * self.alpha[l]
                )
