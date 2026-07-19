"""Port of matrixlu.jl: rank-revealing LU (rrLU) with full or rook pivoting.

The most important file in the matrix layer. 0-based indexing throughout.

Manual rank-1 (`ger`) updates done via explicit loops in Julia become
vectorized ``np.outer`` updates here (faster in numpy, no reason to keep a
manual loop) -- see CLAUDE.md.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.matrix._numba_kernels import HAVE_NUMBA, _rrlu_pivot_kernel
from pyqula.qutecipytk.util import pushrandomsubset

_NUMBA_DTYPES = (np.dtype(np.float64), np.dtype(np.complex128))


def submatrixargmax(A: np.ndarray, startindex: int, f: Callable = np.abs):
    """Location of the max of f(A) within the trailing submatrix A[startindex:, startindex:],
    in original (unshifted) coordinates.

    Julia's submatrixargmax scans columns-outer/rows-inner, so on an exact tie it
    favors the smallest column index. np.argmax on a plain (row-major) array would
    instead favor the smallest row index -- transpose first so np.argmax's
    first-occurrence tie-break follows the same column-outer/row-inner order.
    """
    sub = A[startindex:, startindex:]
    vals = f(sub)
    idx = np.unravel_index(np.argmax(vals.T), vals.T.shape)
    return startindex + idx[1], startindex + idx[0]


class rrLU:
    def __init__(self, rowpermutation, colpermutation, L, U, leftorthogonal, npivot, error):
        self.rowpermutation = np.asarray(rowpermutation, dtype=np.int64)
        self.colpermutation = np.asarray(colpermutation, dtype=np.int64)
        self.L = np.asarray(L)
        self.U = np.asarray(U)
        self.leftorthogonal = bool(leftorthogonal)
        self.npivot = int(npivot)
        self.error = float(error)

        if self.npivot != self.L.shape[1]:
            raise ValueError("L must have the same number of columns as the number of pivots.")
        if self.npivot != self.U.shape[0]:
            raise ValueError("U must have the same number of rows as the number of pivots.")
        if len(self.rowpermutation) != self.L.shape[0]:
            raise ValueError("rowpermutation must have length equal to the number of rows of L.")
        if len(self.colpermutation) != self.U.shape[1]:
            raise ValueError("colpermutation must have length equal to the number of columns of U.")

    @classmethod
    def empty(cls, dtype, nrows: int, ncols: int, leftorthogonal: bool = True) -> "rrLU":
        return cls(
            np.arange(nrows), np.arange(ncols),
            np.zeros((nrows, 0), dtype=dtype), np.zeros((0, ncols), dtype=dtype),
            leftorthogonal, 0, np.nan,
        )

    @classmethod
    def from_matrix(cls, A: np.ndarray, leftorthogonal: bool = True) -> "rrLU":
        return cls.empty(A.dtype, A.shape[0], A.shape[1], leftorthogonal)

    def _swap_row(self, A: np.ndarray, a: int, b: int) -> None:
        self.rowpermutation[a], self.rowpermutation[b] = self.rowpermutation[b], self.rowpermutation[a]
        A[[a, b], :] = A[[b, a], :]

    def _swap_col(self, A: np.ndarray, a: int, b: int) -> None:
        self.colpermutation[a], self.colpermutation[b] = self.colpermutation[b], self.colpermutation[a]
        A[:, [a, b]] = A[:, [b, a]]

    def _add_pivot(self, A: np.ndarray, newpivot: tuple[int, int]) -> None:
        k = self.npivot
        self.npivot += 1
        self._swap_row(A, k, newpivot[0])
        self._swap_col(A, k, newpivot[1])

        if self.leftorthogonal:
            A[k + 1:, k] /= A[k, k]
        else:
            A[k, k + 1:] /= A[k, k]

        x = A[k + 1:, k]
        y = A[k, k + 1:]
        A[k + 1:, k + 1:] -= np.outer(x, y)

    def _optimize(
        self, A: np.ndarray, maxrank: int | None = None, reltol: float = 1e-14, abstol: float = 0.0
    ) -> None:
        maxrank = min(maxrank if maxrank is not None else min(A.shape), A.shape[0], A.shape[1])

        # numba path: same algorithm (identical argmax tie-breaking, identical stopping
        # rule), just compiled -- see pyqula/qutecipytk/matrix/_numba_kernels.py. Falls back to the
        # pure-Python loop below for dtypes numba can't safely handle here (e.g. integer,
        # which numpy's own in-place division would already reject) or if numba isn't
        # installed at all.
        if HAVE_NUMBA and A.dtype in _NUMBA_DTYPES and A.flags["C_CONTIGUOUS"]:
            self.npivot, self.error, _ = _rrlu_pivot_kernel(
                A, self.rowpermutation, self.colpermutation, self.npivot, maxrank, reltol, abstol,
                self.leftorthogonal, self.error,
            )
        else:
            maxerror = 0.0
            while self.npivot < maxrank:
                k = self.npivot
                # argmax(|x|^2) == argmax(|x|) (monotonic), and self.error re-reads the raw
                # value separately below -- so this avoids a squaring pass over the trailing
                # submatrix every pivot step without changing which pivot gets picked.
                newpivot = submatrixargmax(A, k)
                self.error = abs(A[newpivot[0], newpivot[1]])
                # Add at least 1 pivot to get a well-defined L * U.
                if (abs(self.error) < reltol * maxerror or abs(self.error) < abstol) and self.npivot > 0:
                    break
                maxerror = max(maxerror, self.error)
                self._add_pivot(A, newpivot)

        self.L = np.tril(A[:, : self.npivot])
        self.U = np.triu(A[: self.npivot, :])
        if np.any(np.isnan(self.L)):
            raise FloatingPointError("lu.L contains NaNs")
        if np.any(np.isnan(self.U)):
            raise FloatingPointError("lu.U contains NaNs")

        if self.leftorthogonal:
            np.fill_diagonal(self.L, 1.0)
        else:
            np.fill_diagonal(self.U, 1.0)

        if self.npivot >= min(A.shape):
            self.error = 0.0

    def size(self) -> tuple[int, int]:
        return self.L.shape[0], self.U.shape[1]

    def left(self, permute: bool = True) -> np.ndarray:
        if not permute:
            return self.L
        l = np.empty_like(self.L)
        l[self.rowpermutation, :] = self.L
        return l

    def right(self, permute: bool = True) -> np.ndarray:
        if not permute:
            return self.U
        u = np.empty_like(self.U)
        u[:, self.colpermutation] = self.U
        return u

    def diag(self) -> np.ndarray:
        if self.leftorthogonal:
            return np.diag(self.U[: self.npivot, : self.npivot])
        return np.diag(self.L[: self.npivot, : self.npivot])

    def rowindices(self) -> list[int]:
        return self.rowpermutation[: self.npivot].tolist()

    def colindices(self) -> list[int]:
        return self.colpermutation[: self.npivot].tolist()

    def npivots(self) -> int:
        return self.npivot

    def pivoterrors(self) -> np.ndarray:
        return np.concatenate([np.abs(self.diag()), [self.error]])

    def lastpivoterror(self) -> float:
        return self.error

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b, given A == this (square, full-rank) rrLU."""
        if self.size()[0] != self.size()[1]:
            raise ValueError("Matrix must be square.")
        if self.npivot != self.size()[0]:
            raise ValueError("rank-deficient matrix is not supported")
        b_perm = b[self.rowpermutation, :]
        x_perm = _lu_solve(self.L, self.U, b_perm)
        x = np.empty_like(x_perm)
        x[self.colpermutation, :] = x_perm
        return x

    def transpose(self) -> "rrLU":
        return rrLU(
            self.colpermutation.copy(), self.rowpermutation.copy(),
            self.U.T.copy(), self.L.T.copy(),
            not self.leftorthogonal, self.npivot, self.error,
        )


def _lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (LU) x = b for L lower-triangular, U upper-triangular. Not
    performance-optimized (mirrors the reference)."""
    n2 = L.shape[1]
    n3 = U.shape[1]
    M = b.shape[1]

    y = np.zeros((n2, M), dtype=b.dtype)
    for i in range(n2):
        y[i, :] = b[i, :] - L[i, :i] @ y[:i, :]
        y[i, :] /= L[i, i]

    x = np.zeros((n3, M), dtype=b.dtype)
    for i in range(n3 - 1, -1, -1):
        x[i, :] = y[i, :] - U[i, i + 1:] @ x[i + 1:, :]
        x[i, :] /= U[i, i]

    return x


def rrlu_inplace(
    A: np.ndarray, maxrank: int | None = None, reltol: float = 1e-14, abstol: float = 0.0,
    leftorthogonal: bool = True,
) -> rrLU:
    """Rank-revealing LU decomposition, mutating A in place."""
    lu = rrLU.from_matrix(A, leftorthogonal=leftorthogonal)
    lu._optimize(A, maxrank=maxrank, reltol=reltol, abstol=abstol)
    return lu


def rrlu(
    A: np.ndarray, maxrank: int | None = None, reltol: float = 1e-14, abstol: float = 0.0,
    leftorthogonal: bool = True,
) -> rrLU:
    """Rank-revealing LU decomposition (non-mutating)."""
    return rrlu_inplace(A.copy(), maxrank=maxrank, reltol=reltol, abstol=abstol, leftorthogonal=leftorthogonal)


def cols2Lmatrix(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """C <- C @ P^-1 in place, for P a small square (upper) triangular pivot block."""
    if C.shape[1] != P.shape[1]:
        raise ValueError("C and P matrices must have the same number of columns.")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P matrix must be square.")
    for k in range(P.shape[0]):
        C[:, k] /= P[k, k]
        x = C[:, k]
        y = P[k, k + 1:]
        C[:, k + 1:] -= np.outer(x, y)
    return C


def rows2Umatrix(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """R <- P^-1 @ R in place, for P a small square (lower) triangular pivot block."""
    if R.shape[0] != P.shape[0]:
        raise ValueError("R and P matrices must have the same number of rows.")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P matrix must be square.")
    for k in range(P.shape[0]):
        R[k, :] /= P[k, k]
        x = P[k + 1:, k]
        y = R[k, :]
        R[k + 1:, :] -= np.outer(x, y)
    return R


def _default_batchf(f: Callable, dtype) -> Callable:
    def batchf(rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
        rows = list(rows)
        cols = list(cols)
        return np.array([[f(r, c) for c in cols] for r in rows], dtype=dtype).reshape(len(rows), len(cols))
    return batchf


def arrlu(
    dtype,
    f: Callable,
    matrixsize: tuple[int, int],
    I0: Sequence[int] | None = None,
    J0: Sequence[int] | None = None,
    maxrank: int | None = None,
    reltol: float = 1e-14,
    abstol: float = 0.0,
    leftorthogonal: bool = True,
    numrookiter: int = 5,
    usebatcheval: bool = False,
) -> rrLU:
    """Adaptive rook-pivoted rrLU: evaluates f lazily on submatrices, never
    materializing the full matrix. f(row, col) -> scalar (or, if
    usebatcheval, f(rows, cols) -> matrix directly)."""
    I0 = list(I0) if I0 else []
    J0 = list(J0) if J0 else []
    lu = rrLU.empty(dtype, matrixsize[0], matrixsize[1], leftorthogonal)
    islowrank = False
    maxrank = min(maxrank if maxrank is not None else min(matrixsize), *matrixsize)

    batchf = f if usebatcheval else _default_batchf(f, dtype)

    while True:
        if leftorthogonal:
            pushrandomsubset(J0, range(matrixsize[1]), max(1, len(J0)))
        else:
            pushrandomsubset(I0, range(matrixsize[0]), max(1, len(I0)))

        for rookiter in range(1, numrookiter + 1):
            colmove = (rookiter % 2 == 0) == leftorthogonal
            if colmove:
                submatrix = batchf(I0, lu.colpermutation.tolist())
            else:
                submatrix = batchf(lu.rowpermutation.tolist(), J0)
            lu.npivot = 0
            lu._optimize(submatrix, maxrank=maxrank, reltol=reltol, abstol=abstol)
            islowrank = islowrank or (lu.npivots() < min(submatrix.shape))
            if lu.rowindices() == I0 and lu.colindices() == J0:
                break

            J0 = lu.colindices()
            I0 = lu.rowindices()

        if islowrank or len(I0) >= maxrank:
            break

    if lu.L.shape[0] < matrixsize[0]:
        I2 = [i for i in range(matrixsize[0]) if i not in set(I0)]
        lu.rowpermutation = np.array(I0 + I2, dtype=np.int64)
        L2 = batchf(I2, J0)
        cols2Lmatrix(L2, lu.U[: lu.npivot, : lu.npivot])
        lu.L = np.vstack([lu.L[: lu.npivot, : lu.npivot], L2])

    if lu.U.shape[1] < matrixsize[1]:
        J2 = [j for j in range(matrixsize[1]) if j not in set(J0)]
        lu.colpermutation = np.array(J0 + J2, dtype=np.int64)
        U2 = batchf(I0, J2)
        rows2Umatrix(U2, lu.L[: lu.npivot, : lu.npivot])
        lu.U = np.hstack([lu.U[: lu.npivot, : lu.npivot], U2])

    return lu


def rrlu_from_function(
    dtype,
    f: Callable,
    matrixsize: tuple[int, int],
    I0: Sequence[int] | None = None,
    J0: Sequence[int] | None = None,
    pivotsearch: str = "full",
    **kwargs,
) -> rrLU:
    if pivotsearch == "rook":
        return arrlu(dtype, f, matrixsize, I0, J0, **kwargs)
    if pivotsearch == "full":
        A = np.array(
            [[f(i, j) for j in range(matrixsize[1])] for i in range(matrixsize[0])], dtype=dtype
        )
        return rrlu_inplace(A, **kwargs)
    raise ValueError(f"Unknown pivot search strategy {pivotsearch!r}. Choose 'rook' or 'full'.")
