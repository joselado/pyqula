"""Port of abstractmatrixci.jl.

``AbstractMatrixCI`` is the shared interface for "matrix cross interpolation"
representations (``MatrixCI``, ``MatrixACA``, ``MatrixLUCI``). Concrete
subclasses must implement ``nrows``, ``ncols``, ``submatrix``, ``evaluate``,
``rank``, ``available_rows``, ``available_cols``.

Julia's 4-way multiple-dispatch ``getindex`` (matrix/vector/scalar cases,
distinguished by whether each argument is ``Int``, ``Colon``, or a vector)
collapses here into one ``__getitem__`` doing explicit isinstance checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _normalize_index(idx, n: int):
    """Return (array_of_indices, is_scalar, is_full_slice)."""
    if idx is None or (isinstance(idx, slice) and idx == slice(None)):
        return np.arange(n), False, True
    if isinstance(idx, (int, np.integer)):
        return np.array([idx]), True, False
    return np.asarray(idx), False, False


class AbstractMatrixCI(ABC):
    @abstractmethod
    def nrows(self) -> int: ...

    @abstractmethod
    def ncols(self) -> int: ...

    @abstractmethod
    def submatrix(self, rows, cols) -> np.ndarray: ...

    @abstractmethod
    def evaluate(self, i: int, j: int): ...

    @abstractmethod
    def rank(self) -> int: ...

    @abstractmethod
    def available_rows(self) -> list[int]: ...

    @abstractmethod
    def available_cols(self) -> list[int]: ...

    def size(self) -> tuple[int, int]:
        return self.nrows(), self.ncols()

    def row(self, i: int, cols=None) -> np.ndarray:
        return np.asarray(self.submatrix([i], self._cols_or_all(cols))).reshape(-1)

    def col(self, j: int, rows=None) -> np.ndarray:
        return np.asarray(self.submatrix(self._rows_or_all(rows), [j])).reshape(-1)

    def _rows_or_all(self, rows):
        return list(range(self.nrows())) if rows is None else rows

    def _cols_or_all(self, cols):
        return list(range(self.ncols())) if cols is None else cols

    def __getitem__(self, key):
        rows, cols = key
        rows_arr, rows_scalar, _ = _normalize_index(rows, self.nrows())
        cols_arr, cols_scalar, _ = _normalize_index(cols, self.ncols())
        if rows_scalar and cols_scalar:
            return self.evaluate(int(rows_arr[0]), int(cols_arr[0]))
        result = self.submatrix(rows_arr.tolist(), cols_arr.tolist())
        if rows_scalar:
            return np.asarray(result).reshape(-1)
        if cols_scalar:
            return np.asarray(result).reshape(-1)
        return result

    def local_error(self, a: np.ndarray, rowindices=None, colindices=None) -> np.ndarray:
        rows = self._rows_or_all(rowindices)
        cols = self._cols_or_all(colindices)
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)
        a_sub = a[np.ix_(rows_arr, cols_arr)]
        ci_sub = np.asarray(self.submatrix(rows_arr.tolist(), cols_arr.tolist()))
        return np.abs(a_sub - ci_sub)

    def find_new_pivot(self, a: np.ndarray, rowindices=None, colindices=None):
        rowindices = self.available_rows() if rowindices is None else rowindices
        colindices = self.available_cols() if colindices is None else colindices
        if self.rank() == min(a.shape):
            raise ValueError(
                "Cannot find a new pivot for this matrix CI, as it is already full rank."
            )
        if len(rowindices) == 0:
            raise ValueError(f"Cannot find a new pivot in an empty set of rows ({rowindices})")
        if len(colindices) == 0:
            raise ValueError(f"Cannot find a new pivot in an empty set of cols ({colindices})")

        localerrors = self.local_error(a, rowindices, colindices)
        ij_raw = np.unravel_index(np.argmax(localerrors), localerrors.shape)
        return (rowindices[ij_raw[0]], colindices[ij_raw[1]]), localerrors[ij_raw]
