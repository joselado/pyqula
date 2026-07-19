"""Port of cachedtensortrain.jl: TTCache, memoized left/right environment
evaluation and vectorized batch evaluation.

The single most performance-critical routine here is ``batchevaluate``: it
builds stacked left/right environment matrices for *all* candidate left/
right index sets at once and contracts the "center" sites via a handful of
big matmuls instead of one contraction per candidate pivot pair -- this is
what makes TCI2's rook/full pivot search tractable.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain
from pyqula.qutecipytk.tensortrain.batcheval import BatchEvaluator


class TTCache(AbstractTensorTrain, BatchEvaluator):
    def __init__(self, sitetensors: Sequence[np.ndarray], sitedims: Sequence[Sequence[int]] | None = None):
        sitetensors = [np.asarray(T) for T in sitetensors]
        if sitedims is None:
            sitedims = [list(T.shape[1:-1]) for T in sitetensors]
        sitedims = [list(d) for d in sitedims]
        if len(sitetensors) != len(sitedims):
            raise ValueError("The number of site tensors and site dimensions must be the same.")
        for n in range(len(sitetensors)):
            expected = int(np.prod(sitedims[n])) if sitedims[n] else 1
            actual = int(np.prod(sitetensors[n].shape[1:-1])) if sitetensors[n].ndim > 2 else 1
            if expected != actual:
                raise ValueError(f"Site dimensions do not match the site tensor dimensions at {n}.")

        self._fused = [T.reshape(T.shape[0], -1, T.shape[-1]) for T in sitetensors]
        self._sitedims = sitedims
        self._cacheleft: list[dict] = [dict() for _ in range(len(sitetensors) + 1)]
        self._cacheright: list[dict] = [dict() for _ in range(len(sitetensors) + 1)]

    @classmethod
    def from_tt(cls, tt: AbstractTensorTrain, sitedims: Sequence[Sequence[int]] | None = None) -> "TTCache":
        return cls(list(tt.sitetensors()), sitedims)

    def __len__(self) -> int:
        return len(self._fused)

    @property
    def dtype(self):
        return self._fused[0].dtype

    def sitedims(self) -> list[list[int]]:
        return self._sitedims

    def sitetensors(self) -> list[np.ndarray]:
        return [self.sitetensor(n) for n in range(len(self))]

    def sitetensor(self, i: int) -> np.ndarray:
        t = self._fused[i]
        return t.reshape(t.shape[0], *self._sitedims[i], t.shape[-1])

    def clear_cache(self) -> None:
        for d in self._cacheleft:
            d.clear()
        for d in self._cacheright:
            d.clear()

    def evaluate_left(self, indexset: Sequence[int]) -> np.ndarray:
        n = len(indexset)
        if n > len(self):
            raise ValueError("For evaluate_left, number of indices must be <= the number of TTCache legs.")
        if n == 0:
            return np.array([1], dtype=self._fused[0].dtype)
        if n == 1:
            return self._fused[0][:, indexset[0], :].reshape(-1)

        cache = self._cacheleft[n]
        key = tuple(indexset)
        if key not in cache:
            prefix = self.evaluate_left(indexset[:-1])
            cache[key] = (prefix.reshape(1, -1) @ self._fused[n - 1][:, indexset[n - 1], :]).reshape(-1)
        return cache[key]

    def evaluate_right(self, indexset: Sequence[int]) -> np.ndarray:
        n = len(indexset)
        if n > len(self):
            raise ValueError("For evaluate_right, number of indices must be <= the number of TTCache legs.")
        if n == 0:
            return np.array([1], dtype=self._fused[0].dtype)
        if n == 1:
            return self._fused[-1][:, indexset[0], :].reshape(-1)

        ell = len(self) - n
        cache = self._cacheright[n]
        key = tuple(indexset)
        if key not in cache:
            suffix = self.evaluate_right(indexset[1:])
            cache[key] = (self._fused[ell][:, indexset[0], :] @ suffix.reshape(-1, 1)).reshape(-1)
        return cache[key]

    def evaluate(self, indexset: Sequence[int], usecache: bool = True, midpoint: int | None = None):
        if len(self) != len(indexset):
            raise ValueError(
                f"To evaluate a tensor train of length {len(self)}, need {len(self)} index "
                f"values, but got {len(indexset)}."
            )
        if usecache:
            if midpoint is None:
                midpoint = len(self) // 2
            left = self.evaluate_left(indexset[:midpoint])
            right = self.evaluate_right(indexset[midpoint:])
            # unconjugated dot (tensor-network convention, not the Hermitian inner product)
            return np.dot(left, right)

        result = None
        for T, i in zip(self._fused, indexset):
            mat = T[:, i, :]
            result = mat if result is None else result @ mat
        return result[0, 0]

    def __call__(self, *args):
        if len(args) == 1:
            return self.evaluate(args[0], usecache=True)
        if len(args) == 3:
            return self.batchevaluate(*args)
        raise TypeError("TTCache.__call__ expects either (indexset,) or (leftindexset, rightindexset, ncent)")

    def batchevaluate(
        self, leftindexset: Sequence, rightindexset: Sequence, ncent: int, projector: Sequence | None = None
    ) -> np.ndarray:
        """projector[c] entries are None (free) or a fixed local-index value per site leg."""
        dtype = self._fused[0].dtype
        if len(leftindexset) * len(rightindexset) == 0:
            return np.empty((0,) * (ncent + 2), dtype=dtype)

        N = len(self)
        nleft = len(leftindexset[0])
        nright = len(rightindexset[0])
        nleftindexset = len(leftindexset)
        nrightindexset = len(rightindexset)
        computed_ncent = N - nleft - nright
        if computed_ncent != ncent:
            raise ValueError(f"Invalid parameter M: {ncent}")

        if projector is None:
            projector = [[None] * len(d) for d in self._sitedims[nleft:N - nright]]
        if len(projector) != ncent:
            raise ValueError(f"Invalid length of projector: {projector}, correct length should be M={ncent}")
        for idx, n in enumerate(range(nleft, N - nright)):
            if len(projector[idx]) != len(self._sitedims[n]):
                raise ValueError(f"Invalid projector at {n}: {projector[idx]}")
            for p, d in zip(projector[idx], self._sitedims[n]):
                if p is not None and not (0 <= p < d):
                    raise ValueError(f"Invalid projector: {projector[idx]}")

        DL = self._fused[nleft].shape[0] if 0 < nleft < N else 1
        lenv = np.ones((nleftindexset, DL), dtype=dtype)
        if nleft > 0:
            for il, lindex in enumerate(leftindexset):
                lenv[il, :] = self.evaluate_left(lindex)

        DR = self._fused[N - nright - 1].shape[-1] if 0 < nright < N else 1
        renv = np.ones((DR, nrightindexset), dtype=dtype)
        if nright > 0:
            for ir, rindex in enumerate(rightindexset):
                renv[:, ir] = self.evaluate_right(rindex)

        localdim = [0] * ncent
        for idx, n in enumerate(range(nleft, N - nright)):
            proj = projector[idx]
            slc = tuple(slice(None) if p is None else p for p in proj)
            s = self.sitetensor(n)[(slice(None), *slc, slice(None))]
            T_ = s.reshape(s.shape[0], -1, s.shape[-1])
            localdim[idx] = T_.shape[1]
            bonddim_ = T_.shape[0]
            lenv = lenv.reshape(-1, bonddim_) @ T_.reshape(bonddim_, -1)

        bonddim_ = renv.shape[0]
        lenv = lenv.reshape(-1, bonddim_) @ renv.reshape(bonddim_, -1)
        return lenv.reshape((nleftindexset, *localdim, nrightindexset))
