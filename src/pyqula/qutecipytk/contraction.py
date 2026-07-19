"""Port of contraction.jl: lazy MPO x MPO contraction (Contraction, exposing
the BatchEvaluator interface so it can itself be fed into crossinterpolate2),
plus three concrete contraction algorithms.

``_contract(a, b, idx_a, idx_b)`` (generic reshape/permute/matmul contraction
primitive) maps directly onto ``np.tensordot(a, b, axes=(idx_a, idx_b))`` --
no separate helper needed, unlike the Julia original which hand-rolls it via
permutedims+reshape+matmul (numpy's tensordot already does exactly that).

Contraction's projector semantics differ from TTCache's: a *fixed*
projector entry here keeps the corresponding leg as a size-1 axis (not
squeezed away), matching Julia's `reshape(..., shape_ab[1], ...)` step --
achieved here more directly with a length-1 slice instead of scalar
indexing + reshape-back.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain
from pyqula.qutecipytk.tensortrain.batcheval import BatchEvaluator
from pyqula.qutecipytk.tensortrain.core import TensorTrain, _factorize
from pyqula.qutecipytk.util import optfirstpivot

_UNBOUNDED_RANK = 2**62


def _proj_slice(p: int | None) -> slice:
    return slice(None) if p is None else slice(p, p + 1)


class Contraction(BatchEvaluator):
    """Lazy contraction of two 4-leg tensor trains (MPOs), optionally with
    an elementwise function f applied to the contracted scalar result."""

    def __init__(self, a: TensorTrain, b: TensorTrain, f: Callable | None = None):
        if len(a) != len(b):
            raise ValueError("Tensor trains must have the same length.")
        for n in range(len(a)):
            if a.sitetensor(n).shape[2] != b.sitetensor(n).shape[1]:
                raise ValueError(f"Tensor trains must share the identical index at n={n}!")

        self.mpo = (a, b)
        self.leftcache: dict[tuple, np.ndarray] = {}
        self.rightcache: dict[tuple, np.ndarray] = {}
        self.f = f
        localdims1 = [a.sitetensor(n).shape[1] for n in range(len(a))]
        localdims3 = [b.sitetensor(n).shape[2] for n in range(len(b))]
        self._sitedims = [[x, y] for x, y in zip(localdims1, localdims3)]

    def __len__(self) -> int:
        return len(self.mpo[0])

    def sitedims(self) -> list[list[int]]:
        return self._sitedims

    def __getitem__(self, i):
        return self.mpo[0][i]

    @property
    def dtype(self):
        return self.mpo[0].sitetensor(0).dtype

    def _localdims(self, n: int) -> tuple[int, int]:
        return (self.mpo[0].sitetensor(n).shape[1], self.mpo[1].sitetensor(n).shape[2])

    def _unfuse_idx(self, n: int, idx: int) -> tuple[int, int]:
        d1 = self._localdims(n)[0]
        return (idx % d1, idx // d1)

    def _fuse_idx(self, n: int, ij: tuple[int, int]) -> int:
        d1 = self._localdims(n)[0]
        return ij[0] + d1 * ij[1]

    def _extend_cache(self, oldcache: np.ndarray, a_ell: np.ndarray, b_ell: np.ndarray, i: int, j: int) -> np.ndarray:
        tmp1 = np.tensordot(oldcache, a_ell[:, i, :, :], axes=([0], [0]))
        return np.tensordot(tmp1, b_ell[:, :, j, :], axes=([0, 1], [0, 1]))

    def evaluate_left(self, indexset: Sequence[tuple[int, int]]) -> np.ndarray:
        if len(indexset) >= len(self):
            raise ValueError(f"Invalid indexset: {indexset}")
        a, b = self.mpo
        if len(indexset) == 0:
            return np.ones((1, 1), dtype=self.dtype)
        ell = len(indexset)
        if ell == 1:
            i, j = indexset[0]
            return a.sitetensor(0)[0, i, :, :].T @ b.sitetensor(0)[0, :, j, :]

        key = tuple(indexset)
        if key not in self.leftcache:
            i, j = indexset[-1]
            self.leftcache[key] = self._extend_cache(
                self.evaluate_left(indexset[:-1]), a.sitetensor(ell - 1), b.sitetensor(ell - 1), i, j
            )
        return self.leftcache[key]

    def evaluate_right(self, indexset: Sequence[tuple[int, int]]) -> np.ndarray:
        if len(indexset) >= len(self):
            raise ValueError(f"Invalid indexset: {indexset}")
        a, b = self.mpo
        N = len(self)
        if len(indexset) == 0:
            return np.ones((1, 1), dtype=self.dtype)
        if len(indexset) == 1:
            i, j = indexset[0]
            return a.sitetensor(N - 1)[:, i, :, 0] @ b.sitetensor(N - 1)[:, :, j, 0].T

        ell = N - len(indexset)
        key = tuple(indexset)
        if key not in self.rightcache:
            i, j = indexset[0]
            a_ell = np.transpose(a.sitetensor(ell), (3, 1, 2, 0))
            b_ell = np.transpose(b.sitetensor(ell), (3, 1, 2, 0))
            self.rightcache[key] = self._extend_cache(self.evaluate_right(indexset[1:]), a_ell, b_ell, i, j)
        return self.rightcache[key]

    def evaluate(self, indexset: Sequence):
        if len(self) != len(indexset):
            raise ValueError(f"Length mismatch: {len(self)} != {len(indexset)}")
        if len(indexset) > 0 and isinstance(indexset[0], (int, np.integer)):
            pairs = [self._unfuse_idx(n, idx) for n, idx in enumerate(indexset)]
        else:
            pairs = [tuple(x) for x in indexset]

        midpoint = len(self) // 2
        left = self.evaluate_left(pairs[:midpoint])
        right = self.evaluate_right(pairs[midpoint:])
        res = np.sum(left * right)
        return self.f(res) if self.f is not None else res

    def __call__(self, *args):
        if len(args) == 1:
            return self.evaluate(args[0])
        if len(args) == 3:
            return self.batchevaluate(*args)
        raise TypeError("Contraction.__call__ expects (indexset,) or (leftindexset, rightindexset, ncent)")

    def batchevaluate(
        self, leftindexset: Sequence, rightindexset: Sequence, ncent: int, projector: Sequence | None = None
    ) -> np.ndarray:
        N = len(self)
        Nr = len(rightindexset[0])
        s_ = len(leftindexset[0])
        e_ = N - Nr - 1  # 0-based, inclusive last "center" site
        a, b = self.mpo

        if projector is None:
            projector = [[None, None] for _ in range(s_, e_ + 1)]
        if len(projector) != ncent:
            raise ValueError(f"Length mismatch: length of projector (={len(projector)}) must be {ncent}")
        for idx, n in enumerate(range(s_, e_ + 1)):
            if len(projector[idx]) != 2:
                raise ValueError(f"Invalid projector at {n}: {projector[idx]}, the length must be 2")
            for p, d in zip(projector[idx], self._sitedims[n]):
                if p is not None and not (0 <= p < d):
                    raise ValueError(f"Invalid projector: {projector[idx]}")

        leftindexset_unfused = [
            [self._unfuse_idx(n, idx) for n, idx in enumerate(idxs)] for idxs in leftindexset
        ]
        rightindexset_unfused = [
            [self._unfuse_idx(N - Nr + n, idx) for n, idx in enumerate(idxs)] for idxs in rightindexset
        ]

        linkdims_a = [1, *a.linkdims(), 1]
        linkdims_b = [1, *b.linkdims(), 1]

        left_ = np.empty((len(leftindexset), linkdims_a[s_], linkdims_b[s_]), dtype=self.dtype)
        for i, idx in enumerate(leftindexset_unfused):
            left_[i, :, :] = self.evaluate_left(idx)

        right_ = np.empty((linkdims_a[e_ + 1], linkdims_b[e_ + 1], len(rightindexset)), dtype=self.dtype)
        for i, idx in enumerate(rightindexset_unfused):
            right_[:, :, i] = self.evaluate_right(idx)

        leftobj = left_.reshape(*left_.shape, 1)  # (left_index, link_a, link_b, S)
        return_size_siteinds: list[int] = []
        for idx, n in enumerate(range(s_, e_ + 1)):
            proj = projector[idx]
            a_n_org = self.mpo[0].sitetensor(n)
            b_n_org = self.mpo[1].sitetensor(n)
            a_n = a_n_org[:, _proj_slice(proj[0]), :, :]
            b_n = b_n_org[:, :, _proj_slice(proj[1]), :]
            return_size_siteinds.append(a_n.shape[1] * b_n.shape[2])

            # (left_index,link_a,link_b,S) x (link_a,s_n,shared,link_a') -> (left_index,link_b,S,s_n,shared,link_a')
            tmp1 = np.tensordot(leftobj, a_n, axes=([1], [0]))
            # x (link_b,shared,s_n'',link_b') over (link_b,shared) -> (left_index,S,s_n,link_a',s_n'',link_b')
            tmp2 = np.tensordot(tmp1, b_n, axes=([1, 4], [0, 1]))
            # -> (left_index,link_a',link_b',S,s_n'',s_n) -- s_n (the first/"i" component of this site's fused
            # local index) must end up last so it's fastest-varying after flattening, matching _fuse_idx's
            # convention (i + d_i*j, i fast) -- note the swap vs. a naive (...,s_n,s_n'') transpose.
            tmp3 = np.transpose(tmp2, (0, 3, 5, 1, 4, 2))
            leftobj = tmp3.reshape(tmp3.shape[0], tmp3.shape[1], tmp3.shape[2], -1)

        return_size = (len(leftindexset), *return_size_siteinds, len(rightindexset))
        res = np.tensordot(leftobj, right_, axes=([1, 2], [0, 1]))

        if self.f is not None:
            res = np.vectorize(self.f)(res)

        return res.reshape(return_size)


def _contractsitetensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = np.tensordot(a, b, axes=([2], [1]))
    abpermuted = np.transpose(ab, (0, 3, 1, 4, 2, 5))
    la, s1, lap = a.shape[0], a.shape[1], a.shape[3]
    lb, s3, lbp = b.shape[0], b.shape[2], b.shape[3]
    return abpermuted.reshape(la * lb, s1, s3, lap * lbp)


def contract_naive(a, b=None, tolerance: float = 0.0, maxbonddim: int | None = None) -> TensorTrain:
    """Dense per-site contraction, then a single global SVD recompression
    (largest transient bond dimension, but simplest)."""
    obj = a if b is None else Contraction(a, b)
    if obj.f is not None:
        raise ValueError(
            "Naive contraction implementation cannot contract matrix product with a function. "
            "Use algorithm='TCI' instead."
        )
    aa, bb = obj.mpo
    tensors = [_contractsitetensors(aa.sitetensor(n), bb.sitetensor(n)) for n in range(len(aa))]
    tt = TensorTrain(tensors)
    if tolerance > 0 or (maxbonddim is not None and maxbonddim < _UNBOUNDED_RANK):
        tt.compress("SVD", tolerance=tolerance, maxbonddim=maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK)
    return tt


def contract_zipup(
    A: TensorTrain, B: TensorTrain, tolerance: float = 1e-12, method: str = "SVD", maxbonddim: int | None = None,
) -> TensorTrain:
    """Zip-up MPO x MPO: sweeps left to right maintaining only a small
    running interface tensor R, factorizing as it goes to keep the bond
    dimension bounded (unlike contract_naive's largest-transient-bond
    approach). See https://tensornetwork.org/mps/algorithms/zip_up_mpo/"""
    if len(A) != len(B):
        raise ValueError("Cannot contract tensor trains with different length.")
    maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
    dtype = A.sitetensor(0).dtype
    R = np.ones((1, 1, 1), dtype=dtype)

    sitetensors: list[np.ndarray] = [None] * len(A)
    for n in range(len(A)):
        RA = np.tensordot(R, A.sitetensor(n), axes=([1], [0]))
        C_pre = np.tensordot(RA, B.sitetensor(n), axes=([1, 3], [0, 1]))
        C = np.transpose(C_pre, (0, 1, 3, 2, 4))

        if n == len(A) - 1:
            sitetensors[n] = C.reshape(*C.shape[:3], 1)
            break

        left, right, newbonddim = _factorize(
            C.reshape(int(np.prod(C.shape[:3])), int(np.prod(C.shape[3:5]))), method,
            tolerance=tolerance, maxbonddim=maxbonddim,
        )
        sitetensors[n] = left.reshape(*C.shape[:3], newbonddim)
        R = right.reshape(newbonddim, *C.shape[3:5])

    return TensorTrain(sitetensors)


def _find_initial_pivots(f: Callable, localdims: Sequence[int], nmaxpivots: int) -> list[list[int]]:
    pivots = []
    for _ in range(nmaxpivots):
        pivot = [random.randrange(d) for d in localdims]
        pivot = optfirstpivot(f, localdims, pivot)
        if abs(f(pivot)) == 0.0:
            continue
        pivots.append(pivot)
    return pivots


def contract_TCI(
    A: TensorTrain, B: TensorTrain, initialpivots: int | Sequence[Sequence[int]] = 10,
    f: Callable | None = None, **kwargs,
) -> TensorTrain:
    """Treat the contraction as a black-box tensor and cross-interpolate it
    directly -- can beat contract_naive/contract_zipup when the *result* is
    low rank even if the intermediates aren't."""
    if len(A) != len(B):
        raise ValueError("Cannot contract tensor trains with different length.")
    for i in range(len(A)):
        if A.sitetensor(i).shape[2] != B.sitetensor(i).shape[1]:
            raise ValueError("Cannot contract tensor trains with non-matching site dimensions.")

    from pyqula.qutecipytk.tci2 import crossinterpolate2  # local import: avoids a tci2<->contraction import cycle

    matrixproduct = Contraction(A, B, f=f)
    localdims = [d1 * d2 for d1, d2 in matrixproduct.sitedims()]
    if isinstance(initialpivots, int):
        initialpivots = _find_initial_pivots(matrixproduct, localdims, initialpivots)
        if not initialpivots:
            raise ValueError("No initial pivots found.")

    dtype = A.sitetensor(0).dtype
    tci, ranks, errors = crossinterpolate2(dtype, matrixproduct, localdims, initialpivots, **kwargs)
    legdims = [matrixproduct._localdims(i) for i in range(len(tci))]
    # Un-fuse each site's combined local index back into its (i, j) pair. _fuse_idx uses
    # "i + d_i*j" (i fast), which is the opposite of a plain C-order reshape's "last-axis-fastest"
    # -- reshape with the dims swapped (j slow, i fast-as-last-axis) then transpose into (i, j) order.
    tensors = [
        np.transpose(t.reshape(t.shape[0], d[1], d[0], t.shape[-1]), (0, 2, 1, 3))
        for t, d in zip(tci.sitetensors(), legdims)
    ]
    return TensorTrain(tensors)


def contract(
    A: AbstractTensorTrain, B: AbstractTensorTrain, algorithm: str = "TCI", tolerance: float = 1e-12,
    maxbonddim: int | None = None, f: Callable | None = None, **kwargs,
) -> TensorTrain:
    """Contract two tensor trains A and B. 3-leg operands (plain TensorTrain,
    TensorCI1, TensorCI2) are promoted to 4-leg with a dummy size-1 leg,
    contracted, then squeezed back."""
    if not isinstance(A, TensorTrain):
        A = TensorTrain.from_tt_like(A)
    if not isinstance(B, TensorTrain):
        B = TensorTrain.from_tt_like(B)

    a_is3 = A.sitetensor(0).ndim == 3
    b_is3 = B.sitetensor(0).ndim == 3
    A4 = TensorTrain.reshaped(A, [(1, *d) for d in A.sitedims()]) if a_is3 else A
    B4 = TensorTrain.reshaped(B, [(*d, 1) for d in B.sitedims()]) if b_is3 else B

    result_dtype = np.result_type(A4.sitetensor(0).dtype, B4.sitetensor(0).dtype)
    A4 = A4.astype(result_dtype)
    B4 = B4.astype(result_dtype)
    maxbonddim_eff = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK

    if algorithm == "TCI":
        tt = contract_TCI(A4, B4, tolerance=tolerance, maxbonddim=maxbonddim_eff, f=f, **kwargs)
    elif algorithm == "naive":
        if f is not None:
            raise ValueError(
                "Naive contraction implementation cannot contract matrix product with a function. "
                "Use algorithm='TCI' instead."
            )
        tt = contract_naive(A4, B4, tolerance=tolerance, maxbonddim=maxbonddim_eff)
    elif algorithm == "zipup":
        if f is not None:
            raise ValueError(
                "Zipup contraction implementation cannot contract matrix product with a function. "
                "Use algorithm='TCI' instead."
            )
        tt = contract_zipup(A4, B4, tolerance=tolerance, maxbonddim=maxbonddim_eff)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}.")

    if a_is3 or b_is3:
        new_sitedims = [[int(np.prod(d))] for d in tt.sitedims()]
        tt = TensorTrain.reshaped(tt, new_sitedims)
    return tt
