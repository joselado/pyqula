"""Port of tensortrain.jl: concrete dense TensorTrain (MPS/MPO)."""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from pyqula.qutecipytk.matrix.luci import MatrixLUCI
from pyqula.qutecipytk.matrix.rrlu import rrlu
from pyqula.qutecipytk.tensortrain.base import AbstractTensorTrain

_UNBOUNDED_RANK = 2**62


class TensorTrain(AbstractTensorTrain):
    def __init__(self, sitetensors: Sequence[np.ndarray]):
        sitetensors = [np.asarray(T) for T in sitetensors]
        for i in range(len(sitetensors) - 1):
            if sitetensors[i].shape[-1] != sitetensors[i + 1].shape[0]:
                raise ValueError(
                    f"The tensors at {i} and {i + 1} must have consistent dimensions "
                    "for a tensor train."
                )
        self._sitetensors = sitetensors

    def sitetensors(self) -> list[np.ndarray]:
        return self._sitetensors

    @classmethod
    def from_tt_like(cls, tci: AbstractTensorTrain) -> "TensorTrain":
        """Materialize any tensor-train-like object (TCI1, TCI2, ...) as a TensorTrain."""
        return cls(list(tci.sitetensors()))

    @classmethod
    def reshaped(cls, tt: AbstractTensorTrain, localdims: Sequence[Sequence[int]], dtype=None) -> "TensorTrain":
        """Convert/reshape a tensor-train-like object's site tensors into ``localdims``
        (a list of per-site local-index shapes), e.g. splitting a 3-leg TT into a 4-leg MPO."""
        sitetensors = []
        for n, t in enumerate(tt.sitetensors()):
            expected = int(np.prod(localdims[n])) if len(localdims[n]) else 1
            actual = int(np.prod(t.shape[1:-1])) if t.ndim > 2 else 1
            if expected != actual:
                raise ValueError(f"The local dimensions at n={n} must match the tensor sizes.")
            arr = t.astype(dtype) if dtype is not None else t
            sitetensors.append(arr.reshape((t.shape[0], *localdims[n], t.shape[-1])))
        return cls(sitetensors)

    def astype(self, dtype) -> "TensorTrain":
        return TensorTrain([T.astype(dtype) for T in self._sitetensors])

    def compress(
        self, method: str = "LU", tolerance: float = 1e-12, maxbonddim: int | None = None,
        normalizeerror: bool = True,
    ) -> None:
        """Two-pass sweep compression (mutates self): left-to-right
        orthogonalize with no truncation, then right-to-left truncate."""
        maxbonddim = maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK
        n = len(self._sitetensors)

        for ell in range(n - 1):
            shapel = self._sitetensors[ell].shape
            mat = self._sitetensors[ell].reshape(-1, shapel[-1])
            left, right, newbonddim = _factorize(
                mat, method, tolerance=0.0, maxbonddim=_UNBOUNDED_RANK, leftorthogonal=True
            )
            self._sitetensors[ell] = left.reshape(shapel[:-1] + (newbonddim,))
            shaper = self._sitetensors[ell + 1].shape
            nexttensor = right @ self._sitetensors[ell + 1].reshape(shaper[0], -1)
            self._sitetensors[ell + 1] = nexttensor.reshape((newbonddim,) + shaper[1:])

        for ell in range(n - 1, 0, -1):
            shaper = self._sitetensors[ell].shape
            mat = self._sitetensors[ell].reshape(shaper[0], -1)
            left, right, newbonddim = _factorize(
                mat, method, tolerance=tolerance, maxbonddim=maxbonddim,
                leftorthogonal=False, normalizeerror=normalizeerror,
            )
            self._sitetensors[ell] = right.reshape((newbonddim,) + shaper[1:])
            shapel = self._sitetensors[ell - 1].shape
            nexttensor = self._sitetensors[ell - 1].reshape(-1, shapel[-1]) @ left
            self._sitetensors[ell - 1] = nexttensor.reshape(shapel[:-1] + (newbonddim,))

    def multiply(self, a) -> None:
        self._sitetensors[-1] = self._sitetensors[-1] * a

    def scaled(self, a) -> "TensorTrain":
        tt2 = copy.deepcopy(self)
        tt2.multiply(a)
        return tt2

    def __mul__(self, a) -> "TensorTrain":
        return self.scaled(a)

    def __rmul__(self, a) -> "TensorTrain":
        return self.scaled(a)

    def divide(self, a) -> None:
        self._sitetensors[-1] = self._sitetensors[-1] / a

    def divided(self, a) -> "TensorTrain":
        tt2 = copy.deepcopy(self)
        tt2.divide(a)
        return tt2

    def __truediv__(self, a) -> "TensorTrain":
        return self.divided(a)

    def __add__(self, other: AbstractTensorTrain) -> "TensorTrain":
        return add(self, other)

    def __sub__(self, other: AbstractTensorTrain) -> "TensorTrain":
        return subtract(self, other)

    def flatten(self) -> np.ndarray:
        return np.concatenate([T.reshape(-1) for T in self._sitetensors])

    def fulltensor(self) -> np.ndarray:
        sitedims_ = self.sitedims()
        localdims = [int(np.prod(d)) if d else 1 for d in sitedims_]
        result = self._sitetensors[0].reshape(localdims[0], -1)
        leftdim = localdims[0]
        for l in range(1, len(self)):
            st = self._sitetensors[l]
            nextmatrix = st.reshape(st.shape[0], localdims[l] * st.shape[-1])
            leftdim *= localdims[l]
            result = (result @ nextmatrix).reshape(leftdim, st.shape[-1])
        returnsize = [d for dims in sitedims_ for d in dims]
        return result.reshape(returnsize)


def tensortrain(tci: AbstractTensorTrain) -> TensorTrain:
    return TensorTrain.from_tt_like(tci)


def _factorize(
    A: np.ndarray, method: str, tolerance: float, maxbonddim: int, leftorthogonal: bool = False,
    normalizeerror: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    reltol = 1e-14
    abstol = 0.0
    if normalizeerror:
        reltol = tolerance
    else:
        abstol = tolerance

    if method == "LU":
        factorization = rrlu(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        return factorization.left(), factorization.right(), factorization.npivots()
    if method == "CI":
        factorization = MatrixLUCI.from_matrix(
            A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal
        )
        return factorization.left(), factorization.right(), factorization.npivots()
    if method == "SVD":
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        total_energy = np.sum(S ** 2)
        err = np.array([np.sum(S[n + 1:] ** 2) for n in range(len(S))])
        normalized_err = err / total_energy if total_energy > 0 else err

        below_abs = np.nonzero(err < abstol ** 2)[0]
        trunc_abs = int(below_abs[0]) + 1 if below_abs.size > 0 else len(err)
        below_rel = np.nonzero(normalized_err < reltol ** 2)[0]
        trunc_rel = int(below_rel[0]) + 1 if below_rel.size > 0 else len(err)
        trunci = min(trunc_abs, trunc_rel, maxbonddim)

        if leftorthogonal:
            return U[:, :trunci], np.diag(S[:trunci]) @ Vt[:trunci, :], trunci
        return U[:, :trunci] @ np.diag(S[:trunci]), Vt[:trunci, :], trunci

    raise NotImplementedError(f"Factorization method {method!r} not implemented.")


def reverse(tt: AbstractTensorTrain) -> TensorTrain:
    new_tensors = [
        np.transpose(T, (T.ndim - 1,) + tuple(range(1, T.ndim - 1)) + (0,))
        for T in tt.sitetensors()
    ]
    new_tensors.reverse()
    return tensortrain(TensorTrain(new_tensors))


def _add_tt_tensor(
    A: np.ndarray, B: np.ndarray, factorA=1.0, factorB=1.0, lefttensor: bool = False, righttensor: bool = False
) -> np.ndarray:
    if A.ndim != B.ndim:
        raise ValueError(
            "Elementwise addition only works if both tensors have the same number of indices, "
            f"but A and B have different numbers ({A.ndim} and {B.ndim}) of indices."
        )
    nd = A.ndim
    offset1 = 0 if lefttensor else A.shape[0]
    offset3 = 0 if righttensor else A.shape[-1]
    shape = (offset1 + B.shape[0],) + A.shape[1:-1] + (offset3 + B.shape[-1],)
    C = np.zeros(shape, dtype=np.result_type(A, B))

    idx_a = (slice(0, A.shape[0]),) + (slice(None),) * (nd - 2) + (slice(0, A.shape[-1]),)
    C[idx_a] = factorA * A
    idx_b = (
        (slice(offset1, offset1 + B.shape[0]),) + (slice(None),) * (nd - 2)
        + (slice(offset3, offset3 + B.shape[-1]),)
    )
    C[idx_b] = factorB * B
    return C


def add(
    lhs: AbstractTensorTrain, rhs: AbstractTensorTrain, factorlhs=1.0, factorrhs=1.0,
    tolerance: float = 0.0, maxbonddim: int | None = None,
) -> TensorTrain:
    """C = add(A, B) such that C(v) ~= factorlhs*A(v) + factorrhs*B(v).
    Increases bond dimension to chi1+chi2, then recompresses via SVD."""
    if len(lhs) != len(rhs):
        raise ValueError(
            f"Two tensor trains with different length ({len(lhs)} and {len(rhs)}) "
            "cannot be added elementwise."
        )
    L = len(lhs)
    tensors = [
        _add_tt_tensor(
            lhs.sitetensor(ell), rhs.sitetensor(ell),
            factorA=factorlhs if ell == L - 1 else 1.0,
            factorB=factorrhs if ell == L - 1 else 1.0,
            lefttensor=(ell == 0), righttensor=(ell == L - 1),
        )
        for ell in range(L)
    ]
    tt = TensorTrain(tensors)
    tt.compress("SVD", tolerance=tolerance, maxbonddim=maxbonddim if maxbonddim is not None else _UNBOUNDED_RANK)
    return tt


def subtract(
    lhs: AbstractTensorTrain, rhs: AbstractTensorTrain, tolerance: float = 0.0, maxbonddim: int | None = None
) -> TensorTrain:
    return add(lhs, rhs, factorrhs=-1.0, tolerance=tolerance, maxbonddim=maxbonddim)


class TensorTrainFit:
    """Fitting data with a TensorTrain object; useful when interpolated data is noisy.
    Optimizer-agnostic: exposes a loss function of the flattened core parameters,
    the caller supplies the optimizer (e.g. scipy.optimize.minimize)."""

    def __init__(self, indexsets: Sequence, values: Sequence, tt: TensorTrain):
        self.indexsets = list(indexsets)
        self.values = np.asarray(values)
        self.tt = tt
        offsets = [0]
        for T in tt.sitetensors():
            offsets.append(offsets[-1] + T.size)
        self.offsets = offsets

    def to_tensors(self, x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x)
        return [
            x[self.offsets[n]:self.offsets[n + 1]].reshape(self.tt.sitetensor(n).shape)
            for n in range(len(self.tt))
        ]

    def __call__(self, x: np.ndarray) -> float:
        tensors = self.to_tensors(x)
        total = 0.0
        for indexset, val in zip(self.indexsets, self.values):
            total += abs(_evaluate_tensors(tensors, indexset) - val) ** 2
        return total


def _evaluate_tensors(tensors: Sequence[np.ndarray], indexset) -> complex:
    result = None
    for T, i in zip(tensors, indexset):
        mat = T[:, i, :]
        result = mat if result is None else result @ mat
    return result[0, 0]
