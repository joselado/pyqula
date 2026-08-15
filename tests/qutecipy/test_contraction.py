import itertools

import numpy as np
import pytest

from pyqula.qutecipytk.contraction import Contraction, contract
from pyqula.qutecipytk.tensortrain.core import TensorTrain

rng = np.random.default_rng(0)


def _rand_complex(shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _tomat(tto: TensorTrain) -> np.ndarray:
    sitedims = tto.sitedims()
    d1 = [s[0] for s in sitedims]
    d2 = [s[1] for s in sitedims]
    mat = np.empty((int(np.prod(d1)), int(np.prod(d2))), dtype=complex)
    for i, inds1 in enumerate(itertools.product(*[range(d) for d in d1])):
        for j, inds2 in enumerate(itertools.product(*[range(d) for d in d2])):
            mat[i, j] = tto.evaluate(list(zip(inds1, inds2)))
    return mat


def _tovec(tt: TensorTrain) -> np.ndarray:
    d1 = [s[0] for s in tt.sitedims()]
    return np.array([tt.evaluate(list(idx)) for idx in itertools.product(*[range(d) for d in d1])])


def _gen_tto_tto():
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]
    a = TensorTrain([_rand_complex((bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1])) for n in range(N)])
    b = TensorTrain([_rand_complex((bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n + 1])) for n in range(N)])
    return N, a, b, localdims1, localdims2, localdims3


def _gen_tto_tts():
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [3, 3, 3, 3]
    localdims2 = [3, 3, 3, 3]
    a = TensorTrain([_rand_complex((bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1])) for n in range(N)])
    b = TensorTrain([_rand_complex((bonddims_b[n], localdims2[n], bonddims_b[n + 1])) for n in range(N)])
    return N, a, b, localdims1, localdims2


def test_contract_matches_tensordot():
    a = rng.random((2, 3, 4))
    b = rng.random((2, 5, 4))
    ab = np.tensordot(a, b, axes=([0, 2], [0, 2]))
    expect = np.transpose(a, (1, 0, 2)).reshape(3, -1) @ np.transpose(b, (0, 2, 1)).reshape(-1, 5)
    assert np.allclose(ab, expect)


@pytest.mark.parametrize("f", [None, lambda x: 2 * x])
@pytest.mark.parametrize("algorithm", ["TCI", "naive"])
def test_mpo_mpo_contraction(f, algorithm):
    N, a, b, localdims1, localdims2, localdims3 = _gen_tto_tto()

    if f is not None and algorithm == "naive":
        with pytest.raises(ValueError):
            contract(a, b, f=f, algorithm=algorithm)
        return

    ab = contract(a, b, f=f, algorithm=algorithm, tolerance=1e-10)
    assert ab.sitedims() == [[localdims1[i], localdims3[i]] for i in range(N)]
    expect = _tomat(a) @ _tomat(b)
    if f is not None:
        expect = f(expect)
    assert np.allclose(_tomat(ab), expect, atol=1e-8)


def test_contraction_batchevaluate_with_projector():
    N, a, b, localdims1, localdims2, localdims3 = _gen_tto_tto()
    ab = Contraction(a, b)

    leftindexset = [[0]]
    rightindexset = [[0]]
    ref = ab.batchevaluate(leftindexset, rightindexset, 2)
    # Unfuse each site's combined index using the same "i fast" convention as
    # Contraction._fuse_idx/_unfuse_idx (i + d_i*j) -- NOT a plain C-order reshape.
    ref_mi = np.zeros((1, 2, 2, 2, 2, 1), dtype=complex)
    for c1 in range(4):
        i1, j1 = c1 % 2, c1 // 2
        for c2 in range(4):
            i2, j2 = c2 % 2, c2 // 2
            ref_mi[0, i1, j1, i2, j2, 0] = ref[0, c1, c2, 0]

    res = ab.batchevaluate(leftindexset, rightindexset, 2, projector=[[0, None], [1, None]])
    assert np.allclose(res.reshape(-1), ref_mi[:, 0, :, 1, :, :].reshape(-1))

    res = ab.batchevaluate(leftindexset, rightindexset, 2, projector=[[0, None], [1, 1]])
    assert np.allclose(res.reshape(-1), ref_mi[:, 0, :, 1, 1, :].reshape(-1))

    res = ab.batchevaluate(leftindexset, rightindexset, 2, projector=[[0, 1], [1, None]])
    assert np.allclose(res.reshape(-1), ref_mi[:, 0, 1, 1, :, :].reshape(-1))


@pytest.mark.parametrize("f", [None, lambda x: 2 * x])
@pytest.mark.parametrize("algorithm", ["TCI", "naive"])
def test_mpo_mps_contraction(f, algorithm):
    N, a, b, localdims1, localdims2 = _gen_tto_tts()

    if f is not None and algorithm == "naive":
        with pytest.raises(ValueError):
            contract(a, b, f=f, algorithm=algorithm)
        with pytest.raises(ValueError):
            contract(b, a, f=f, algorithm=algorithm)
        return

    ab = contract(a, b, f=f, algorithm=algorithm, tolerance=1e-10)
    ba = contract(b, a, f=f, algorithm=algorithm, tolerance=1e-10)
    assert ab.sitedims() == [[localdims1[i]] for i in range(N)]

    expect_ab = _tomat(a) @ _tovec(b)
    expect_ba = _tovec(b) @ _tomat(a)
    if f is not None:
        expect_ab = f(expect_ab)
        expect_ba = f(expect_ba)
    assert np.allclose(_tovec(ab), expect_ab, atol=1e-8)
    assert np.allclose(_tovec(ba), expect_ba, atol=1e-8)


@pytest.mark.parametrize("method", ["SVD", "LU"])
def test_mpo_mpo_contraction_zipup(method):
    N, a, b, localdims1, localdims2, localdims3 = _gen_tto_tto()
    ab = contract(a, b, algorithm="zipup", method=method)
    assert np.allclose(_tomat(ab), _tomat(a) @ _tomat(b), atol=1e-8)


@pytest.mark.parametrize("method", ["SVD", "LU"])
def test_mpo_mps_contraction_zipup(method):
    N, a, b, localdims1, localdims2 = _gen_tto_tts()
    ab = contract(a, b, algorithm="zipup", method=method)
    assert np.allclose(_tovec(ab), _tomat(a) @ _tovec(b), atol=1e-8)
