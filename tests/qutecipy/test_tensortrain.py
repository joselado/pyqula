import itertools

import numpy as np
import pytest
import scipy.optimize

from pyqula.qutecipytk.tensortrain.core import TensorTrain, TensorTrainFit, add, subtract


def _random_tt(rng, dtype, linkdims, localdims):
    L = len(localdims)
    return TensorTrain([
        (rng.standard_normal((linkdims[n], localdims[n], linkdims[n + 1]))
         + (1j * rng.standard_normal((linkdims[n], localdims[n], linkdims[n + 1]))
            if np.issubdtype(dtype, np.complexfloating) else 0)).astype(dtype)
        for n in range(L)
    ])


def test_fulltensor():
    rng = np.random.default_rng(0)
    for dtype in [np.float64, np.complex128]:
        linkdims = [1, 2, 3, 1]
        L = len(linkdims) - 1
        localdims = [4] * L
        tts = _random_tt(rng, dtype, linkdims, localdims)

        def brute_fulltensor(obj):
            sitedims_ = obj.sitedims()
            localdims_ = [int(np.prod(d)) for d in sitedims_]
            r = np.empty(localdims_, dtype=dtype)
            for idx in itertools.product(*[range(d) for d in localdims_]):
                r[idx] = obj.evaluate(list(idx))
            return r

        assert np.allclose(brute_fulltensor(tts), tts.fulltensor())


def test_shape_conversion():
    rng = np.random.default_rng(1)
    for dtype in [np.float64, np.complex128]:
        linkdims = [1, 2, 3, 1]
        L = len(linkdims) - 1
        localdims = [4] * L
        tts = _random_tt(rng, dtype, linkdims, localdims)
        tto = TensorTrain.reshaped(tts, [[2, 2]] * L)
        tts_reconst = TensorTrain.reshaped(tto, [[4]] * L)

        for n in range(L):
            assert np.allclose(tts[n], tts_reconst[n])

        with pytest.raises(ValueError):
            TensorTrain.reshaped(tts, [[2, 3]] * L)
        with pytest.raises(ValueError):
            TensorTrain.reshaped(tts, [[1, 2, 3]] * L)


def test_ttfit():
    # TensorTrainFit is optimizer-agnostic in the Julia original (the caller supplies
    # the optimizer, e.g. Optim.jl + Zygote there); here scipy.optimize.minimize with
    # its own finite-difference gradient plays that role (see CLAUDE.md External deps).
    # atol=1e-6: L-BFGS-B with a finite-difference gradient converges this toy 2-point
    # fit to ~1e-9 (measured), well inside Julia's exact-gradient isapprox tolerance
    # (~1.5e-8) -- 1e-6 leaves margin for platform/BLAS variation without loosening the
    # test past what the algorithm actually achieves (unlike the previous atol=1e-4,
    # ~4 orders of magnitude looser than necessary).
    localdims = [2, 2, 2]
    linkdims = [1, 2, 3, 1]
    options = {"ftol": 1e-15, "gtol": 1e-10}

    for dtype in [np.float64, np.complex128]:
        rng = np.random.default_rng(10)
        tt0 = _random_tt(rng, dtype, linkdims, localdims)

        indexsets = [[0, 0, 0], [1, 1, 1]]
        values = rng.standard_normal(len(indexsets))
        if np.issubdtype(dtype, np.complexfloating):
            values = values + 1j * rng.standard_normal(len(indexsets))
        values = values.astype(dtype)

        ttfit = TensorTrainFit(indexsets, values, tt0)
        x0 = tt0.flatten()

        if np.issubdtype(dtype, np.complexfloating):
            # scipy.optimize.minimize requires real-valued parameters; pack/unpack
            # complex coefficients as concatenated (real, imag) halves.
            n = len(x0)

            def loss(x_real, n=n):
                return ttfit(x_real[:n] + 1j * x_real[n:])

            res = scipy.optimize.minimize(
                loss, np.concatenate([x0.real, x0.imag]), method="L-BFGS-B", options=options
            )
            xopt = res.x[:n] + 1j * res.x[n:]
        else:
            res = scipy.optimize.minimize(ttfit, x0, method="L-BFGS-B", options=options)
            xopt = res.x

        tensors = ttfit.to_tensors(xopt)
        ttopt = TensorTrain(tensors)
        got = [ttopt.evaluate(idx) for idx in indexsets]
        assert np.allclose(got, values, atol=1e-6)


def test_addition_and_multiplication():
    for dtype in [np.float64, np.complex128]:
        rng = np.random.default_rng(10)
        localdims = [2, 2, 2]
        linkdims = [1, 2, 3, 1]
        L = len(localdims)
        tt1 = _random_tt(rng, dtype, linkdims, localdims)
        tt2 = _random_tt(rng, dtype, linkdims, localdims)

        indices = list(itertools.product(range(2), range(2), range(2)))
        ttadd = add(tt1, tt2)
        assert np.allclose([ttadd(v) for v in indices], [tt1(v) + tt2(v) for v in indices])
        ttadd2 = tt1 + tt2
        assert np.allclose([ttadd2(v) for v in indices], [tt1(v) + tt2(v) for v in indices])

        tt1mul = 1.6 * tt1
        assert np.allclose([tt1mul(v) for v in indices], [1.6 * tt1(v) for v in indices])

        tt1div = tt1mul / 3.2
        assert np.allclose([tt1div(v) for v in indices], [tt1(v) / 2.0 for v in indices])

        tt1sub = tt1 - tt1div
        assert np.allclose([tt1sub(v) for v in indices], [tt1(v) / 2.0 for v in indices])

        ttshort = TensorTrain(tt1.sitetensors()[: L - 1])
        with pytest.raises(ValueError):
            add(tt1, ttshort)

        multileg = [
            (rng.standard_normal((linkdims[n], localdims[n], localdims[n], linkdims[n + 1]))).astype(dtype)
            for n in range(L)
        ]
        ttmultileg = TensorTrain(multileg)
        with pytest.raises(ValueError):
            add(tt1, ttmultileg)
        ttmultileg2 = ttmultileg + ttmultileg
        indices_multileg = [list(zip(v, v)) for v in indices]
        got = [ttmultileg2.evaluate(v) for v in indices_multileg]
        expect = [2 * ttmultileg.evaluate(v) for v in indices_multileg]
        assert np.allclose(got, expect)


def test_norm():
    sitedims_ = [[2], [2], [2]]
    N = len(sitedims_)
    bonddims = [1, 1, 1, 1]

    tt = TensorTrain([np.ones((bonddims[n], *sitedims_[n], bonddims[n + 1])) for n in range(N)])

    prod_dims = np.prod([d[0] for d in sitedims_])
    assert np.isclose(tt.norm2(), prod_dims)
    assert np.isclose((2 * tt).norm2(), 4 * prod_dims)
    assert np.isclose(tt.norm2(), tt.norm() ** 2)


def test_compress_svd():
    rng = np.random.default_rng(1234)
    N = 10
    sitedims_ = [[2] for _ in range(N)]
    chi = 10
    tol = 0.1
    bonddims = [1] + [chi] * (N - 1) + [1]

    tt = TensorTrain([rng.standard_normal((bonddims[n], *sitedims_[n], bonddims[n + 1])) for n in range(N)])

    import copy

    tt_compressed = copy.deepcopy(tt)
    tt_compressed.compress("SVD", tolerance=tol)
    diff = subtract(tt, tt_compressed)
    assert np.sqrt(diff.norm2() / tt.norm2()) < np.sqrt(N) * tol

    tt_compressed2 = copy.deepcopy(tt)
    tt_compressed2.compress("SVD", tolerance=tt.norm() * tol, normalizeerror=False)
    diff2 = subtract(tt, tt_compressed2)
    assert np.sqrt(diff2.norm2() / tt.norm2()) < np.sqrt(N) * tol


def test_tensor_train_cast():
    rng = np.random.default_rng(10)
    localdims = [2, 2, 2]
    linkdims = [1, 2, 3, 1]
    L = len(localdims)

    tt1 = _random_tt(rng, np.float64, linkdims, localdims)
    tt2 = tt1.astype(np.complex128)
    assert np.allclose(tt1.fulltensor(), tt2.fulltensor())
    tt3 = tt2.astype(np.float64)
    assert np.allclose(tt1.fulltensor(), tt3.fulltensor())
