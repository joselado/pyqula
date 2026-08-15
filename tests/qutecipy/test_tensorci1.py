import itertools

import numpy as np

from pyqula.qutecipytk.tci1 import TensorCI1, crossinterpolate1
from pyqula.qutecipytk.tensortrain.core import TensorTrain


def test_trivial_mps():
    n = 5

    def f(v):
        return 1

    tci = TensorCI1(np.int64, [2] * n)
    for i in range(n):
        assert len(tci.Iset[i]) == 0
        assert len(tci.Jset[i]) == 0
        assert tci.T[i].shape == (0, 2, 0)
        assert tci.P[i].shape == (0, 0)
        assert len(tci.PiIset[i]) == 0
        assert len(tci.PiJset[i]) == 0
    for i in range(n - 1):
        assert tci.Pi[i].shape == (0, 0)
        assert tci.pivoterrors[i] == np.inf

    tci = TensorCI1.from_function(np.int64, f, [2] * n, [0] * n)
    for i in range(n):
        assert tci.Iset[i].fromint == [tuple([0] * i)]
        assert tci.Jset[i].fromint == [tuple([0] * (n - i - 1))]
        assert np.array_equal(tci.T[i], np.ones((1, 2, 1)))
        assert np.array_equal(tci.P[i], np.ones((1, 1)))
        assert tci.PiIset[i].fromint == [tuple([0] * i) + (k,) for k in range(2)]
        assert tci.PiJset[i].fromint == [(k,) + tuple([0] * (n - i - 1)) for k in range(2)]
    for i in range(n - 1):
        assert np.array_equal(tci.Pi[i], np.ones((2, 2)))

    # Because the MPS is trivial, no new pivot should be added.
    for i in range(n - 1):
        tci.add_pivot(i, f, 1e-8)

    for i in range(n):
        assert len(tci.Iset[i]) == 1
        assert len(tci.Jset[i]) == 1
        assert np.array_equal(tci.T[i], np.ones((1, 2, 1)))
        assert np.array_equal(tci.P[i], np.ones((1, 1)))
        assert len(tci.PiIset[i]) == 2
        assert len(tci.PiJset[i]) == 2
    for i in range(n - 1):
        assert np.array_equal(tci.Pi[i], np.ones((2, 2)))


def test_lorentz_mps():
    # f uses (x+1) so that 0-based position x represents the same value as Julia's
    # 1-based index x -- this reproduces the exact numerical scenario of the
    # upstream Julia test (test_tensorci1.jl "Lorentz MPS"), including its
    # hardcoded globalpivot and the exact expected post-insertion ranks.
    for coeff, dtype in [(1.0, np.float64), (1.0j, np.complex128)]:
        n = 5

        def f(v, coeff=coeff):
            return coeff / (sum((x + 1) ** 2 for x in v) + 1)

        tci = TensorCI1.from_function(dtype, f, [10] * n, [0] * n)
        assert tci.linkdims() == [1] * (n - 1)
        assert tci.rank() == 1

        for p in range(n - 1):
            tci.add_pivot(p, f, 1e-8)
        assert tci.linkdims() == [2] * (n - 1)
        assert tci.rank() == 2

        globalpivot = [1, 8, 9, 4, 6]  # 0-based analogue of Julia's [2, 9, 10, 5, 7]
        tci.add_global_pivot(f, globalpivot, 1e-12)
        assert tci.linkdims() == [3] * (n - 1)
        assert tci.rank() == 3
        assert np.isclose(tci.evaluate(globalpivot), f(globalpivot))

        tci.add_global_pivot(f, globalpivot, 1e-12)
        assert tci.linkdims() == [3] * (n - 1)
        assert tci.rank() == 3
        assert np.isclose(tci.evaluate(globalpivot), f(globalpivot))

        for it in range(4, 9):
            for p in range(n - 1):
                tci.add_pivot(p, f, 1e-8)
            assert tci.linkdims() == [it] * (n - 1)
            assert tci.rank() == it

        tci2, ranks, errors = crossinterpolate1(
            dtype, f, [10] * n, [0] * n, tolerance=1e-8, maxiter=8, sweepstrategy="forward"
        )
        assert tci.linkdims() == tci2.linkdims()
        assert tci.rank() == tci2.rank()

        tci3, ranks, errors = crossinterpolate1(dtype, f, [10] * n, [0] * n, tolerance=1e-12, maxiter=200)
        assert all(e <= 1e-12 for e in tci3.pivoterrors)
        assert all(d <= 200 for d in tci3.linkdims())
        assert tci3.rank() <= 200

        tci4, ranks, errors = crossinterpolate1(
            dtype, f, [10] * n, [0] * n, tolerance=1e-12, maxiter=200,
            additionalpivots=[
                [9, 7, 9, 3, 3],
                [4, 3, 7, 8, 2],
                [6, 6, 9, 4, 8],
                [6, 6, 9, 4, 8],
            ],
        )
        assert all(e <= 1e-12 for e in tci4.pivoterrors)
        assert all(d <= 200 for d in tci4.linkdims())
        assert tci4.rank() <= 200

        tt3 = TensorTrain.from_tt_like(tci3)
        for v in itertools.product(range(3), repeat=n):
            value = tci3.evaluate(list(v))
            manual = None
            for p in range(n):
                mat = tt3[p][:, v[p], :]
                manual = mat if manual is None else manual @ mat
            assert np.isclose(value, manual[0, 0])
            assert np.isclose(value, f(v))
