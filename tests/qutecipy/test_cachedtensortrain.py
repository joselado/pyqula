import itertools

import numpy as np

from pyqula.qutecipytk.tensortrain.cache import TTCache

rng = np.random.default_rng(3)


def gen_indices(localdims):
    if len(localdims) == 0:
        return [[]]
    return [list(t) for t in itertools.product(*[range(d) for d in localdims])]


def test_batchevaluate():
    N = 4
    bonddims = [1, 2, 3, 2, 1]
    A = TTCache([rng.random((bonddims[n], 2, bonddims[n + 1])) for n in range(N)])

    leftindexset = [[0], [1]]
    rightindexset = [[0], [1]]

    result = A.batchevaluate(leftindexset, rightindexset, 2)
    for cindex in [[0, 0], [0, 1]]:
        for il, lindex in enumerate(leftindexset):
            for ir, rindex in enumerate(rightindexset):
                expect = A.evaluate(lindex + cindex + rindex)
                assert np.isclose(result[il, cindex[0], cindex[1], ir], expect)


def test_batchevaluate2():
    N = 4
    bonddims = [1, 2, 3, 2, 1]
    localdims = [2, 3, 3, 2]

    A = TTCache([rng.random((bonddims[n], localdims[n], bonddims[n + 1])) for n in range(N)])

    for nleft in range(N + 1):
        for nright in range(N + 1):
            leftindexset = gen_indices(localdims[:nleft])
            rightindexset = gen_indices(localdims[N - nright:])

            ncent = N - nleft - nright
            if ncent < 0:
                continue

            result = A.batchevaluate(leftindexset, rightindexset, ncent)
            for cindex in gen_indices(localdims[nleft:nleft + ncent]):
                for il, lindex in enumerate(leftindexset):
                    for ir, rindex in enumerate(rightindexset):
                        full = lindex + cindex + rindex
                        key = (il, *cindex, ir)
                        assert np.isclose(result[key], A.evaluate(full, usecache=True))
                        assert np.isclose(result[key], A.evaluate(full, usecache=False))


def test_evaluate_left_right_consistency():
    N = 5
    bonddims = [1, 2, 3, 3, 2, 1]
    localdims = [2, 2, 3, 2, 2]
    A = TTCache([rng.random((bonddims[n], localdims[n], bonddims[n + 1])) for n in range(N)])

    for full in gen_indices(localdims):
        left = A.evaluate_left(full[:2])
        right = A.evaluate_right(full[2:])
        assert np.isclose(np.dot(left, right), A.evaluate(full, usecache=False))


def test_projector():
    N = 3
    bonddims = [1, 2, 2, 1]
    localdims = [2, 2, 2]
    A = TTCache([rng.random((bonddims[n], localdims[n], bonddims[n + 1])) for n in range(N)])

    leftindexset = [[0], [1]]
    rightindexset = [[0], [1]]
    projector = [[1]]
    result = A.batchevaluate(leftindexset, rightindexset, 1, projector=projector)
    for il, lindex in enumerate(leftindexset):
        for ir, rindex in enumerate(rightindexset):
            expect = A.evaluate(lindex + [1] + rindex)
            assert np.isclose(result[il, 0, ir], expect)
