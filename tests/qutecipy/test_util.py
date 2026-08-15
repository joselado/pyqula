import numpy as np

from pyqula.qutecipytk.util import isconstant, maxabs, optfirstpivot, pushunique, randomsubset


def test_maxabs():
    s = 1.0
    assert maxabs(s, []) == 1.0

    u = [0.11892436782208138, -0.5312119179782191, 0.15328557552100353,
         0.9343319135479445, -0.04286173791053016]
    assert maxabs(s, u) == 1.0

    v = [-7.512961239635482, -0.644254782278785, 1.1242493861712504,
         6.5875869748554186, -5.400768247401216]
    assert maxabs(s, v) == 7.512961239635482


def test_optfirstpivot():
    # 0-based analogue of the Julia test (which used 1-based v and subtracted 1).
    def f(v):
        return 4 * v[2] + 2 * v[1] + v[0]

    localdims = [2, 2, 2]
    firstpivot = [0, 0, 0]
    pivot = optfirstpivot(f, localdims, firstpivot)
    assert pivot == [1, 1, 1]


def test_pushunique():
    v = [9, 29, 4, 5]
    pushunique(v, 10)
    assert v == [9, 29, 4, 5, 10]
    pushunique(v, 10)
    assert v == [9, 29, 4, 5, 10]
    pushunique(v, 2, 3)
    assert v == [9, 29, 4, 5, 10, 2, 3]
    pushunique(v, 29, 8, 4, 5)
    assert v == [9, 29, 4, 5, 10, 2, 3, 8]


def test_isconstant():
    v = [0.2925784784483926, 0.46371163262378456, 0.8705399558524782, 0.8906186678633707,
         0.31339518781618236, 0.5340770167795297, 0.8908239232701285, 0.9880309208645528,
         0.8254716317895107, 0.07517813257571271]
    u = [3, 3, 3, 3]
    assert not isconstant(v)
    assert isconstant(u)


def test_randomsubset():
    v = np.array([
        [0.22859485344235864, 0.9192240341080489],
        [0.08698212281811202, 0.834857219760308],
        [0.9167576448882734, 0.9701323128191051],
    ])
    rows = [tuple(row) for row in v]
    b = randomsubset(rows, 3)
    assert len(b) == 3
    assert set(b).issubset(set(rows))
