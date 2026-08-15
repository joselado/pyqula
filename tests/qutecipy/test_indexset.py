from pyqula.qutecipytk.indexset import IndexSet, isnested


def test_indexset_basic():
    is_ = IndexSet()
    assert is_.toint == {}
    assert is_.fromint == []
    assert len(is_) == 0
    assert not is_
    assert is_ == IndexSet()

    L = [
        (6, 0, 9, 1, 0), (8, 7, 4, 7, 6), (1, 8, 4, 3, 0), (3, 7, 1, 6, 8),
        (7, 7, 0, 6, 0), (8, 3, 6, 0, 9), (1, 1, 4, 7, 0), (9, 6, 9, 9, 5),
        (1, 8, 5, 9, 9), (6, 3, 6, 4, 6),
    ]

    for i, item in enumerate(L):
        is_.append(item)
        assert is_[i] == item
        assert is_.toint[item] == i
        assert is_.fromint[i] == item

    assert len(is_) == len(L)
    assert bool(is_)
    assert is_ == IndexSet(L)


def test_indexset_nested():
    is1 = [(0,), (1,)]
    is2 = [(0, 3), (1, 2)]
    assert isnested(is1, is2)

    is3 = [(3, 0), (2, 1)]
    assert isnested(is1, is3, "col")
