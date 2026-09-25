import numpy as np

from pyqula.multihopping import MultiHopping

# The dagger of the block at R lives at -R. get_dagger used to look up the
# partner at -R and drop the block when it was absent, so a MultiHopping
# with a lone (1,0,0) block had an empty dagger.


def _lone_block():
    m = np.random.random((3, 3)) + 1j*np.random.random((3, 3))
    return m, MultiHopping({(1, 0, 0): m})


def test_a_lone_block_has_its_dagger_at_minus_R():
    m, mh = _lone_block()
    d = mh.get_dagger().get_dict()
    assert set(d) == {(-1, 0, 0)}
    assert np.max(np.abs(d[(-1, 0, 0)] - np.conjugate(m.T))) < 1e-14


def test_adding_the_dagger_gives_a_hermitian_multihopping():
    m, mh = _lone_block()
    assert not mh.is_hermitian()
    assert (mh + mh.get_dagger()).is_hermitian()


def test_the_dagger_of_the_dagger_is_the_original():
    m = np.random.random((2, 2)) + 1j*np.random.random((2, 2))
    onsite = np.random.random((2, 2))
    mh = MultiHopping({(0, 0, 0): onsite, (0, 1, 0): m})
    back = mh.get_dagger().get_dagger().get_dict()
    assert set(back) == {(0, 0, 0), (0, 1, 0)}
    assert np.max(np.abs(back[(0, 1, 0)] - m)) < 1e-14
    assert np.max(np.abs(back[(0, 0, 0)] - onsite)) < 1e-14
