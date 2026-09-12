import numpy as np
from scipy.sparse import csc_matrix

import pyqula.superconductivity  # noqa: F401
from pyqula.sctk import extract


def _random_nambu(nr, seed):
    """Random matrix with the shape of a spinful Nambu block, 4 components
    (spin x electron-hole) per site"""
    rng = np.random.default_rng(seed)
    n = 4*nr
    return rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))


# reference implementations: the index arithmetic spelled out one element at
# a time, which is what the kernels used to do. They are the oracle for the
# strided form, which must reproduce them bit for bit.

def _ref_pairing(m):
    nr = m.shape[0]//4
    uu = np.zeros((nr, nr), dtype=np.complex128)
    dd = np.zeros((nr, nr), dtype=np.complex128)
    ud = np.zeros((nr, nr), dtype=np.complex128)
    for i in range(nr):
        for j in range(nr):
            ud[i, j] = m[4*i, 4*j+2]
            dd[i, j] = m[4*i+1, 4*j+2]
            uu[i, j] = m[4*i, 4*j+3]
    return (uu, dd, ud)


def _ref_triplet(m):
    nr = m.shape[0]//4
    uu = np.zeros((nr, nr), dtype=np.complex128)
    dd = np.zeros((nr, nr), dtype=np.complex128)
    ud = np.zeros((nr, nr), dtype=np.complex128)
    for i in range(nr):
        for j in range(nr):
            ud[i, j] = (m[4*i, 4*j+2] - np.conjugate(m[4*j+3, 4*i+1]))/2.
            dd[i, j] = m[4*i+1, 4*j+2]
            uu[i, j] = m[4*i, 4*j+3]
    return (uu, dd, ud)


def _ref_singlet(m):
    nr = m.shape[0]//4
    ud = np.zeros((nr, nr), dtype=np.complex128)
    for i in range(nr):
        for j in range(nr):
            ud[i, j] = (m[4*i, 4*j+2] + np.conjugate(m[4*j+3, 4*i+1]))/2.
    return ud


def _ref_singlet_dict(dd):
    out = dict()
    for key in dd:
        d = dd[key]
        nr = d.shape[0]//4
        m = np.zeros(d.shape, dtype=np.complex128)
        m0 = dd[key]
        m1 = dd[(-key[0], -key[1], -key[2])]
        for i in range(nr):
            for j in range(nr):
                m[4*i, 4*j+2] = (m0[4*i, 4*j+2]
                                 + np.conjugate(m1[4*j+3, 4*i+1]))/2.
                m[4*j+3, 4*i+1] = (m0[4*j+3, 4*i+1]
                                   + np.conjugate(m1[4*i, 4*j+2]))/2.
                m[4*i+2, 4*j] = (m0[4*i+2, 4*j]
                                 + np.conjugate(m1[4*j+1, 4*i+3]))/2.
                m[4*j+1, 4*i+3] = (m0[4*j+1, 4*i+3]
                                   + np.conjugate(m1[4*i+2, 4*j]))/2.
        out[key] = m
    return out


def test_pairing_kernels_reproduce_the_element_by_element_reference():
    """The four pairing-extraction kernels are strided slices of the Nambu
    matrix. Vectorizing them must not move a single bit, so the test asks
    for exact equality with the explicit index arithmetic, not a tolerance."""
    for nr in [1, 3, 7]:
        m = _random_nambu(nr, seed=nr)
        for (got, ref) in [(extract.extract_pairing(m), _ref_pairing(m)),
                           (extract.extract_triplet_pairing(m),
                            _ref_triplet(m))]:
            for (a, b) in zip(got, ref):
                assert np.array_equal(np.array(a), b)
        assert np.array_equal(np.array(extract.extract_singlet_pairing(m)),
                              _ref_singlet(m))


def test_pairing_kernels_accept_a_sparse_matrix():
    """The stored hoppings of a sparse Hamiltonian are scipy sparse matrices,
    and dict2absdeltas feeds them straight into extract_pairing."""
    m = _random_nambu(4, seed=11)
    ms = csc_matrix(m)
    for (got, ref) in [(extract.extract_pairing(ms), _ref_pairing(m)),
                       (extract.extract_triplet_pairing(ms), _ref_triplet(m)),
                       ((extract.extract_singlet_pairing(ms),),
                        (_ref_singlet(m),))]:
        for (a, b) in zip(got, ref):
            assert np.array_equal(np.array(a), b)


def test_extract_singlet_dict_reproduces_the_element_by_element_reference():
    """Same for the dictionary version, which symmetrizes a hopping against
    the one in the opposite direction."""
    keys = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]
    dd = {k: _random_nambu(3, seed=hash(k) % 1000) for k in keys}
    got = extract.extract_singlet_dict(dd)
    ref = _ref_singlet_dict(dd)
    assert set(got) == set(ref)
    for key in ref:
        assert np.array_equal(np.array(got[key]), ref[key]), key
