import numpy as np
from scipy.sparse import random as sparse_random, csc_matrix

from pyqula import superconductivity
from pyqula.sctk import reorder


def _random_matrix(n, seed):
    rng = np.random.default_rng(seed)
    m = sparse_random(n, n, density=0.2, random_state=rng,
            dtype=np.complex128)
    return csc_matrix(m + 1j*sparse_random(n, n, density=0.2,
            random_state=rng))


def test_time_reversal_is_sigma_y_conjugation_on_every_site():
    """time_reversal builds 1 x sigma_y as a Kronecker product, linear in the
    number of sites, where it used to assemble an n x n list of blocks; it
    has to stay sigma_y m^* sigma_y on every spin block, for a sparse and a
    dense input alike"""
    sy = np.array([[0., -1j], [1j, 0.]])
    for n in [1, 3, 7]:
        m = _random_matrix(2*n, n)
        S = np.kron(np.identity(n), sy)
        expected = S @ np.conjugate(m.toarray()) @ S
        out = superconductivity.time_reversal(m)
        assert np.max(np.abs(out.toarray() - expected)) < 1e-14
        out = superconductivity.time_reversal(m.toarray())
        assert np.max(np.abs(np.asarray(out) - expected)) < 1e-14


def test_nambu_reorder_sparse_matches_dense():
    """The sparse Nambu reordering is now built from index arrays rather than
    by growing Python lists, which was quadratic; it is the same
    permutation as the dense construction, site by site"""
    for nr in [1, 2, 5, 11]:
        m = np.zeros((4*nr, 4*nr))
        a = reorder.block2nambu_matrix_sparse(m).toarray()
        b = reorder.block2nambu_matrix_dense(m).toarray()
        assert np.max(np.abs(a - b)) == 0.
