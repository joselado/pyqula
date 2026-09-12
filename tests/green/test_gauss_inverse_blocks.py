import numpy as np
import pytest

from pyqula.algebratk.gaussinv import gauss_inverse
from pyqula.green import block_inverse

# gauss_inverse is the recursive (Gauss/Dyson sweep) block inversion used by
# transporttk/landauer.py and transporttk/smatrix.py; block_inverse is the
# brute-force second implementation in green.py, selected by
# green.mode_block_inverse="full". They compute the same object -- one block
# of the inverse of a block-tridiagonal matrix -- through unrelated
# algorithms, so block_inverse is the oracle here and no reference number is
# needed. Both landauer.py and smatrix.py carry the comment "blocks can have
# different sizes", and transporttk/central.py builds exactly such junctions
# (a central region bigger than the lead cell), so the non-uniform case is
# part of the contract and not an exotic input.


def block_tridiagonal(sizes, seed):
    """Random block-tridiagonal matrix with the given block sizes. The
    couplings are independent random complex matrices, so the matrix is
    neither Hermitian nor block-symmetric -- m[i][i+1] is unrelated to
    m[i+1][i]^dagger. That matters: a bug that confuses a block with the
    dagger of its partner is invisible on a Hermitian fixture."""
    rng = np.random.default_rng(seed)
    def rnd(n, m):
        return rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))
    nb = len(sizes)
    mat = [[None for _ in range(nb)] for _ in range(nb)]
    for i in range(nb):
        # a diagonal shift keeps the matrix comfortably invertible
        mat[i][i] = rnd(sizes[i], sizes[i]) + 4.*np.identity(sizes[i])
    for i in range(nb-1):
        mat[i][i+1] = rnd(sizes[i], sizes[i+1])
        mat[i+1][i] = rnd(sizes[i+1], sizes[i])
    return mat


@pytest.mark.parametrize("sizes", [[2, 2, 2], [2, 3, 2], [3, 1, 2, 4],
                                   [1, 3, 1], [3]])
@pytest.mark.parametrize("seed", [0, 4])
def test_gauss_inverse_matches_brute_force_for_every_block(sizes, seed):
    """Every (i,j) block of the inverse must agree with the brute-force
    inversion, whatever the block sizes are. The uniform-size case pins
    that nothing regressed; the non-uniform ones are the contract that
    landauer/smatrix advertise and that central.py builds."""
    mat = block_tridiagonal(sizes, seed)
    nb = len(sizes)
    for i in range(nb):
        for j in range(nb):
            got = np.array(gauss_inverse(mat, i=i, j=j))
            ref = np.array(block_inverse(mat, i=i, j=j))
            assert got.shape == (sizes[i], sizes[j])
            assert np.max(np.abs(got-ref)) < 1e-9*max(1., np.max(np.abs(ref)))


def test_gauss_inverse_accepts_negative_block_indices():
    """smatrix.py asks for the (0,-1) and (-1,0) corner blocks by the
    python negative-index convention, so those must resolve to the same
    answer as the positive spelling."""
    sizes = [2, 3, 4]
    mat = block_tridiagonal(sizes, seed=7)
    for (i, j), (i2, j2) in [((0, -1), (0, 2)), ((-1, 0), (2, 0)),
                             ((-1, -1), (2, 2))]:
        a = np.array(gauss_inverse(mat, i=i, j=j))
        b = np.array(gauss_inverse(mat, i=i2, j=j2))
        assert np.max(np.abs(a-b)) < 1e-12
