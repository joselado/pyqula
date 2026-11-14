import numpy as np
from .. import algebra


def get_sharpen(delta=None):
    """Return a function that renormalizes the eigenvalues of a Hermitian
    matrix to +-1 (a smoothed sign function), used to turn a smooth mass
    operator (e.g. valley tau_z/tau_x/tau_y) into a clean +-1 projector.

    This needs every eigenvalue, so a sparse matrix is diagonalized dense
    (algebra.eigh refuses a matrix above algebra.maxsize) and the result,
    which has no sparsity left, is returned in the sparse storage of the
    input. A sparse matrix used to come back untouched, so delta was
    dropped without a word on a sparse Hamiltonian."""
    def sharpen(m):
        if delta is None: return m # do nothing
        (es,vs) = algebra.eigh(m) # diagonalize, dense
        es = es/(np.abs(es)+delta) # renormalize the eigenvalues
        vs = np.array(vs) # convert
        out = (vs*es[None,:])@vs.conj().T # the renormalized operator
        if algebra.issparse(m): # keep the storage of the input
            from scipy.sparse import csc_matrix
            return csc_matrix(out)
        return out
    return sharpen
