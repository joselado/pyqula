import numpy as np
from .. import algebra


def get_sharpen(delta=None):
    """Return a function that renormalizes the eigenvalues of a Hermitian
    matrix to +-1 (a smoothed sign function), used to turn a smooth mass
    operator (e.g. valley tau_z/tau_x/tau_y) into a clean +-1 projector"""
    def sharpen(m):
        if delta is None: return m # do nothing
        if algebra.issparse(m): return m # temporal workaround
        (es,vs) = algebra.eigh(m) # diagonalize
        es = es/(np.abs(es)+delta) # renormalize the eigenvalues
        vs = np.matrix(vs) # convert
        m0 = np.matrix(np.diag(es)) # build new hamiltonian
        return vs@m0@vs.H # return renormalized operator
    return sharpen
