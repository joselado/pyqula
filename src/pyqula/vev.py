# compute vacuum expectation values

import numpy as np
from .operators import Operator

def get_dm_vev(H,A,**kwargs):
    """Compute a vacuum expectation value of two operators"""
    if H.dimensionality != 0:
        raise ValueError("the density-matrix expectation value is only "
                "implemented for 0d Hamiltonians")
    # every keyword (the temperature above all) used to be dropped here,
    # so the vev was the zero-temperature one whatever was asked for
    dm = H.get_density_matrix(**kwargs) # return the DM, as a matrix
    A = Operator(A) # convert to operator
    # transposed for the same reason as in spectrum.ev: full_dm's
    # convention is the transpose of the usual density matrix, so
    # contracting it directly gives <A*> instead of <A>
    return np.trace(A@np.transpose(dm)) # return the expectation value

