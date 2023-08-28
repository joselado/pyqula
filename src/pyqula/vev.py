# compute vacuum expectation values

import numpy as np
from operators import Operator

def get_dm_vev(H,A,**kwargs):
    """Compute a vacuum expectation value of two operators"""
    if H.dimensionality != 0:
        print("Only implemented for 0d Hamiltonians")
        raise
    dm = H.get_density_matrix() # return the DM, as a matrix
    A = Operator(A) # convert to operator
    return np.trace(A@dm) # return the expectation value

