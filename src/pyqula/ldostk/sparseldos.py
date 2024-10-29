
import numpy as np

import scipy.sparse.linalg as slg 
import scipy.sparse as sp

from ..kpmtk.ldos import index2vector

def get_ldos(A,i=0,e=0.,delta=1e-3,**kwargs):
    """Return the LDOS in site i using sparse inversion"""
    # this is so far just implemented for real matrices
    ##################################
    b = index2vector(i,A.shape[0]) # create vector
    size = 1000  # Size of the identity matrix
    I = sp.identity(A.shape[0], dtype=np.complex128) # complex identity
    H = A + (1j*delta-e)*I # Green function to solve
    v = slg.spsolve(H,b) # correction vector
    out = np.conjugate(v).dot(b).imag # value of the output
    return out/np.pi # return LDOS



