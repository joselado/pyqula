from numba import jit
import numpy as np


def crop_matrix(m,store):
    """Retain just some entries of a matrix"""
    if len(store)!=m.shape[0]: raise
    store = np.array(store,dtype=int) # transform to an integer
    n = np.sum(store) # number of entries of the matrix
    mout = np.zeros((n,n),dtype=np.complex)
    out = crop_matrix_jit(m,store,mout)
#    print(np.sum(np.abs(out))) 
    return out

@jit(nopython=True)
def crop_matrix_jit(m,store,mout):
    """Just retain the right elements"""
    ii = 0
    jj = 0
    n0 = m.shape[0] # input dimension
    n = mout.shape[0] # output dimension
    ii = 0
    for i in range(n0):
        if not store[i]==1: continue # skip iteration
        jj = 0 # initialize
        for j in range(n0):
            if store[j]==1: # store this one
                mout[ii,jj] = m[i,j]
                jj += 1
        ii += 1
    return mout





