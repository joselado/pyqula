import numpy as np
from scipy.sparse import coo_matrix
from numba import jit

@jit(nopython=True)
def sum_column(data,row,col,out):
    for i in range(len(data)): # loop over data
        r,c = row[i],col[i]
        out[r] = out[r] + data[i]
    return out

def estimate_bandwidth(H):
    """Return an upper bound to the bandwidth of a matrix"""
    mi = coo_matrix(H)
    row = mi.row
    col = mi.col
    data = np.abs(mi.data) # absolute value
    out = np.zeros(mi.shape[0]) # output
    out = np.max(sum_column(data,row,col,out))
    return out







