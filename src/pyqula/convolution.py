import numpy as np
from numba import jit

def selfconvolve(ds):
    """Do a selfconvolution"""
    if len(ds.shape)!=2: raise
    return selfconvolve_jit(ds,ds*0.0)


@jit(nopython=True)
def selfconvolve_jit(ds,out):
    """Convolve a 2D array with itself using periodic boundary conditions"""
    nx = ds.shape[0]
    ny = ds.shape[1]
    for i in range(nx):
      for j in range(ny):
        for ii in range(nx):
          for jj in range(ny):
              out[ii,jj] = ds[(i+ii)%nx,(j+jj)%nx]*ds[i,j] + out[ii,jj]
    return out


