import numpy as np
from numba import jit

def selfconvolve(ds):
    """Do a selfconvolution"""
    if len(ds.shape)!=2: raise
    return selfconvolve_fft(ds)


def selfconvolve_fft(ds):
    """Convolve a 2D array with itself using periodic boundary conditions,
    out[q] = sum_k ds[k]*ds[k+q], computed exactly (up to float roundoff)
    via FFT instead of the O(nx^2*ny^2) brute-force double loop in
    selfconvolve_jit below. By the Wiener-Khinchin theorem this
    autocorrelation equals real(ifft2(fft2(ds)*conj(fft2(ds)))); the usual
    convolution-vs-correlation direction-of-shift ambiguity of that
    identity doesn't matter here because out is always even in q
    (relabeling k'=k+q in the defining sum maps q -> -q back onto the same
    sum, so out[q]==out[-q] regardless of ds), so no extra index flip is
    needed to match selfconvolve_jit's convention."""
    F = np.fft.fft2(ds)
    return np.real(np.fft.ifft2(F*np.conj(F)))


@jit(nopython=True)
def selfconvolve_jit(ds,out):
    """Convolve a 2D array with itself using periodic boundary conditions.
    O(nx^2*ny^2) reference implementation, kept for testing against
    selfconvolve_fft (see tests/fermisurface/test_qpi_fft_convolution.py)."""
    nx = ds.shape[0]
    ny = ds.shape[1]
    for i in range(nx):
      for j in range(ny):
        for ii in range(nx):
          for jj in range(ny):
              out[ii,jj] = ds[(i+ii)%nx,(j+jj)%nx]*ds[i,j] + out[ii,jj]
    return out

from .temperaturetk.convolution import temperature_convolution





