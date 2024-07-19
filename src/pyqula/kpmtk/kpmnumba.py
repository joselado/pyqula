import numpy as np
import numba
from numba import jit


def kpm_moments(v,m,n=100,kpm_prec="double",
        kpm_cpugpu="CPU",**kwargs):
    """Return the local moments"""
    from scipy.sparse import coo_matrix
    mo = coo_matrix(m)
    data = np.array(mo.data,dtype=np.complex_)
    if np.max(np.abs(data.imag))<1e-6 and np.max(np.abs(v.imag))<1e-6: # real
        if kpm_prec == "single": dtype = np.float32
        elif kpm_prec == "double": dtype = np.float_
        v = np.array(v.real,dtype=dtype) # convert to float
        data = np.array(data.real,dtype=dtype) # convert to float
        if kpm_cpugpu=="CPU": # use the CPU
            mus = python_kpm_moments_real(v,data,mo.row,mo.col,n=n)
        elif kpm_cpugpu=="GPU": # use the GPU
            from .kpmjax import kpm_moments_real_gpu
            mus = kpm_moments_real_gpu(v,data,mo.row,mo.col,n=n)
    else:
        mus = python_kpm_moments_complex(v,data,mo.row,mo.col,n=n)
    return np.array(mus,dtype=np.complex_)


@jit(nopython=True)
def python_kpm_moments_complex(v,data,row,col,n=100):
    """Python routine to calculate moments"""
    mus = np.zeros(2*n,dtype=np.complex_) # empty array for the moments
    am = v.copy() # zero vector
    a = Mtimesv(data,row,col,v) #m@v  # vector number 1
    bk = np.sum(np.conjugate(v)*v)
    bk1 = np.sum(np.conjugate(a)*v) #algebra.braket_ww(a,v)

    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    for i in range(1,n):
        ap = 2*Mtimesv(data,row,col,a) - am # recursion relation
#        ap = 2*m@a - am # recursion relation
        bk = np.sum(np.conjugate(a)*a) # algebra.braket_ww(a,a)
        bk1 = np.sum(np.conjugate(ap)*a) # algebra.braket_ww(ap,a)
        mus[2*i] = 2.*bk
        mus[2*i+1] = 2.*bk1
        am = a.copy() # new variables
        a = ap.copy() # new variables
    mu0 = mus[0] # first
    mu1 = mus[1] # second
    for i in range(1,n):
      mus[2*i] +=  - mu0
      mus[2*i+1] += -mu1
    return mus



@jit(nopython=True)
def Mtimesv(data,row,col,v):
    """Matrix times vector"""
    out = np.zeros_like(v) # initilize
    n = len(data) # number of terms
    for i in range(n): # loop over terms
        ii = row[i]
        jj = col[i]
        out[ii] = out[ii] + data[i]*v[jj]
    return out




@jit(nopython=True)
def python_kpm_moments_real(v,data,row,col,n=100):
    """Python routine to calculate moments"""
    mus = np.zeros(2*n,dtype=v.dtype) # empty array for the moments
    am = v.copy() # zero vector
    a = Mtimesv(data,row,col,v) #m@v  # vector number 1
    bk = np.sum(v*v)
    bk1 = np.sum(a*v) #algebra.braket_ww(a,v)

    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    for i in range(1,n):
        ap = 2*Mtimesv(data,row,col,a) - am # recursion relation
#        ap = 2*m@a - am # recursion relation
        bk = np.sum(a*a) # algebra.braket_ww(a,a)
        bk1 = np.sum(ap*a) # algebra.braket_ww(ap,a)
        mus[2*i] = 2.*bk
        mus[2*i+1] = 2.*bk1
#        am = a.copy() # new variables
#        a = ap.copy() # new variables
        am,a = a,ap # reassign
    mu0 = mus[0] # first
    mu1 = mus[1] # second
    for i in range(1,n):
      mus[2*i] +=  - mu0
      mus[2*i+1] += -mu1
    return mus




