import numpy as np
import numba
from numba import jit

# precision names -> numpy dtypes, for the real and complex code paths
_REAL_DTYPES = {"single": np.float32, "double": np.float64}
_COMPLEX_DTYPES = {"single": np.complex64, "double": np.complex128}


def kpm_moments_v(v,m,n=100,kpm_prec="double",
        kpm_cpugpu="CPU",**kwargs):
    """Return the local moments.

    kpm_prec selects the floating point precision ("single" or "double"),
    and kpm_cpugpu selects the backend: "CPU" (numba) or "GPU" (JAX, which
    falls back to running on the CPU if no GPU is available). Real input
    (real matrix and real starting vector) is detected automatically and
    computed with real arithmetic; otherwise complex arithmetic is used.
    Both code paths support single and double precision."""
    from scipy.sparse import coo_matrix
    mo = coo_matrix(m)
    v = np.asarray(v)
    # a literally all-zero matrix (e.g. a decoupled orbital subspace, or an
    # electron-only sector left with no hopping after some projection) has
    # an empty mo.data -- there is no matrix-side imaginary part to check,
    # so realness then depends only on the starting vector v
    if mo.data.size == 0:
        is_real = np.max(np.abs(v.imag))<1e-6
    else:
        is_real = np.max(np.abs(mo.data.imag))<1e-6 and np.max(np.abs(v.imag))<1e-6
    if is_real:
        dtype = _REAL_DTYPES[kpm_prec]
        v = np.array(v.real,dtype=dtype) # convert to float
        data = np.array(mo.data.real,dtype=dtype) # convert to float
    else:
        dtype = _COMPLEX_DTYPES[kpm_prec]
        v = np.array(v,dtype=dtype)
        data = np.array(mo.data,dtype=dtype)
    if kpm_cpugpu=="CPU": # use the CPU
        if is_real: mus = python_kpm_moments_real(v,data,mo.row,mo.col,n=n)
        else: mus = python_kpm_moments_complex(v,data,mo.row,mo.col,n=n)
    elif kpm_cpugpu=="GPU": # use the GPU (or CPU, if no GPU is available)
        from .kpmjax import kpm_moments_gpu
        mus = kpm_moments_gpu(v,data,mo.row,mo.col,n=n)
    else: raise ValueError("kpm_cpugpu must be 'CPU' or 'GPU', got "+str(kpm_cpugpu))
    return np.array(mus,dtype=np.complex128)


@jit(nopython=True)
def python_kpm_moments_complex(v,data,row,col,n=100):
    """Python routine to calculate moments.

    Each Chebyshev iteration used to build ap=2*H@a-am and the two moment
    inner products <a|a>,<ap|a> out of separate numpy expressions (2*Mtimesv(),
    then -am, then conj(a)*a, then conj(ap)*a), each allocating its own
    nsites-length temporary and re-.copy()ing am/a into fresh arrays. Below,
    the sparse matvec is scattered directly into ap (initialized to -am
    first) and both inner products are accumulated in the same dense pass
    that finishes computing ap, so each iteration only pays for one
    O(nnz) sparse pass and one O(nsites) dense pass; am/a/ap are three
    fixed buffers rotated between iterations (no copy, no reallocation)."""
    nsites = len(v)
    nnz = len(data)
    mus = np.zeros(2*n,dtype=v.dtype) # empty array for the moments
    am = v.copy() # zero vector
    a = Mtimesv(data,row,col,v) #m@v  # vector number 1
    bk = np.sum(np.conjugate(v)*v)
    bk1 = np.sum(np.conjugate(a)*v) #algebra.braket_ww(a,v)

    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    ap = np.empty_like(v)
    for i in range(1,n):
        for s in range(nsites): ap[s] = -am[s]
        for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
        bk = a[0]-a[0]   # zero of a's dtype
        bk1 = a[0]-a[0]
        for s in range(nsites):
            bk += np.conjugate(a[s])*a[s] # algebra.braket_ww(a,a)
            bk1 += np.conjugate(ap[s])*a[s] # algebra.braket_ww(ap,a)
        mus[2*i] = 2.*bk
        mus[2*i+1] = 2.*bk1
        am,a,ap = a,ap,am # rotate the three buffers, no allocation
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
    """Python routine to calculate moments. See python_kpm_moments_complex
    for why the per-iteration work is fused into one O(nnz) sparse scatter
    into ap (initialized to -am) plus one O(nsites) dense reduction pass,
    instead of a chain of separate numpy temporaries."""
    nsites = len(v)
    nnz = len(data)
    mus = np.zeros(2*n,dtype=v.dtype) # empty array for the moments
    am = v.copy() # zero vector
    a = Mtimesv(data,row,col,v) #m@v  # vector number 1
    bk = np.sum(v*v)
    bk1 = np.sum(a*v) #algebra.braket_ww(a,v)

    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    ap = np.empty_like(v)
    for i in range(1,n):
        for s in range(nsites): ap[s] = -am[s]
        for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
        bk = a[0]-a[0]   # zero of a's dtype
        bk1 = a[0]-a[0]
        for s in range(nsites):
            bk += a[s]*a[s] # algebra.braket_ww(a,a)
            bk1 += ap[s]*a[s] # algebra.braket_ww(ap,a)
        mus[2*i] = 2.*bk
        mus[2*i+1] = 2.*bk1
        am,a,ap = a,ap,am # rotate the three buffers, no allocation
    mu0 = mus[0] # first
    mu1 = mus[1] # second
    for i in range(1,n):
      mus[2*i] +=  - mu0
      mus[2*i+1] += -mu1
    return mus





def kpm_moments_vivj(m,vi,vj,n=100,**kwargs):
    """Return the local moments"""
    from scipy.sparse import coo_matrix
    mo = coo_matrix(m)
    data = np.array(mo.data,dtype=np.complex128)
    vi = np.array(vi,dtype=np.complex128)
    vj = np.array(vj,dtype=np.complex128)
    mus = numba_kpm_moments_ij(vi,vj,data,mo.row,mo.col,n=2*n)
    return mus




def kpm_moments_ij(m0,i=0,j=0,**kwargs):
    """Return the KPM moments between sites i and j"""
    n = m0.shape[0] # size of the matrix
    from .ldos import index2vector
    vi = index2vector(i,n) # generate vector
    vj = index2vector(j,n) # generate vector
    return kpm_moments_vivj(m0,vi,vj,**kwargs) # return moments





@jit(nopython=True)
def numba_kpm_moments_ij(vi,vj,data,row,col,n=100):
  """ Get the first n moments of a the |vi><vj| operator
  using the Chebychev recursion relations. See
  python_kpm_moments_complex for why each iteration scatters the sparse
  matvec directly into ap (initialized to -am) and accumulates <vj|ap> in
  the same dense pass, instead of separate numpy temporaries."""
  nsites = len(vi)
  nnz = len(data)
  mus = np.zeros(n,dtype=np.complex128) # empty array for the moments
  am = vi.copy() # am must be a private copy: it gets overwritten below
  a = Mtimesv(data,row,col,vi)
  bk = np.sum(np.conjugate(vj)*vi) # scalar
  bk1 = np.sum(np.conjugate(vj)*a)
#  bk1 = (vj.H*a).todense().trace()[0,0] # calculate bk
  mus[0] = bk  # mu0
  mus[1] = bk1 # mu1
  ap = np.empty_like(am)
  for ii in range(2,n):
    for s in range(nsites): ap[s] = -am[s]
    for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
    bk = a[0]-a[0]  # zero of a's dtype
    for s in range(nsites):
        bk += np.conjugate(vj[s])*ap[s]
    mus[ii] = bk
    am,a,ap = a,ap,am # rotate the three buffers, no allocation
  return mus




