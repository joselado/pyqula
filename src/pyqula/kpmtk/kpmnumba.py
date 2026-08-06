import numpy as np
import numba
from numba import jit,prange

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


def kpm_moments_batch(vs,m,n=100,kpm_prec="double",
        kpm_cpugpu="CPU",**kwargs):
    """Return the moments for a batch of starting vectors against the same
    matrix, one vector per numba thread (see python_kpm_moments_batch_complex).
    vs has shape (nvec,nsites); returns an (nvec,2n) array of moments. This
    is the batched counterpart of kpm_moments_v, for the common case of
    many independent vectors (random-trace tries, or one vector per site)
    sharing the same matrix -- see that function for the precision/realness
    handling this mirrors."""
    from scipy.sparse import coo_matrix
    mo = coo_matrix(m)
    vs = np.asarray(vs)
    if mo.data.size == 0:
        is_real = np.max(np.abs(vs.imag))<1e-6
    else:
        is_real = np.max(np.abs(mo.data.imag))<1e-6 and np.max(np.abs(vs.imag))<1e-6
    if is_real:
        dtype = _REAL_DTYPES[kpm_prec]
        vs = np.array(vs.real,dtype=dtype) # convert to float
        data = np.array(mo.data.real,dtype=dtype) # convert to float
    else:
        dtype = _COMPLEX_DTYPES[kpm_prec]
        vs = np.array(vs,dtype=dtype)
        data = np.array(mo.data,dtype=dtype)
    if kpm_cpugpu=="CPU": # use the CPU, one vector per thread
        if is_real: mus = python_kpm_moments_batch_real(vs,data,mo.row,mo.col,n=n)
        else: mus = python_kpm_moments_batch_complex(vs,data,mo.row,mo.col,n=n)
    elif kpm_cpugpu=="GPU": # no batched GPU path, loop over the single-vector one
        from .kpmjax import kpm_moments_gpu
        mus = np.array([kpm_moments_gpu(vs[iv],data,mo.row,mo.col,n=n)
                for iv in range(vs.shape[0])])
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



@jit(nopython=True,parallel=True,cache=True)
def python_kpm_moments_batch_complex(vs,data,row,col,n=100):
    """Chebyshev moments for a batch of vectors sharing the same sparse
    matrix, one vector per numba thread. vs has shape (nvec,nsites); each
    thread only ever writes its own output row of mus, mirroring the
    prange-batch pattern used elsewhere in the package (e.g.
    dmtk/fulldm.py's full_dm_batch_vectorized). The per-vector recursion
    body is otherwise identical to python_kpm_moments_complex, which stays
    the (unbatched) entry point used when there is only one vector."""
    nvec = vs.shape[0]
    nsites = vs.shape[1]
    nnz = len(data)
    mus = np.zeros((nvec,2*n),dtype=vs.dtype)
    for iv in prange(nvec):
        v = vs[iv]
        am = v.copy() # zero vector
        a = Mtimesv(data,row,col,v) #m@v  # vector number 1
        bk = np.sum(np.conjugate(v)*v)
        bk1 = np.sum(np.conjugate(a)*v)
        mus[iv,0] = bk  # mu0
        mus[iv,1] = bk1 # mu1
        ap = np.empty_like(v)
        for i in range(1,n):
            for s in range(nsites): ap[s] = -am[s]
            for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
            bk = a[0]-a[0]   # zero of a's dtype
            bk1 = a[0]-a[0]
            for s in range(nsites):
                bk += np.conjugate(a[s])*a[s]
                bk1 += np.conjugate(ap[s])*a[s]
            mus[iv,2*i] = 2.*bk
            mus[iv,2*i+1] = 2.*bk1
            am,a,ap = a,ap,am # rotate the three buffers, no allocation
        mu0 = mus[iv,0] # first
        mu1 = mus[iv,1] # second
        for i in range(1,n):
            mus[iv,2*i] += -mu0
            mus[iv,2*i+1] += -mu1
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



@jit(nopython=True,parallel=True,cache=True)
def python_kpm_moments_batch_real(vs,data,row,col,n=100):
    """Real-arithmetic counterpart of python_kpm_moments_batch_complex --
    see that function for the prange-over-vectors batching scheme."""
    nvec = vs.shape[0]
    nsites = vs.shape[1]
    nnz = len(data)
    mus = np.zeros((nvec,2*n),dtype=vs.dtype)
    for iv in prange(nvec):
        v = vs[iv]
        am = v.copy() # zero vector
        a = Mtimesv(data,row,col,v) #m@v  # vector number 1
        bk = np.sum(v*v)
        bk1 = np.sum(a*v)
        mus[iv,0] = bk  # mu0
        mus[iv,1] = bk1 # mu1
        ap = np.empty_like(v)
        for i in range(1,n):
            for s in range(nsites): ap[s] = -am[s]
            for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
            bk = a[0]-a[0]   # zero of a's dtype
            bk1 = a[0]-a[0]
            for s in range(nsites):
                bk += a[s]*a[s]
                bk1 += ap[s]*a[s]
            mus[iv,2*i] = 2.*bk
            mus[iv,2*i+1] = 2.*bk1
            am,a,ap = a,ap,am # rotate the three buffers, no allocation
        mu0 = mus[iv,0] # first
        mu1 = mus[iv,1] # second
        for i in range(1,n):
            mus[iv,2*i] += -mu0
            mus[iv,2*i+1] += -mu1
    return mus




def kpm_moments_A_batch(vs,m,A,n=100,kpm_prec="double",
        kpm_cpugpu="CPU",**kwargs):
    """Return the operator-weighted moments mus[k,i] = <T_i(m) v_k|A|v_k>
    for a batch of starting vectors sharing the same matrix m and operator
    A, one vector per numba thread. vs has shape (nvec,nsites); returns an
    (nvec,n) array of moments. A@v_k depends only on v_k, not on the
    Chebyshev iterate T_i(m) v_k, so it is computed once per vector via a
    single sparse-dense matmul up front instead of being recomputed inside
    the O(n) recursion loop (the prior per-vector implementation in
    kpm.get_momentsA_jit recomputed A@v on every iteration)."""
    from scipy.sparse import coo_matrix, csc_matrix
    mo = coo_matrix(m)
    vs = np.asarray(vs)
    Ac = csc_matrix(A)
    Avs = np.asarray(Ac@vs.T).T # (nvec,nsites), A@v for each starting vector
    if mo.data.size == 0:
        is_real = np.max(np.abs(vs.imag))<1e-6 and np.max(np.abs(Avs.imag))<1e-6
    else:
        is_real = (np.max(np.abs(mo.data.imag))<1e-6 and np.max(np.abs(vs.imag))<1e-6
                and np.max(np.abs(Avs.imag))<1e-6)
    if is_real:
        dtype = _REAL_DTYPES[kpm_prec]
        vs = np.array(vs.real,dtype=dtype)
        Avs = np.array(Avs.real,dtype=dtype)
        data = np.array(mo.data.real,dtype=dtype)
    else:
        dtype = _COMPLEX_DTYPES[kpm_prec]
        vs = np.array(vs,dtype=dtype)
        Avs = np.array(Avs,dtype=dtype)
        data = np.array(mo.data,dtype=dtype)
    if kpm_cpugpu=="CPU":
        if is_real: mus = python_kpm_momentsA_batch_real(vs,Avs,data,mo.row,mo.col,n=n)
        else: mus = python_kpm_momentsA_batch_complex(vs,Avs,data,mo.row,mo.col,n=n)
    else: raise ValueError("kpm_cpugpu must be 'CPU', got "+str(kpm_cpugpu))
    return np.array(mus,dtype=np.complex128)


@jit(nopython=True,parallel=True,cache=True)
def python_kpm_momentsA_batch_complex(vs,Avs,data,row,col,n=100):
    """See kpm_moments_A_batch. Av is precomputed per vector and reused
    across all n iterations of the Chebyshev recursion."""
    nvec = vs.shape[0]
    nsites = vs.shape[1]
    nnz = len(data)
    mus = np.zeros((nvec,n),dtype=vs.dtype)
    for iv in prange(nvec):
        v = vs[iv]
        Av = Avs[iv]
        am = v.copy()
        a = Mtimesv(data,row,col,v)
        bk = np.sum(np.conjugate(v)*Av)
        bk1 = np.sum(np.conjugate(a)*Av)
        mus[iv,0] = bk
        mus[iv,1] = bk1
        ap = np.empty_like(v)
        for i in range(2,n):
            for s in range(nsites): ap[s] = -am[s]
            for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
            bk = a[0]-a[0] # zero of a's dtype
            for s in range(nsites):
                bk += np.conjugate(ap[s])*Av[s]
            mus[iv,i] = bk
            am,a,ap = a,ap,am # rotate the three buffers, no allocation
    return mus


@jit(nopython=True,parallel=True,cache=True)
def python_kpm_momentsA_batch_real(vs,Avs,data,row,col,n=100):
    """Real-arithmetic counterpart of python_kpm_momentsA_batch_complex."""
    nvec = vs.shape[0]
    nsites = vs.shape[1]
    nnz = len(data)
    mus = np.zeros((nvec,n),dtype=vs.dtype)
    for iv in prange(nvec):
        v = vs[iv]
        Av = Avs[iv]
        am = v.copy()
        a = Mtimesv(data,row,col,v)
        bk = np.sum(v*Av)
        bk1 = np.sum(a*Av)
        mus[iv,0] = bk
        mus[iv,1] = bk1
        ap = np.empty_like(v)
        for i in range(2,n):
            for s in range(nsites): ap[s] = -am[s]
            for k in range(nnz): ap[row[k]] += 2.*data[k]*a[col[k]]
            bk = a[0]-a[0]
            for s in range(nsites):
                bk += ap[s]*Av[s]
            mus[iv,i] = bk
            am,a,ap = a,ap,am
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




