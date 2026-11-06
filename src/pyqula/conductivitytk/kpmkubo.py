"""Chebyshev (KPM) Kubo conductivity of large sparse Hamiltonians.

This module computes the linear conductivity tensor of a system with 10^4
to 10^6 orbitals (moire cells, disordered supercells, finite flakes)
without diagonalizing anything: the spectral density of two velocity
operators is expanded in Chebyshev polynomials of the Hamiltonian, and the
trace is estimated with random vectors. The physics, the units and the
sign conventions are the ones of conductivity.py (hbar = e = 1, sigma in
e^2/hbar times a^(2-d), retarded branch, sigma_xy(0) 2 pi = -C for a Chern
insulator), and the velocity operator is the same one, v = i[H,r] with the
full bond vector, so both paths give the same tensor.

Double expansion
----------------
Let H~ = (H - b)/a be the Hamiltonian rescaled into (-1,1). The central
object is the double Chebyshev moment

  mu^ab_nm = g_n g_m Tr[v_a T_n(H~) v_b T_m(H~)] / V

with g_n the kernel (Jackson or Lorentz) and V the volume (the number of
k-points times the cell volume). With the kernel-broadened delta function

  delta(E - H) = sum_n phi_n(E) T_n(H~),
  phi_n(E) = (2 - delta_n0) T_n(x) / (pi a sqrt(1 - x^2)),  x = (E-b)/a,

the spectral density of the two currents is

  rho_ab(E,E') = Tr[v_b delta(E-H) v_a delta(E'-H)]/V
               = sum_nm mu^ab_nm phi_n(E') phi_m(E),

which in the eigenbasis is sum_nm v^a_nm v^b_mn delta(E-E_n) delta(E'-E_m)/V.

1) Absorptive part at finite frequency (Kubo-Greenwood). The Hermitian
part of the tensor, the part that absorbs energy, is

  sigma^H_ab(w) = pi int dE [f(E) - f(E+w)]/w rho_ab(E,E+w).

Its diagonal is Re sigma_aa(w), the absorption, and its off-diagonal
imaginary part is i Im sigma^A_xy(w), the circular dichroism. It is the
delta-function part of the Lehmann formula of conductivity.py, so for exact
moments it is the same number. The energy integral is done on
Chebyshev-Gauss nodes, which absorb the 1/sqrt(1-x^2) of the first delta.

2) The DC tensor (Kubo-Bastin), Garcia, Covaci and Rappoport, PRL 114,
116602 (2015), Eq. (3), re-derived here from

  sigma_ab = i int dE f(E) Tr[v_a delta(E-H) v_b dG+/dE - v_a dG-/dE v_b delta(E-H)]/V

with G+(e) = -i sum_n (2-delta_n0) e^(-i n arccos e) T_n(H~)/(a sqrt(1-e^2)).
Differentiating, the bracket becomes Garcia's Gamma_nm, and

  sigma_ab = 1/(pi a^2) int de f(a e + b) sum_nm (2-delta_n0)(2-delta_m0)
             mu^ab_nm Gamma_nm(e) / (1-e^2)^2,
  Gamma_nm(e) = (e - i n s) e^(i n t) T_m(e) + (e + i m s) e^(-i m t) T_n(e),

s = sqrt(1-e^2), t = arccos e. This holds the Fermi sea (Streda) part, so
it gives the quantized Hall conductivity of a Chern insulator at any
Fermi level in the gap. The moments do not depend on the Fermi level or on
the temperature, so a scan in the Fermi level is almost free.

Resolution
----------
delta is the half width at half maximum of the absorption line of a single
transition, the same meaning as in the band modes. A transition involves
two broadened deltas, so the KPM resolution of each one is delta/2 for the
Lorentz kernel (two Lorentzians convolve into one of the summed width,
the same line shape as the band modes) and delta/sqrt(2) for the Jackson
kernel (two Gaussians, a Gaussian line, with no Lorentzian tails). With
M = 2 npol moments (the package convention) the Lorentz kernel has half
width lamb a/M and the Jackson kernel sqrt(2 ln 2) pi a/M at the center of
the spectrum, narrowing as sqrt(1-x^2) towards the edges.

Cost
----
Per random vector and k-point, the left set L_n = T_n(H~) v_a |r> and the
right set R_m = v_b T_m(H~) |r> are built with sparse products, and
mu_nm = L_n^dagger R_m is a dense complex matrix product. The cost is 2 M
sparse products plus N M^2 operations at BLAS speed, and the memory is
2 nd N B complex numbers, with nd the number of Cartesian directions
and B <= M the number of polynomials stored at once. B is set by the
memory= budget. When B < M the left set is recomputed M/B times, so a
smaller memory costs more sparse products but never more memory.
"""
import numpy as np
import scipy.sparse as sp
from numba import jit, prange

from .. import parallel
from .. import klist
from ..kpmtk import kernels as kpmkernels


# ---------------------------------------------------------------------------
# sparse kernels


def _csr_step_py(indptr,indices,data,X,Z,c,d,e,out):
    """out = c A X + d X - e Z for a CSR matrix A and dense blocks X, Z of
    shape (N,ncol), parallel over the rows. This is one step of the
    Chebyshev recursion (c = 2/a, d = -2b/a, e = 1) and, with d = e = 0
    and c = 1, a plain sparse product"""
    n = X.shape[0]
    ncol = X.shape[1]
    for i in prange(n): # parallel loop over rows
        for q in range(ncol):
            out[i,q] = d*X[i,q] - e*Z[i,q]
        for p in range(indptr[i],indptr[i+1]): # nonzero entries of row i
            j = indices[p]
            hij = c*data[p]
            for q in range(ncol):
                out[i,q] = out[i,q] + hij*X[j,q]
    return out

_csr_step_parallel = jit(nopython=True,parallel=True,cache=True)(_csr_step_py)
_csr_step_serial = jit(nopython=True,cache=True)(_csr_step_py)
_serial_size = 50000 # below this many nonzeros times columns, the thread
                     # launch costs more than the product itself


def _csr_step(indptr,indices,data,X,Z,c,d,e,out):
    if len(data)*X.shape[1]<_serial_size:
        return _csr_step_serial(indptr,indices,data,X,Z,c,d,e,out)
    return _csr_step_parallel(indptr,indices,data,X,Z,c,d,e,out)


class _CSR():
    """A sparse matrix stored as CSR arrays for _csr_step"""
    def __init__(self,m):
        m = sp.csr_matrix(m,dtype=np.complex128)
        m.sum_duplicates()
        self.indptr = m.indptr.astype(np.int64)
        self.indices = m.indices.astype(np.int64)
        self.data = m.data
        self.n = m.shape[0]
        self.m = m
    def dot(self,X):
        out = np.empty(X.shape,dtype=np.complex128)
        return _csr_step(self.indptr,self.indices,self.data,X,X,
                1.,0.,0.,out)


def _chebyshev_vectors(H,X,a,b,M):
    """Generator of T_n(H~) X for n = 0..M-1, with H~ = (H-b)/a"""
    X = np.ascontiguousarray(X,dtype=np.complex128)
    t0 = X
    yield t0
    if M==1: return
    t1 = _csr_step(H.indptr,H.indices,H.data,t0,t0,1./a,-b/a,0.,
            np.empty(X.shape,dtype=np.complex128))
    yield t1
    for n in range(2,M):
        t2 = _csr_step(H.indptr,H.indices,H.data,t1,t0,2./a,-2.*b/a,1.,
                np.empty(X.shape,dtype=np.complex128))
        yield t2
        t0,t1 = t1,t2


# ---------------------------------------------------------------------------
# operators


def _positions(h,n):
    """Position of every orbital, (n,3), with the site index slowest as in
    kubo._bond_vectors"""
    if h.has_eh: raise NotImplementedError(
        "the KPM conductivity is not implemented for Nambu (superconducting) "
        "Hamiltonians: the charge current operator there is not i[H,r]")
    r = np.array(h.geometry.r,dtype=np.float64)
    if n%len(r)!=0: raise ValueError(
        "Hamiltonian dimension "+str(n)+" is not a multiple of the number "
        "of sites "+str(len(r)))
    return np.repeat(r,n//len(r),axis=0)


def _lattice(h):
    """Lattice vectors (d,3) and the cell volume (length, area or volume)"""
    g = h.geometry
    d = h.dimensionality
    avecs = np.array([g.a1,g.a2,g.a3][:d],dtype=np.float64).reshape((d,3))
    if d==0: vol = 1.
    elif d==1: vol = np.linalg.norm(avecs[0])
    elif d==2: vol = np.linalg.norm(np.cross(avecs[0],avecs[1]))
    else: vol = abs(np.dot(avecs[0],np.cross(avecs[1],avecs[2])))
    return avecs,vol


class _Operators():
    """Every sparse entry of the multicell Hamiltonian, with its lattice
    vector and its full bond vector d_ij = R + r_j - r_i, so that H(k),
    the velocities v_a(k) = i sum_R t_ij(R) d_ij^a exp(2 pi i k.R) and the
    inverse mass -sum_R t_ij(R) d_ij^a d_ij^b exp(2 pi i k.R) at any k are
    one rescaling of the same data. This is kubo._velocities (dH/dK plus
    i[H,r] for the intracell bonds) built sparsely."""
    def __init__(self,h):
        hd = h.get_hopping_dict() # {(n1,n2,n3): matrix}
        n = None
        rows,cols,data,Rs = [],[],[],[]
        for R,m in hd.items():
            m = sp.coo_matrix(m)
            n = m.shape[0]
            rows.append(m.row); cols.append(m.col)
            data.append(np.asarray(m.data,dtype=np.complex128))
            Rs.append(np.tile(np.array(R,dtype=np.float64),(len(m.data),1)))
        self.n = n
        self.row = np.concatenate(rows).astype(np.int64)
        self.col = np.concatenate(cols).astype(np.int64)
        self.data = np.concatenate(data)
        self.R = np.concatenate(Rs) # reduced lattice vector of each entry
        rr = _positions(h,n)
        avecs,self.cellvol = _lattice(h)
        d = h.dimensionality
        Rc = self.R[:,:d]@avecs if d>0 else np.zeros((len(self.data),3))
        self.d = Rc + rr[self.col] - rr[self.row] # full bond vectors
        scale = max(np.max(np.abs(self.data)),1e-300)
        w = np.abs(self.data)/scale
        # Cartesian directions with any current at all
        self.dirs = [i for i in range(3)
                if np.max(np.abs(self.d[:,i])*w)>1e-10]
    def _matrix(self,vals):
        return sp.csr_matrix((vals,(self.row,self.col)),shape=(self.n,self.n))
    def phase(self,k):
        return np.exp(2j*np.pi*(self.R@np.array(k,dtype=np.float64)))
    def hamiltonian(self,k):
        return self._matrix(self.data*self.phase(k))
    def velocity(self,k,alpha):
        return self._matrix(1j*self.d[:,alpha]*self.data*self.phase(k))
    def inverse_mass(self,k,alpha,beta):
        return self._matrix(-self.d[:,alpha]*self.d[:,beta]*self.data
                *self.phase(k))


def _kpoints(h,nk):
    if h.dimensionality==0: return np.zeros((1,3))
    return np.array(klist.kmesh(h.dimensionality,nk=nk),dtype=np.float64)


def _spectral_bounds(ops,ks):
    """Lowest and highest eigenvalue over the k-points, from Lanczos for a
    large matrix and from a dense diagonalization for a small one"""
    from scipy.sparse.linalg import eigsh
    emin,emax = np.inf,-np.inf
    for k in ks:
        Hk = ops.hamiltonian(k)
        if Hk.shape[0]<=400:
            es = np.linalg.eigvalsh(Hk.toarray())
            lo,hi = es[0],es[-1]
        else:
            try:
                hi = eigsh(Hk,k=1,which="LA",tol=1e-5,
                        return_eigenvectors=False)[0]
                lo = eigsh(Hk,k=1,which="SA",tol=1e-5,
                        return_eigenvectors=False)[0]
            except Exception: # ARPACK did not converge, take a strict bound
                from ..kpmtk.bandwidth import estimate_bandwidth
                hi = estimate_bandwidth(Hk); lo = -hi
        emin,emax = min(emin,lo),max(emax,hi)
    return emin,emax


# ---------------------------------------------------------------------------
# kernels and resolution


def _jackson(M,lamb): return kpmkernels.jackson_kernel(np.ones(M))
def _lorentz(M,lamb): return kpmkernels.lorentz_kernel(np.ones(M),lamb)

_kernels = {"jackson":_jackson,"lorentz":_lorentz}


def get_kernel_names():
    """Accepted values of kernel= in the KPM conductivity"""
    return list(_kernels)


def _check_kernel(kernel):
    if kernel not in _kernels: raise ValueError("unknown kernel '"
            +str(kernel)+"' for the KPM conductivity; it must be one of "
            +str(get_kernel_names()))


def _single_width(kernel,delta):
    """KPM half width of one delta function for a transition line of half
    width delta (see the module docstring)"""
    if kernel=="lorentz": return delta/2.
    return delta/np.sqrt(2.)


def npol_from_delta(a,delta,kernel="jackson",lorentz_lambda=3.):
    """Number of polynomials (M = 2 npol moments) for which a single
    transition gives an absorption line of half width delta, for a
    spectrum rescaled by a"""
    _check_kernel(kernel)
    if not np.isfinite(delta) or delta<=0.: raise ValueError(
        "delta is the half width of the absorption line of a transition "
        "and sets the number of Chebyshev polynomials; it must be a "
        "finite positive number, got "+str(delta))
    eta = _single_width(kernel,delta)
    if kernel=="jackson": return kpmkernels.jackson_npol(a,eta)
    return max(int(np.ceil(lorentz_lambda*a/(2.*eta))),3)


def line_width(a,npol,kernel="jackson",lorentz_lambda=3.):
    """Half width of the absorption line of one transition at the center of
    the spectrum, the inverse of npol_from_delta"""
    _check_kernel(kernel)
    M = 2*npol
    if kernel=="jackson": eta = kpmkernels.jackson_hwhm(a,npol)
    else: eta = lorentz_lambda*a/M
    return eta*(2. if kernel=="lorentz" else np.sqrt(2.))


# ---------------------------------------------------------------------------
# moments


_traces = ["stochastic","exact"]


def get_trace_names():
    """Accepted values of trace= in the KPM conductivity"""
    return list(_traces)


def _probe_blocks(n,trace,ntries,rng,exact_block=64):
    """Blocks of probe vectors (n,c). The stochastic trace gives one random
    phase vector per sample, <r|A|r> estimating Tr A; the exact trace gives
    the basis vectors, in blocks, all summed into a single sample"""
    if trace=="stochastic":
        for i in range(ntries):
            yield [np.exp(2j*np.pi*rng.random((n,1)))]
    elif trace=="exact":
        out = []
        for i0 in range(0,n,exact_block):
            i1 = min(n,i0+exact_block)
            X = np.zeros((n,i1-i0),dtype=np.complex128)
            X[np.arange(i0,i1),np.arange(i1-i0)] = 1.
            out.append(X)
        yield out
    else: _check_trace(trace)


def _check_trace(trace):
    if trace not in _traces: raise ValueError("unknown trace '"+str(trace)
            +"' for the KPM conductivity; it must be one of "
            +str(get_trace_names()))


def _block_size(n,ncol,nd,M,memory):
    """Number of polynomials stored at once within the memory budget (GB)"""
    per = 2*nd*n*ncol*16. # bytes per stored polynomial, left and right
    return int(max(1,min(M,np.floor(memory*1e9/per))))


def _vector_moments(H,vs,X,a,b,M,memory):
    """Raw moments Tr_X[v_a T_n(H~) v_b T_m(H~)] = sum over the columns of X
    of L_n^dagger R_m, with L_n = T_n v_a X and R_m = v_b T_m X, as an
    array (nd,nd,M,M). The right set is streamed in blocks of B
    polynomials, and for each block the left set is run again"""
    nd = len(vs)
    n,ncol = X.shape
    B = _block_size(n,ncol,nd,M,memory)
    mu = np.zeros((nd,nd,M,M),dtype=np.complex128)
    VX = np.concatenate([v.dot(X) for v in vs],axis=1) # (n,nd*ncol)
    R = np.empty((nd,B,n*ncol),dtype=np.complex128)
    L = np.empty((nd,B,n*ncol),dtype=np.complex128)
    right = _chebyshev_vectors(H,X,a,b,M)
    for m0 in range(0,M,B): # blocks of the right set
        m1 = min(M,m0+B)
        for im in range(m1-m0):
            t = next(right)
            for ib in range(nd): R[ib,im] = vs[ib].dot(t).ravel()
        left = _chebyshev_vectors(H,VX,a,b,M)
        for n0 in range(0,M,B): # blocks of the left set
            n1 = min(M,n0+B)
            for i_n in range(n1-n0):
                t = next(left)
                for ia in range(nd): # stored conjugated, for the product
                    L[ia,i_n] = t[:,ia*ncol:(ia+1)*ncol].ravel().conj()
            for ia in range(nd):
                for ib in range(ia,nd): # (b,a) follows from (a,b) below
                    mu[ia,ib,n0:n1,m0:m1] += L[ia,:n1-n0]@R[ib,:m1-m0].T
    # Tr[v_b T_n v_a T_m] = Tr[v_a T_m v_b T_n], so mu^ba = (mu^ab)^T
    for ia in range(nd):
        for ib in range(ia+1,nd): mu[ib,ia] = mu[ia,ib].T
    return mu


class _Setup():
    """Everything the moments need: the operators, the k-points, the
    rescaling and the number of moments"""
    def __init__(self,h,nk=1,delta=0.1,npol=None,kernel="jackson",
            lorentz_lambda=3.,scale=None,volume=None):
        _check_kernel(kernel)
        self.ops = _Operators(h)
        self.ks = _kpoints(h,nk)
        self.dirs = self.ops.dirs
        self.emin,self.emax = _spectral_bounds(self.ops,self.ks)
        if scale is None:
            self.b = (self.emax+self.emin)/2.
            self.a = 1.05*(self.emax-self.emin)/2. + 1e-6
        else:
            self.b = 0.
            self.a = float(scale)
            if self.a<=max(abs(self.emin),abs(self.emax)): raise ValueError(
                "scale="+str(scale)+" does not cover the spectrum, "
                "which spans ["+str(self.emin)+","+str(self.emax)+"]")
        if npol is None:
            npol = npol_from_delta(self.a,delta,kernel=kernel,
                    lorentz_lambda=lorentz_lambda)
        self.npol = int(npol)
        self.M = 2*self.npol
        self.kernel = kernel
        self.g = _kernels[kernel](self.M,lorentz_lambda)
        self.delta = line_width(self.a,self.npol,kernel=kernel,
                lorentz_lambda=lorentz_lambda)
        if volume is None: volume = self.ops.cellvol
        self.V = len(self.ks)*volume


def moments(setup,ntries=10,trace="stochastic",seed=None,memory=2.):
    """Generator of the kernel-weighted double moments mu^ab_nm (one array
    (nd,nd,M,M) per random vector, summed over the k-points and divided by
    the volume). With trace="exact" it yields a single exact sample"""
    parallel.set_num_threads()
    rng = np.random.default_rng(seed)
    st = setup
    gg = st.g[:,None]*st.g[None,:]
    nd = len(st.dirs)
    # one random vector per k-point and sample
    probes = [_probe_blocks(st.ops.n,trace,ntries,rng) for k in st.ks]
    Hs = [_CSR(st.ops.hamiltonian(k)) for k in st.ks]
    vs = [[_CSR(st.ops.velocity(k,al)) for al in st.dirs] for k in st.ks]
    for blocks in zip(*probes): # loop over samples
        mu = np.zeros((nd,nd,st.M,st.M),dtype=np.complex128)
        for ik in range(len(st.ks)):
            for X in blocks[ik]:
                mu += _vector_moments(Hs[ik],vs[ik],X,st.a,st.b,st.M,memory)
        yield mu*gg/st.V


def exact_moments(setup):
    """The same moments from a full diagonalization at every k-point, the
    reference for the Chebyshev and stochastic errors (small systems only)"""
    st = setup
    nd = len(st.dirs)
    mu = np.zeros((nd,nd,st.M,st.M),dtype=np.complex128)
    ns = np.arange(st.M)
    for k in st.ks:
        Hk = st.ops.hamiltonian(k).toarray()
        es,U = np.linalg.eigh(Hk)
        x = np.clip((es-st.b)/st.a,-1.,1.)
        Tn = np.cos(ns[:,None]*np.arccos(x)[None,:]) # (M,nstates)
        vv = [U.conj().T@st.ops.velocity(k,al).toarray()@U for al in st.dirs]
        for ia in range(nd):
            for ib in range(nd):
                # Tr[v_a T_n v_b T_m] = sum_pq va_pq Tn(E_q) vb_qp Tm(E_p)
                A = vv[ia]*vv[ib].T # A[p,q] = va_pq vb_qp
                mu[ia,ib] += Tn@A.T@Tn.T
    return mu*(st.g[:,None]*st.g[None,:])/st.V


# ---------------------------------------------------------------------------
# energy integrals


def _fermi(E,mu,T):
    if T<=0.: return np.where(E<mu,1.,np.where(E==mu,0.5,0.))
    return 1./(1.+np.exp(np.clip((E-mu)/T,-500.,500.)))


def _ratio(E,w,mu,T):
    """[f(E) - f(E+w)]/w, with its w -> 0 limit -df/dE"""
    if abs(w)>1e-10: return (_fermi(E,mu,T)-_fermi(E+w,mu,T))/w
    if T<=0.: return np.zeros(E.shape)
    f = _fermi(E,mu,T)
    return f*(1.-f)/T


def _phi(x,M):
    """(2 - delta_n0) T_n(x) for n < M, (M,len(x))"""
    out = np.cos(np.arange(M)[:,None]*np.arccos(np.clip(x,-1.,1.))[None,:])
    out[1:] *= 2.
    return out


@jit(nopython=True,parallel=True,cache=True)
def _contract(x2,idx,wts,Y):
    """sum_j wts_j sum_n (2 - delta_n0) T_n(x2_j) Y[idx_j,ab,n], with the
    Chebyshev polynomials built by their recursion, parallel over the
    energy nodes"""
    nj = len(idx)
    nab = Y.shape[1]
    M = Y.shape[2]
    part = np.zeros((nj,nab),dtype=np.complex128)
    for j in prange(nj): # parallel loop over the energy nodes
        x = x2[j]
        y = Y[idx[j]]
        t0 = 1.
        t1 = x
        for ab in range(nab): part[j,ab] = y[ab,0]
        for n in range(1,M):
            for ab in range(nab): part[j,ab] += 2.*t1*y[ab,n]
            t0,t1 = t1,2.*x*t1-t0
        for ab in range(nab): part[j,ab] *= wts[j]
    out = np.zeros(nab,dtype=np.complex128)
    for j in range(nj): out += part[j]
    return out


def absorptive(st,mu,energies,fermi=0.,T=0.,ne=None):
    """Absorptive (Hermitian) part of sigma_ab(w), (nw,3,3), from one set of
    moments (see the module docstring)"""
    M = st.M
    a,b = st.a,st.b
    if ne is None: ne = 4*M
    j = np.arange(ne)
    x = np.cos(np.pi*(j+0.5)/ne) # Chebyshev-Gauss nodes
    E = a*x + b
    # energy window where the spectrum is, padded by many line widths
    pad = 10.*st.delta
    inside = (E>st.emin-pad)&(E<st.emax+pad)
    phit = _phi(x,M)/(np.pi*a) # phi_n times sqrt(1-x^2)
    nd = len(st.dirs)
    Y = mu.real@phit + 1j*(mu.imag@phit) # Y[a,b,n,j] = sum_m mu_nm phi~_m(E_j)
    Y = np.ascontiguousarray(np.transpose(Y.reshape((nd*nd,M,ne)),(2,0,1)))
    parallel.set_num_threads()
    out = np.zeros((len(energies),3,3),dtype=np.complex128)
    for iw,w in enumerate(energies):
        r = _ratio(E,w,fermi,T)
        E2 = E + w
        keep = inside & (E2>st.emin-pad) & (E2<st.emax+pad) & \
                (np.abs(E2-b)<a*(1.-1e-9)) & \
                (np.abs(r)>1e-14*np.max(np.abs(r))+1e-300)
        if not np.any(keep): continue
        idx = np.nonzero(keep)[0]
        x2 = (E2[idx]-b)/a
        # phi_n(E+w) = (2 - delta_n0) T_n(x2)/(pi a sqrt(1-x2^2)), times the
        # weight of pi int dE on the Chebyshev-Gauss nodes
        wts = r[idx]*(np.pi/ne)*a*np.pi/(np.pi*a*np.sqrt(1.-x2**2))
        s = _contract(x2,idx,wts,Y).reshape((nd,nd))
        for ia,al in enumerate(st.dirs):
            for ib,be in enumerate(st.dirs): out[iw,al,be] = s[ia,ib]
    return out


def bastin(st,mu,fermis,T=0.,ne=None):
    """DC conductivity tensor (Kubo-Bastin), (len(fermis),3,3) real, from
    one set of moments (see the module docstring)"""
    M = st.M
    a,b = st.a,st.b
    fermis = np.atleast_1d(np.array(fermis,dtype=np.float64))
    if ne is None: ne = 4*M
    pad = 10.*st.delta
    Tc = max(T,0.)*40.
    elo = max(st.emin-pad,b-a*(1.-1e-6))
    ehi = min(np.max(fermis)+Tc+1e-9,b+a*(1.-1e-6))
    out = np.zeros((len(fermis),3,3))
    if ehi<=elo: return out
    xg,wg = np.polynomial.legendre.leggauss(ne)
    e = ((ehi-elo)*xg/2. + (ehi+elo)/2. - b)/a # rescaled energies
    we = wg*(ehi-elo)/(2.*a) # weights in the rescaled variable
    s = np.sqrt(1.-e**2)
    th = np.arccos(e)
    ns = np.arange(M)
    c = np.where(ns==0,1.,2.)[:,None]
    Tm = c*np.cos(ns[:,None]*th[None,:]) # (2-delta) T_n(e)
    P = c*(e[None,:]-1j*ns[:,None]*s[None,:])*np.exp(1j*ns[:,None]*th[None,:])
    # sum_nm mu_nm [P_n T_m + T_n conj(P_m)]. The exact moments are
    # Hermitian in (n,m), and then the second term is the complex conjugate
    # of the first. A stochastic estimate is Hermitian only on average, so
    # it is symmetrized first, which leaves the estimate unbiased
    muH = (mu + np.conj(np.swapaxes(mu,2,3)))/2.
    re,im = muH.real@Tm,muH.imag@Tm # two real products, not a complex one
    I = 2.*(np.sum(P.real[None,None]*re,axis=2) -
            np.sum(P.imag[None,None]*im,axis=2))
    I = I/(np.pi*a**2*(1.-e**2)**2)
    E = a*e + b
    for i,mf in enumerate(fermis):
        f = _fermi(E,mf,T)
        sab = I@(f*we)
        # Garcia's order of the velocities is the transpose of the retarded
        # Kubo formula of conductivity.py (Crepieux and Bruno's order), so
        # the (a,b) moments give the (b,a) component
        for ia,al in enumerate(st.dirs):
            for ib,be in enumerate(st.dirs): out[i,be,al] = sab[ia,ib]
    return out


# ---------------------------------------------------------------------------
# public entry points


def get_integration_names():
    """The integration= names this module answers to in
    conductivity.optical_conductivity"""
    return ["kpm"]


def _mean_and_error(samples):
    s = np.array(samples)
    m = np.mean(s,axis=0)
    if len(s)<2: return m,np.zeros(m.shape)
    return m,np.std(s,axis=0,ddof=1)/np.sqrt(len(s))


def _check_channels(intraband,interband):
    if not (intraband and interband): raise NotImplementedError(
        "the KPM conductivity does not separate the intraband and interband "
        "channels, since it never builds the eigenstates; use "
        "integration='grid' for intraband=False or interband=False")


def optical_conductivity(h,energies=None,nk=1,T=None,delta=0.1,
        component=None,npol=None,ntries=10,kernel="jackson",
        lorentz_lambda=3.,trace="stochastic",seed=None,fermi=0.,
        scale=None,memory=2.,volume=None,ne=None,return_error=False,
        intraband=True,interband=True,**kwargs):
    """
    integration="kpm"
    -----------------
    For a large sparse Hamiltonian (10^4 to 10^6 orbitals: a big supercell
    at the Gamma point or a few k-points, or a finite flake), the
    Chebyshev double expansion of conductivitytk/kpmkubo.py. It never
    diagonalizes, and its cost and memory are linear in the number of
    orbitals. It returns the absorptive (Hermitian) part of the tensor,
    sigma^H_ab = (sigma_ab + sigma_ba^*)/2: Re sigma_aa(w) on the
    diagonal and i Im sigma^A_ab(w), the circular dichroism, off the
    diagonal. The reactive part (Im sigma_aa, Re sigma_xy at w > 0) is not
    computed; the DC Hall conductivity is kpmkubo.dc_conductivity
    (h.get_dc_conductivity). Keywords of this mode:

    nk : int
      k-points per periodic direction, default 1 (the Gamma point)
    delta : float
      half width of the absorption line of one transition, which sets the
      number of polynomials when npol is None. The KPM width of each of
      the two delta functions is delta/2 (Lorentz kernel, a Lorentzian
      line, as in the band modes) or delta/sqrt(2) (Jackson kernel, a
      Gaussian line)
    npol : int
      number of polynomials, 2 npol Chebyshev moments; the cost grows as
      npol^2
    kernel : str
      "jackson" (default, no tails, the width narrows as sqrt(1-x^2)
      towards the band edges) or "lorentz", with lorentz_lambda (default 3)
    ntries, trace, seed
      number of random vectors, "stochastic" (default) or "exact" trace
      (all the basis vectors, small systems only), and the seed
    fermi : float
      Fermi level, default 0
    T : float
      temperature, default delta
    scale : float
      half width of the rescaled spectrum, default from Lanczos padded by 5%
    memory : float
      budget in GB for the stored Chebyshev vectors (default 2); a smaller
      budget recomputes the left set more often, at the same result
    volume : float
      volume that normalizes the result, default the cell volume (1 for a
      finite system, which then gives the extensive total)
    ne : int
      energy nodes of the integral, default 4 times the moments
    return_error : bool
      also return the standard error over the random vectors
    intraband, interband
      must be True, the expansion does not separate the two channels
    """
    _check_channels(intraband,interband)
    _check_trace(trace)
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta # same default as kubo.optical_conductivity
    st = _Setup(h,nk=nk,delta=delta,npol=npol,kernel=kernel,
            lorentz_lambda=lorentz_lambda,scale=scale,volume=volume)
    samples = [absorptive(st,mu,energies,fermi=fermi,T=T,ne=ne)
            for mu in moments(st,ntries=ntries,trace=trace,seed=seed,
                memory=memory)]
    sigma,err = _mean_and_error(samples)
    if component is not None:
        from .kubo import _index
        i,j = _index(component[0]),_index(component[1])
        sigma,err = sigma[:,i,j],err[:,i,j]
    if return_error: return energies,sigma,err
    return energies,sigma


def dc_conductivity(h,fermi=0.,nk=1,T=None,delta=0.1,npol=None,ntries=10,
        kernel="jackson",lorentz_lambda=3.,trace="stochastic",seed=None,
        scale=None,memory=2.,volume=None,ne=None,component=None,
        return_error=False):
    """DC conductivity tensor sigma_ab(fermi,T) of a large sparse
    Hamiltonian by the Kubo-Bastin formula with a Chebyshev double
    expansion (Garcia, Covaci and Rappoport, PRL 114, 116602 (2015)).

    It holds the Fermi sea part, so sigma_xy gives the quantized Hall
    conductivity of a Chern insulator, sigma_xy 2 pi = -C in units of
    e^2/hbar (the sign convention of conductivity.py). sigma_xx is the DC
    longitudinal conductivity, limited by the KPM resolution in a clean
    system: the Drude peak of width delta gives D/delta.

    Parameters
    ----------
    fermi : float or array
      Fermi level(s). The moments do not depend on it, so an array costs
      almost nothing more
    nk : int
      k-points per periodic direction (Gamma-centered mesh); the default 1
      is the Gamma point of a large cell
    T : float
      temperature of the Fermi function, default delta
    delta, npol, kernel, lorentz_lambda
      the resolution: npol polynomials (2 npol moments), or the npol for
      which a transition line has half width delta
    ntries, trace, seed
      number of random vectors, "stochastic" or "exact" trace, and the
      seed of the random vectors
    scale, memory, volume, ne
      half width of the rescaled spectrum (default from Lanczos, padded by
      5%), the memory budget in GB for the stored polynomials, the volume
      that normalizes a finite system (default 1) and the number of energy
      nodes of the integral (default 4 times the moments)
    component : str or None
      e.g. "xy" to return only that component
    return_error : bool
      also return the standard error over the random vectors

    Returns
    -------
    sigma of shape (3,3), or (len(fermi),3,3) for an array of Fermi levels,
    real; with return_error=True the tuple (sigma, error)
    """
    _check_trace(trace)
    if T is None: T = delta
    st = _Setup(h,nk=nk,delta=delta,npol=npol,kernel=kernel,
            lorentz_lambda=lorentz_lambda,scale=scale,volume=volume)
    samples = [bastin(st,mu,fermi,T=T,ne=ne)
            for mu in moments(st,ntries=ntries,trace=trace,seed=seed,
                memory=memory)]
    sigma,err = _mean_and_error(samples)
    if np.ndim(fermi)==0: sigma,err = sigma[0],err[0]
    if component is not None:
        from .kubo import _index
        i,j = _index(component[0]),_index(component[1])
        sigma,err = sigma[...,i,j],err[...,i,j]
    if return_error: return sigma,err
    return sigma


def sum_rule_weight(h,nk=1,T=0.,fermi=0.,npol=200,ntries=10,
        trace="stochastic",seed=None,scale=None,volume=None):
    """Diamagnetic weight W_ab = Tr[f(H) d2H/dK_a dK_b]/V of the f-sum rule,
    int Re sigma_aa(w) dw = pi W_aa over the whole real axis, from a single
    Chebyshev expansion of the Fermi function with the Jackson kernel.
    Returns a real 3x3 array"""
    parallel.set_num_threads()
    st = _Setup(h,nk=nk,npol=npol,scale=scale,volume=volume)
    M = st.M
    ne = 4*M
    x = np.cos(np.pi*(np.arange(ne)+0.5)/ne)
    # Chebyshev coefficients of f(a x + b), with the Jackson kernel
    cf = _phi(x,M)@_fermi(st.a*x+st.b,fermi,T)/ne*st.g
    rng = np.random.default_rng(seed)
    out = []
    probes = [list(_probe_blocks(st.ops.n,trace,ntries,rng)) for k in st.ks]
    for isample in range(len(probes[0])):
        W = np.zeros((3,3))
        for ik,k in enumerate(st.ks):
            H = _CSR(st.ops.hamiltonian(k))
            taus = {(al,be):_CSR(st.ops.inverse_mass(k,al,be))
                    for al in st.dirs for be in st.dirs}
            for X in probes[ik][isample]:
                tX = {key:t.dot(X) for key,t in taus.items()}
                for n,t in enumerate(_chebyshev_vectors(H,X,st.a,st.b,M)):
                    for (al,be),y in tX.items():
                        W[al,be] += cf[n]*np.vdot(y,t).real
        out.append(W/st.V)
    return _mean_and_error(out)[0]
