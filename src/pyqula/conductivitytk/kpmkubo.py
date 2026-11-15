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
At T=0 the occupation factor is a box of width w, which a small w puts
between two nodes, and at a T below the node spacing the Fermi step falls
between two nodes in the same way. Near the two Fermi edges the rest of
the integrand is interpolated by its Chebyshev series on the nodes and
integrated against the occupation factor exactly: in closed form for the
box at T=0, and with the exact cosine moments of the thermal factor at
T > 0, which is the box averaged over the Fermi level with -df/dE. Away
from the edges the midpoint rule on the nodes is exact. So any w and any
T work, w=0 included, and a T of several node spacings, as the default
T = delta, gives the midpoint rule on the nodes to rounding (absorptive
and _box_weights).

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
import functools
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


class _Bounds():
    """The operators, the k-points, the rescaling H~ = (H - b)/a and the
    volume, which every Chebyshev route of this module needs"""
    def __init__(self,h,nk=1,scale=None,volume=None):
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
        if volume is None: volume = self.ops.cellvol
        self.V = len(self.ks)*volume


class _Setup(_Bounds):
    """Everything the double moments need: the bounds, and the kernel and
    number of moments of the resolution"""
    def __init__(self,h,nk=1,delta=0.1,npol=None,kernel="jackson",
            lorentz_lambda=3.,scale=None,volume=None):
        _check_kernel(kernel)
        super().__init__(h,nk=nk,scale=scale,volume=volume)
        if npol is None:
            npol = npol_from_delta(self.a,delta,kernel=kernel,
                    lorentz_lambda=lorentz_lambda)
        self.npol = int(npol)
        self.M = 2*self.npol
        self.kernel = kernel
        self.g = _kernels[kernel](self.M,lorentz_lambda)
        self.delta = line_width(self.a,self.npol,kernel=kernel,
                lorentz_lambda=lorentz_lambda)


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


def _box_weights(ne,fermi,w,a,b,T=0.):
    """Weights q_j of the ne Chebyshev-Gauss nodes x_j = cos(theta_j),
    theta_j = pi (j + 1/2)/ne, for the absorptive part,

      pi int dE [f(E) - f(E+w)]/w rho(E,E+w) = pi sum_j q_j G_j,

    with G(theta) = a sin(theta) rho(E,E+w) at E = a cos(theta) + b, the
    integrand in theta (dE = a sin(theta) dtheta), and G_j its value at the
    node j (see absorptive), at any temperature T >= 0.

    At T=0 the occupation factor is a box: 1/|w| between fermi - w and
    fermi (between fermi and fermi + |w| for w < 0), zero elsewhere. The
    nodes are spaced by about pi a/ne, so a box narrower than that holds
    zero or one of them, and the midpoint rule, q_j = r(E_j) pi/ne, then
    gives zero or several times the right value. At a finite T below the
    spacing the Fermi step falls between two nodes in the same way.

    Here G is replaced by its Chebyshev interpolant on the same nodes,

      G(theta) = sum_{k<ne} c_k cos(k theta),
      c_k = ((2 - delta_k0)/ne) sum_j G_j cos(k theta_j),

    from the discrete orthogonality of cos(k theta_j) on these nodes. It is
    exact for a polynomial of degree below ne in x = cos(theta). G is a
    polynomial of degree 2M-2, below ne = 4M, times the smooth factor
    1/sqrt(1 - x2^2) of the second delta function, so the interpolant
    follows it closely wherever the spectrum is. The interpolant is then
    integrated over the box in closed form. In x this is the integral of
    the Chebyshev series with its weight, int T_k(x) dx/sqrt(1-x^2) =
    -sin(k arccos x)/k, and in theta, over [theta1, theta1 + dtheta],

      int cos(k theta) dtheta = dtheta cos(k thetac) sinc(k dtheta/2),

    with thetac = theta1 + dtheta/2 the center and sinc(u) = sin(u)/u. So
    the interpolant integrated over the box and divided by |w| is
    sum_j W_j G_j, with

      W_j = (dtheta/|w|) (1/ne) [1 + 2 sum_{k>=1} cos(k theta_j) e_k],
      e_k = cos(k thetac) sinc(k dtheta/2),

    a DCT of type 3 of e_k. A box that covers every node gives
    W_j = pi/(ne |w|), the rule of the finite temperature with r = 1/|w|.

    The box is [x1, x2] in x, cut to [-1, 1], with theta1 = arccos(x2).
    From tan(dtheta/2) = (cos theta1 - cos theta2)/(sin theta1 +
    sin theta2), dtheta = 2 arctan((x2 - x1)/(s1 + s2)) with
    s = sqrt(1 - x^2), which has no cancellation for a small box, and
    x2 - x1 = |w|/a when the box is not cut. So any w works, and w = 0
    is the limit: the box over |w| becomes delta(E - fermi), dtheta/|w|
    tends to 1/(a s), and the result is pi rho(fermi, fermi), the KPM
    Drude peak at w = 0.

    The interpolation is only needed at the two edges of the box, so it is
    kept local there. With psi a smooth window that is one within 36 nodes
    of either edge and zero from 72 nodes on (_edge_window), and B the box,

      int B G dtheta = int B (G psi) dtheta + int B (1 - psi) G dtheta.

    The first term is the interpolant of G psi integrated over the box as
    above, sum_j psi_j W_j G_j with W_j the weights of the interpolant. In
    the second, B (1 - psi) G is smooth, since 1 - psi vanishes where B
    jumps, so the midpoint rule pi/ne is exact for it to the same accuracy.
    So

      q_j = psi_j W_j + (1 - psi_j) B_j pi/(ne |w|).

    A narrow box lies inside the window, and q_j = psi_j W_j. Without the
    window every node of the spectrum would enter the interpolant, and the
    edge of the rescaled spectrum, where the 1/sqrt(1 - x2^2) of the second
    delta function diverges, would ring into the box. With it the weights
    are local, and that edge enters only through the midpoint rule inside
    a box that reaches it.

    At a finite T the same construction holds with the box replaced by the
    thermal factor r(theta) = [f(E) - f(E+w)]/w,

      q_j = psi_j W_j + (1 - psi_j) r(E_j) pi/ne,
      W_j = (1/ne) [R_0 + 2 sum_{0<k<ne} cos(k theta_j) R_k],
      R_k = int_0^pi r(theta) cos(k theta) dtheta,

    with the same window psi at the two edges, fermi and fermi - w, since
    r - r_box decays as exp(-|E - edge|/T) away from them and 1 - psi
    vanishes wherever r is not smooth on the scale of the nodes. The
    finite-T factor is the T=0 box averaged over the Fermi level,
    f_T(E - mu) = int dmu' p(mu' - mu) Theta(mu' - E) with
    p(x) = 1/(4 T cosh^2(x/2T)) = -df/dx, so the R_k are the box moments
    averaged with p, and they are computed exactly at every T
    (_thermal_moments). A T above a few node spacings makes r smooth on
    the node scale: its cosine moments beyond k = ne fall as
    exp(-pi^2 T/spacing), W_j is then r(E_j) pi/ne to rounding, and q_j is
    the midpoint rule of the finite temperature. T -> 0 gives the box"""
    from scipy.fft import dct
    if T>0.: return _thermal_weights(ne,fermi,w,a,b,T)
    lo,hi = fermi-max(w,0.),fermi-min(w,0.) # the box in energy
    x1,x2 = max((lo-b)/a,-1.),min((hi-b)/a,1.) # in x, cut to [-1,1]
    q = np.zeros(ne)
    if w==0.: # the limit, a delta function at the Fermi level
        if not -1.<x2<1.: return q # outside the rescaled spectrum
        th1,dth = np.arccos(x2),0.
        dth_w = 1./(a*np.sqrt(1.-x2**2)) # dtheta/|w| as w -> 0
    else:
        if not x1<x2: return q # the box is outside the rescaled spectrum
        cut = (lo-b)/a<-1. or (hi-b)/a>1.
        d = x2-x1 if cut else abs(w)/a # width of the box in x
        s1,s2 = np.sqrt(1.-x2**2),np.sqrt(1.-x1**2)
        th1 = np.arccos(x2)
        dth = 2.*np.arctan2(d,s1+s2) # width in theta, see above
        dth_w = dth/abs(w)
    k = np.arange(ne)
    th = np.pi*(k+0.5)/ne # the nodes in theta
    e = np.cos(k*(th1+dth/2.))*np.sinc(k*dth/(2.*np.pi)) # np.sinc(u/pi)
    W = dth_w*dct(e,type=3)/ne # interpolant over the box, divided by |w|
    tau = 6.*np.pi/ne # width of the edges of the window, 6 nodes
    psi = 1.-(1.-_edge_window(th,th1,tau))*(1.-_edge_window(th,th1+dth,tau))
    box = (th>=th1)&(th<=th1+dth) # the nodes inside the box
    # the midpoint rule inside; at w=0 the box has no width and no inside
    mid = np.where(box,np.pi/(ne*abs(w)),0.) if w!=0. else 0.
    q = psi*W + (1.-psi)*mid
    q[(psi<1e-16)&~box] = 0. # far from the edges and outside the box
    return q


def _edge_window(th,the,tau):
    """Smooth window in theta, at the angles th in [0, pi], one within 6 tau
    of the edge the and zero beyond 12 tau, each side an error function of
    width tau, so it differs from one and from zero there by
    erfc(6)/2 ~ 1e-17. The Fourier content of a side falls as
    exp(-k^2 tau^2/4), so with tau = 6 pi/ne it is exp(-22), 2e-10, at
    k = ne/2, the room the ne nodes leave above the degree 2M-2 = ne/2 - 2
    of G, and G times the window is interpolated as well as G itself. G
    is a function of cos(theta), even and of period 2 pi in theta, and the
    window is made so too: the union of the window and its images at
    -theta and 2 pi - theta, still one near the edge and smooth across
    theta = 0 and pi"""
    from scipy.special import erf
    m = 6.*tau
    def chi(t): return 0.5*(erf((t-the+m)/tau)-erf((t-the-m)/tau))
    return 1.-(1.-chi(th))*(1.-chi(-th))*(1.-chi(2.*np.pi-th))


# ---------------------------------------------------------------------------
# finite temperature: the exact cosine moments of the thermal factor


_SUPPORT = 44. # |E - edge|/T beyond which p and f - Theta are below 1e-18
_ALIAS = 45. # exp(-45), the aliasing left by the fine midpoint rule
_PANELS = np.array([0.,1.,3.,9.,27.,81.]) # Gauss panels from an edge, in T
_GAUSS = np.polynomial.legendre.leggauss(24)


def _fermi_x(x):
    """1/(1 + e^x) for an array x, without overflow and with the tail
    e^-x kept down to the smallest float"""
    x = np.asarray(x,dtype=np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x>0.,e/(1.+e),1./(1.+e))


def _thermal_ratio(y,w,T,yw=None):
    """r = [f(y) - f(y+w)]/w for the energies y measured from the Fermi
    level, with the difference taken without cancellation: for u < v,
    f(u) - f(v) = f(u) (1 - f(v)) (1 - exp(-(v-u)/T)). It is -df/dy =
    f (1 - f)/T at w = 0. yw is y + w, passed when it is known to more
    digits than the sum (near the edge at the Fermi level minus w)"""
    if yw is None: yw = y+w
    u,v = (yw,y) if w<0. else (y,yw)
    fu,gv = _fermi_x(u/T),_fermi_x(-v/T) # f(u) and 1 - f(v)
    if w==0.: return fu*gv/T
    return fu*gv*(-np.expm1(-abs(w)/T))/abs(w)


def _thermal_slope(y,w,T,yw=None):
    """[p(y) - p(y+w)]/w with p = -df/dy = f (1 - f)/T, without
    cancellation: p(u) - p(v) = [f(u) - f(v)] [1 - f(u) - f(v)]/T"""
    if yw is None: yw = y+w
    return _thermal_ratio(y,w,T,yw=yw)*(_fermi_x(-y/T)-_fermi_x(yw/T))/T


def _rise(x0,e):
    """arccos(x0) - arccos(x0 + e) for e >= 0, cut at arccos(x0) when
    x0 + e passes 1, from tan(dtheta/2) = (x1 - x0)/(s0 + s1) as in
    _box_weights, so a small e loses no digits"""
    e = np.asarray(e,dtype=np.float64)
    x1 = np.minimum(x0+e,1.)
    out = 2.*np.arctan2(e,np.sqrt(1.-x0*x0)+np.sqrt(1.-x1*x1))
    return np.where(x0+e>=1.,np.arccos(x0),out)


def _fall(x0,e):
    """arccos(x0 - e) - arccos(x0) for e >= 0, cut at pi - arccos(x0)"""
    e = np.asarray(e,dtype=np.float64)
    x1 = np.maximum(x0-e,-1.)
    out = 2.*np.arctan2(e,np.sqrt(1.-x0*x0)+np.sqrt(1.-x1*x1))
    return np.where(x0-e<=-1.,np.pi-np.arccos(x0),out)


def _chirp(n,N):
    """exp(i pi n^2/(2N)), the chirp of the transform of _exp_sums. For an
    integer N the phase is reduced exactly, n^2 mod 4N, so a large n loses
    no digits; a float N is only used for a step pi/N so small that the
    phase stays small"""
    if isinstance(N,int):
        n = np.asarray(n,dtype=np.int64)
        return np.exp(1j*np.pi*((n*n)%(4*N))/(2.*N))
    n = np.asarray(n,dtype=np.float64)
    return np.exp(1j*np.pi*(n*n)/(2.*N))


def _exp_sums(g,N,K):
    """S_k = sum_m g_m exp(i pi k m/N) for k < K, the Fourier sums of the
    samples g on a grid of step pi/N, by Bluestein's chirp z-transform,
    k m = [k^2 + m^2 - (k-m)^2]/2, so the cost is three FFTs of length
    about len(g) + K, whatever N"""
    from scipy.fft import fft,ifft,next_fast_len
    Q = len(g)
    L = next_fast_len(Q+K-1)
    ch = _chirp(np.arange(max(Q,K)),N)
    A = np.zeros(L,dtype=np.complex128)
    A[:Q] = g*ch[:Q]
    B = np.zeros(L,dtype=np.complex128)
    B[:K] = np.conj(ch[:K])
    if Q>1: B[L-Q+1:] = np.conj(ch[Q-1:0:-1])
    c = ifft(fft(A)*fft(B))[:K]
    return ch[:K]*c


@functools.lru_cache(maxsize=64)
def _fermi_excess(nu,a,b,T):
    """G(nu) = int_0^pi [f(E - nu) - Theta(nu - E)] dtheta, E = a cos(theta)
    + b, the difference between the cosine moment k=0 of the Fermi function
    and of the step, by Gauss-Legendre on panels that start at the edge
    theta(nu) and grow as 1, 3, 9, 27, 81 T in energy on each side, cut at
    0 and pi. The integrand is smooth on each side and decays as
    exp(-|E - nu|/T), and the pole of f at i pi T above the edge stays
    well outside the Bernstein ellipse of every panel, so the panels and
    their 24 nodes do not depend on T"""
    x = (nu-b)/a
    xc = min(max(x,-1.),1.)
    th0 = np.arccos(xc)
    y0 = 0. if -1.<x<1. else a*xc+b-nu # E(th0) - nu
    xg,wg = _GAUSS
    out = 0.
    for side in (1.,-1.): # E above nu at theta < th0, below at theta > th0
        if (side>0. and th0==0.) or (side<0. and th0==np.pi): continue
        e = T*_PANELS/a
        off = _rise(xc,e) if side>0. else _fall(xc,e) # offsets from th0
        d0,d1 = off[:-1],off[1:]
        ok = d1>d0
        if not np.any(ok): continue
        c,hw = (d0[ok]+d1[ok])/2.,(d1[ok]-d0[ok])/2.
        dl = (c[:,None]+hw[:,None]*xg[None,:]).ravel()
        wt = (hw[:,None]*wg[None,:]).ravel()
        th = th0-side*dl
        y = 2.*a*np.sin(th0-side*dl/2.)*np.sin(side*dl/2.)+y0
        out += side*np.sum(wt*_fermi_x(side*y/T))
    return float(out)


def _thermal_moments(ne,fermi,w,a,b,T,dth_w):
    """Exact cosine moments R_k = int_0^pi r(theta) cos(k theta) dtheta,
    k < ne, of the thermal factor r = [f(E) - f(E+w)]/w at E = a cos(theta)
    + b, with dth_w the moment k=0 of the T=0 box (see _box_weights).

    For k >= 1, an integration by parts leaves the derivative of r,

      R_k = -(1/k) int_0^pi g(theta) sin(k theta) dtheta,
      g = dr/dtheta = a sin(theta) [p(E - fermi) - p(E - fermi + w)]/w,

    a bump of width T around each edge, which decays as exp(-|E-edge|/T).
    g sin(k theta) is analytic and even about 0 and pi, so the midpoint
    rule on a grid of step pi/N is exact up to exp(-(2N - k) d), with d
    the distance of the nearest pole of p (at E = edge + i pi T) from the
    real theta axis. N = (ne + 45/d)/2 makes this exp(-45) for every
    k < ne, and only the nodes within 44 T of an edge are summed: at most
    a few hundred nodes per edge, whatever T, because a small T makes the
    grid finer and the bump narrower in the same proportion. The sums
    over the nodes for all k at once are a chirp z-transform (_exp_sums).
    At w = 0, r = p is itself the bump, and R_k is summed directly.

    For k = 0 there is no such form, and R_0 is the box moment plus the
    excess of the Fermi function over the step at the two edges,
    R_0 = dth_w + [G(fermi) - G(fermi-w)]/w (_fermi_excess)"""
    R = np.zeros(ne)
    edges = [fermi] if w==0. else [fermi,fermi-w]
    d = min(abs(np.arccos(complex((nu-b)/a,np.pi*T/a)).imag) for nu in edges)
    N = (ne+_ALIAS/d)/2.+1.
    if N<2.**52: N = int(np.ceil(N))
    h = np.pi/N # the step of the fine grid
    Y = _SUPPORT*T/a # half width of a bump, in x
    # the bump of each edge as offsets from its angle th0, so a bump far
    # narrower than the rounding of th0 keeps its nodes; pth0 = pi - th0
    bumps = []
    for nu in edges:
        x = (nu-b)/a
        if x-Y>=1. or x+Y<=-1.: continue # the bump misses [-1,1]
        xc = min(max(x,-1.),1.)
        bumps.append(dict(nu=nu,xc=xc,th0=np.arccos(xc),pth0=np.arccos(-xc),
            up=float(_rise(xc,max((x-xc)+Y,0.))),
            down=float(_fall(xc,max((xc-x)+Y,0.))),
            at0=x+Y>=1.,atpi=x-Y<=-1.)) # touching theta = 0 or pi
    if len(bumps)==2 and max(bumps[0]["th0"]-bumps[0]["up"],
            bumps[1]["th0"]-bumps[1]["up"]) <= min(
            bumps[0]["th0"]+bumps[0]["down"],
            bumps[1]["th0"]+bumps[1]["down"]): # overlapping, one grid
        p0,p1 = bumps
        # the angle of fermi - w from fermi, without cancellation
        dw = float(_fall(p0["xc"],w/a) if w>0. else -_rise(p0["xc"],-w/a))
        bumps = [dict(nu=fermi,th0=p0["th0"],
            lo=min(-p0["up"],dw-p1["up"]),hi=max(p0["down"],dw+p1["down"]),
            hi0=max(p0["th0"]+p0["down"],p1["th0"]+p1["down"]),
            lopi=max(p0["pth0"]+p0["up"],p1["pth0"]+p1["up"]),
            at0=p0["at0"] or p1["at0"],atpi=p0["atpi"] or p1["atpi"])]
    else:
        for p in bumps:
            p.update(lo=-p["up"],hi=p["down"],hi0=p["th0"]+p["down"],
                    lopi=p["pth0"]+p["up"])
    k = np.arange(ne)
    X = np.zeros(ne,dtype=np.complex128) # sum_m g_m exp(i k theta_m)
    for p in bumps:
        nu = p["nu"]
        # the grid, y = E - nu at its nodes without cancellation, and the
        # phase exp(i k theta) of its first node
        if p["at0"] and p["atpi"]: # the whole circle, N midpoints
            m = np.arange(N)
            th = h*(m+0.5)
            y = a*np.cos(th)+b-nu
        elif p["at0"]: # from theta = 0, symmetric about it
            m = np.arange(int(np.ceil(p["hi0"]/h))+1)
            th = h*(m+0.5)
            y = -2.*a*np.sin(th/2.)**2+(a+b-nu)
        elif p["atpi"]: # from theta = pi, symmetric about it
            m = np.arange(int(np.ceil(p["lopi"]/h))+1)
            ep = h*(m+0.5)
            th = np.pi-ep
            y = 2.*a*np.sin(ep/2.)**2+(b-a-nu)
        else: # around the edge, which is inside
            th0 = p["th0"]
            m0 = int(np.floor(p["lo"]/h))
            m = np.arange(int(np.ceil(p["hi"]/h))-m0+1)
            dl = h*(m+m0)
            th = th0+dl
            y = -2.*a*np.sin(th0+dl/2.)*np.sin(dl/2.)
        # E - fermi and E - fermi + w, each without the rounding of the
        # other edge
        if nu==fermi: y,yw = y,y+w
        else: y,yw = y-w,y
        if w==0.: g = _thermal_ratio(y,0.,T)
        else: g = a*np.sin(th)*_thermal_slope(y,w,T,yw=yw)
        S = _exp_sums(g,N,ne) # sum_m g_m exp(i pi k m/N)
        if p["atpi"] and not p["at0"]: # theta_m = pi - h (m + 1/2)
            X += (-1.)**k*np.exp(-1j*k*h/2.)*np.conj(S)
        else: X += np.exp(1j*k*th[0])*S
    if w==0.: return h*X.real
    R[1:] = -h*X[1:].imag/k[1:]
    a,b,T = float(a),float(b),float(T) # the cache keys of _fermi_excess
    G0 = _fermi_excess(float(fermi),a,b,T)
    G1 = _fermi_excess(float(fermi-w),a,b,T)
    R[0] = dth_w+(G0-G1)/w
    return R


@functools.lru_cache(maxsize=8)
def _node_window(ne,the):
    """_edge_window(theta_j, the, 6 pi/ne) on the ne nodes, evaluated only
    within 13 tau of the edge and of its images, since it is exactly zero
    beyond 12 tau. Cached, so the window of the Fermi level is built once
    for all the frequencies; the array is not to be modified"""
    th = np.pi*(np.arange(ne)+0.5)/ne
    tau = 6.*np.pi/ne
    m = 13.*tau
    near = (np.abs(th-the)<m)|(th+the<m)|(2.*np.pi-th-the<m)
    out = np.zeros(ne)
    out[near] = _edge_window(th[near],the,tau)
    out.setflags(write=False)
    return out


def _thermal_weights(ne,fermi,w,a,b,T):
    """The weights q_j of _box_weights at T > 0: the window of T=0 at the
    two edges, the exact cosine moments of r (_thermal_moments) inside it
    and the midpoint rule r(E_j) pi/ne outside it. A |w| below 1e-10 is
    taken as the limit w = 0, r = -df/dE, as the midpoint rule always did"""
    from scipy.fft import dct
    if abs(w)<=1e-10: w = 0.
    lo,hi = fermi-max(w,0.),fermi-min(w,0.) # the edges in energy
    xa = min(max((lo-b)/a,-1.),1.) # cut to [-1,1]
    xb = min(max((hi-b)/a,-1.),1.)
    dth_w = 0.
    if w!=0. and xa<xb: # the width of the box in theta, as at T=0
        cut = (lo-b)/a<-1. or (hi-b)/a>1.
        d = xb-xa if cut else abs(w)/a
        dth_w = 2.*np.arctan2(d,np.sqrt(1.-xb**2)+np.sqrt(1.-xa**2))/abs(w)
    k = np.arange(ne)
    th = np.pi*(k+0.5)/ne # the nodes in theta
    R = _thermal_moments(ne,fermi,w,a,b,T,dth_w)
    W = dct(R,type=3)/ne # the interpolant times r, integrated exactly
    r = _thermal_ratio(a*np.cos(th)+b-fermi,w,T) # [f(E) - f(E+w)]/w
    # the window at the two edges, the Fermi level and the Fermi level - w
    t1,t2 = [float(np.arccos(min(max((nu-b)/a,-1.),1.)))
            for nu in (fermi,fermi-w)]
    psi = 1.-(1.-_node_window(int(ne),t1))*(1.-_node_window(int(ne),t2))
    q = psi*W + (1.-psi)*r*(np.pi/ne)
    # far from the edges, where r is negligible, as in the midpoint rule
    q[(psi<1e-16)&(np.abs(r)<=1e-14*np.max(np.abs(r))+1e-300)] = 0.
    return q


def absorptive(st,mu,energies,fermi=0.,T=0.,ne=None):
    """Absorptive (Hermitian) part of sigma_ab(w), (nw,3,3), from one set of
    moments (see the module docstring). The occupation factor
    [f(E) - f(E+w)]/w is integrated exactly over the Chebyshev interpolant
    of the rest of the integrand near the Fermi edges and by the midpoint
    rule on the nodes away from them (_box_weights), at T=0 and at any
    finite T. So any w works, w=0 included, and the result does not depend
    on ne also when T is below the spacing of the nodes, about pi a/ne
    (delta/7 for the default ne). A T of several spacings, as the default
    T = delta, gives the midpoint rule on the nodes to rounding"""
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
        E2 = E + w
        keep = inside & (E2>st.emin-pad) & (E2<st.emax+pad) & \
                (np.abs(E2-b)<a*(1.-1e-9))
        # the occupation factor, integrated over the interpolant of the rest
        q = _box_weights(ne,fermi,w,a,b,T=max(T,0.))
        keep = keep & (q!=0.)
        if not np.any(keep): continue
        idx = np.nonzero(keep)[0]
        x2 = (E2[idx]-b)/a
        # phi_n(E+w) = (2 - delta_n0) T_n(x2)/(pi a sqrt(1-x2^2)), times the
        # weight q_j of pi int dE on the Chebyshev-Gauss nodes, which away
        # from the Fermi edges is r_j pi/ne, the midpoint rule in
        # theta = arccos(x)
        wts = q[idx]/np.sqrt(1.-x2**2)
        s = _contract(x2,idx,wts,Y).reshape((nd,nd))
        for ia,al in enumerate(st.dirs):
            for ib,be in enumerate(st.dirs): out[iw,al,be] = s[ia,ib]
    return out


def _kk_weights(grid,ws,h):
    """Weights K[i,j] of the principal value integral
    P int F(x)/(w_i - x) dx = sum_j K[i,j] F(x_j) for F sampled on the
    uniform grid x_j of spacing h and linear between the nodes. Each hat
    function integrates to [g(d+h) - 2 g(d) + g(d-h)]/h with d = w - x_j and
    g(u) = u ln|u|, which is finite everywhere, so a frequency on a node
    needs no special care"""
    def g(u):
        with np.errstate(divide="ignore",invalid="ignore"):
            out = u*np.log(np.abs(u))
        return np.where(u==0.,0.,out)
    d = np.asarray(ws,dtype=np.float64)[:,None] - grid[None,:]
    return (g(d+h) - 2.*g(d) + g(d-h))/h


def reactive_part(st,mu,energies,fermi=0.,T=0.,ne=None,spacing=None):
    """Anti-Hermitian (reactive) part of sigma_ab(w), (nw,3,3), from one set
    of moments, as the Kramers-Kronig transform of the absorptive part,

      sigma^R_ab(w) = (i/pi) P int dw' sigma^H_ab(w')/(w - w'),

    which makes sigma = sigma^H + sigma^R the retarded function whose
    Hermitian part is exactly absorptive(). The absorptive part is sampled
    on a uniform grid of spacing delta/8 (0.4% error on a Lorentzian line)
    up to the bandwidth, where it vanishes, and continued to negative
    frequencies by sigma^H_ab(-w) = sigma^H_ba(w), which the moments obey
    exactly since mu^ba = (mu^ab)^T. The grid sits at half-integer
    multiples of the spacing, so the grid continued to negative frequencies
    stays uniform across w = 0 and has no node there twice"""
    h = st.delta/8. if spacing is None else float(spacing)
    wmax = (st.emax-st.emin) + 20.*st.delta # absorptive() is zero beyond
    nw = int(np.ceil(wmax/h))
    xs = h*(np.arange(nw+1)+0.5)
    sH = absorptive(st,mu,xs,fermi=fermi,T=T,ne=ne)
    full = np.concatenate([np.transpose(sH[::-1],(0,2,1)),sH]) # -wmax..wmax
    K = _kk_weights(np.concatenate([-xs[::-1],xs]),energies,h)
    out = np.tensordot(K,full,axes=(1,0)) # (nw,3,3)
    return 1j*out/np.pi


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


def _resolvent_batches(n,nc,nz,memory,nb=32):
    """Number of Chebyshev vectors stored in one block of moments, and of
    energies run in one batch, by _imaginary_resolvent within the memory
    budget (GB), as (nb, nzb).

    In units of one set of vectors, 16 n nc bytes, the routine holds at
    once: the block of nb moments; for the batch of nzb energies, the sum
    Y, the product of a block that is added to it, and the previous batch,
    which the caller still holds while the next one is built (3 nzb); the
    three vectors of the recursion; and one more for the work arrays of
    the caller. So 3 nzb + nb + 4 sets must fit in the budget. A budget
    with room to spare keeps nb and runs every energy in one batch, as
    before. A small one gives at most a quarter to the block and the rest
    to the energies, since every batch runs the whole recursion again and
    the size of a block only sets how many moments one matrix product
    takes. Below 8 sets the routine cannot fit, and it runs one energy and
    one moment at a time"""
    slots = int(np.floor(memory*1e9/(16.*n*nc))) # sets in the budget
    nb = int(max(1,min(nb,(slots-4)//4)))
    nzb = int(max(1,min(nz,(slots-4-nb)//3)))
    return nb,nzb


def _imaginary_resolvent(H,D,zs,a,b,memory,nb=32):
    """Generator of (j0, Y), with Y[j] = Im G(zs[j0+j]) D, of shape
    (nz,n,ncol), for G(z) = (z - H)^-1 and the energies in batches that fit
    in the memory budget (GB), see _resolvent_batches.

    With G(z) = (1/a) sum_n (2-delta_n0) q^n/r T_n(H~) (the series and its
    truncation are those of bsetk/chebyshev.py) and T_n(H~) a Hermitian
    polynomial with real coefficients, the Hermitian operator
    Im G = (G - G^dagger)/2i is the series with the real coefficients
    Im[(2-delta_n0) q^n/(a r)], so one recursion serves every energy of a
    batch. No kernel is needed: Im z damps the series by itself"""
    from ..bsetk.chebyshev import _coefficients,_series,number_of_moments
    n,nc = D.shape
    N = number_of_moments(zs,b,a)
    nb,nzb = _resolvent_batches(n,nc,len(zs),memory,nb=nb)
    buf = np.empty((nb,n*nc),dtype=np.complex128)
    for j0 in range(0,len(zs),nzb):
        z = zs[j0:j0+nzb]
        r,q = _coefficients(z,b,a)
        Y = np.zeros((n*nc,len(z)),dtype=np.complex128)
        i,n0 = 0,0
        for t in _chebyshev_vectors(H,D,a,b,N):
            buf[i] = t.ravel()
            i += 1
            if i==nb: # a block of moments, summed as one matrix product
                Y += buf.T@(_series(r,q,np.arange(n0,n0+nb)).imag/a)
                n0 += nb
                i = 0
        if i>0: Y += buf[:i].T@(_series(r,q,np.arange(n0,n0+i)).imag/a)
        yield j0,np.transpose(Y.reshape((n,nc,len(z))),(2,0,1))


def singleshot(st,fermis,eta,ntries=10,trace="stochastic",seed=None,
        memory=2.):
    """Generator of the longitudinal Kubo-Greenwood conductivity at zero
    temperature, one array (len(fermis),nd) per random vector,

      sigma_aa(E) = pi Tr[v_a delta(E-H) v_a delta(E-H)]/V
                  = (1/pi) Tr[v_a Im G v_a Im G]/V,  G = (E + i eta - H)^-1,

    the w -> 0, T -> 0 limit of the absorptive part with each delta a
    Lorentzian of half width eta (the single-shot method of Ferreira and
    Mucciolo, PRL 115, 106601, arXiv:1507.00488). Per random vector r, with
    A = Im G v_a r and B = Im G r, the trace is <A|v_a B>, so the cost is
    the recursion of nd+1 vectors per energy batch, linear in the number
    of moments, which grows as a/eta"""
    parallel.set_num_threads()
    rng = np.random.default_rng(seed)
    zs = np.asarray(fermis,dtype=np.float64) + 1j*eta
    nd = len(st.dirs)
    probes = [_probe_blocks(st.ops.n,trace,ntries,rng) for k in st.ks]
    Hs = [_CSR(st.ops.hamiltonian(k)) for k in st.ks]
    vs = [[_CSR(st.ops.velocity(k,al)) for al in st.dirs] for k in st.ks]
    for blocks in zip(*probes): # loop over samples
        out = np.zeros((len(zs),nd))
        for ik in range(len(st.ks)):
            for X in blocks[ik]:
                ncol = X.shape[1]
                # one recursion for (v_1 X, ..., v_nd X, X)
                D = np.concatenate([v.dot(X) for v in vs[ik]]+[X],axis=1)
                for j0,Y in _imaginary_resolvent(Hs[ik],D,zs,st.a,st.b,
                        memory):
                    for j in range(len(Y)):
                        B = np.ascontiguousarray(Y[j][:,nd*ncol:])
                        for ia in range(nd):
                            A = Y[j][:,ia*ncol:(ia+1)*ncol]
                            vB = vs[ik][ia].dot(B)
                            out[j0+j,ia] += np.vdot(A,vB).real
        yield out/(np.pi*st.V)


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
        intraband=True,interband=True,reactive=True):
    """
    integration="kpm"
    -----------------
    For a large sparse Hamiltonian (10^4 to 10^6 orbitals: a big supercell
    at the Gamma point or a few k-points, or a finite flake), the
    Chebyshev double expansion of conductivitytk/kpmkubo.py. It never
    diagonalizes, and its cost and memory are linear in the number of
    orbitals. It returns the full complex tensor sigma_ab(w), as the band
    modes do. The absorptive (Hermitian) part,
    sigma^H_ab = (sigma_ab + sigma_ba^*)/2, comes from the double moments
    (Re sigma_aa(w) on the diagonal, i Im sigma^A_ab(w), the circular
    dichroism, off the diagonal), and the reactive part (Im sigma_aa(w),
    the finite-frequency Hall conductivity Re sigma_xy(w)) from its
    Kramers-Kronig transform over all frequencies, so the two are
    consistent by construction. The DC Hall conductivity is also
    kpmkubo.dc_conductivity (h.get_dc_conductivity). Keywords of this
    mode:

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
      temperature, default delta. The occupation factor is integrated
      exactly near the Fermi level at every T >= 0, so any frequency works,
      w=0 included (the top of the Drude peak of a metal), also when T is
      below the spacing of the energy nodes, about delta/7
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
    reactive : bool
      add the reactive part (default True). It samples the absorptive
      part on a grid of spacing delta/8 over the whole spectrum, once for
      the mean of the moments, or once per random vector with
      return_error=True. reactive=False returns the absorptive part
      alone, at the cost of the requested frequencies only
    """
    _check_channels(intraband,interband)
    _check_trace(trace)
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta # same default as kubo.optical_conductivity
    st = _Setup(h,nk=nk,delta=delta,npol=npol,kernel=kernel,
            lorentz_lambda=lorentz_lambda,scale=scale,volume=volume)
    samples,musum = [],0.
    for mu in moments(st,ntries=ntries,trace=trace,seed=seed,memory=memory):
        s = absorptive(st,mu,energies,fermi=fermi,T=T,ne=ne)
        # the transform is linear in the moments, so without error bars it
        # is done once, on their mean
        if reactive and return_error:
            s = s + reactive_part(st,mu,energies,fermi=fermi,T=T,
                    ne=ne)
        samples.append(s)
        musum = musum + mu
    sigma,err = _mean_and_error(samples)
    if reactive and not return_error:
        sigma = sigma + reactive_part(st,musum/len(samples),
                energies,fermi=fermi,T=T,ne=ne)
    if component is not None:
        from .kubo import _index
        i,j = _index(component[0]),_index(component[1])
        sigma,err = sigma[:,i,j],err[:,i,j]
    if return_error: return energies,sigma,err
    return energies,sigma


def _dc_bastin(h,fermi,nk,T,delta,npol,ntries,kernel,lorentz_lambda,trace,
        seed,scale,memory,volume,ne):
    """Samples of the Kubo-Bastin tensor, (len(fermi),3,3) each"""
    if T is None: T = delta
    if kernel is None: kernel = "jackson"
    st = _Setup(h,nk=nk,delta=delta,npol=npol,kernel=kernel,
            lorentz_lambda=lorentz_lambda,scale=scale,volume=volume)
    return [bastin(st,mu,fermi,T=T,ne=ne)
            for mu in moments(st,ntries=ntries,trace=trace,seed=seed,
                memory=memory)]


def _dc_singleshot(h,fermi,nk,T,delta,npol,ntries,kernel,lorentz_lambda,
        trace,seed,scale,memory,volume,ne):
    """Samples of the single-shot Kubo-Greenwood tensor, (len(fermi),3,3)
    each, with the longitudinal components on the diagonal and NaN off it"""
    if T is not None and T!=0.: raise NotImplementedError(
        "mode='singleshot' is the zero-temperature Kubo-Greenwood formula, "
        "so T must be None or 0, got T="+str(T)+"; use mode='bastin' for a "
        "finite temperature")
    for name,value in [("npol",npol),("kernel",kernel),("ne",ne)]:
        if value is not None: raise ValueError(
            name+" does not apply to mode='singleshot', whose resolution is "
            "delta alone (each delta function is a Lorentzian of half width "
            "delta/2, with no kernel); got "+name+"="+str(value))
    if not np.isfinite(delta) or delta<=0.: raise ValueError(
        "delta is the half width of the line of a transition and must be a "
        "finite positive number, got "+str(delta))
    st = _Bounds(h,nk=nk,scale=scale,volume=volume)
    fermis = np.atleast_1d(np.array(fermi,dtype=np.float64))
    out = []
    for s in singleshot(st,fermis,delta/2.,ntries=ntries,trace=trace,
            seed=seed,memory=memory):
        sigma = np.full((len(fermis),3,3),np.nan)
        for i in range(3): sigma[:,i,i] = 0. # no current along i
        for ia,al in enumerate(st.dirs): sigma[:,al,al] = s[:,ia]
        out.append(sigma)
    return out


_dc_modes = {"bastin":_dc_bastin,"singleshot":_dc_singleshot}


def get_dc_mode_names():
    """Accepted values of mode= in dc_conductivity"""
    return list(_dc_modes)


def dc_conductivity(h,fermi=0.,nk=1,T=None,delta=0.1,npol=None,ntries=10,
        kernel=None,lorentz_lambda=3.,trace="stochastic",seed=None,
        scale=None,memory=2.,volume=None,ne=None,component=None,
        return_error=False,mode="bastin"):
    """DC conductivity tensor sigma_ab(fermi,T) of a large sparse
    Hamiltonian from a Chebyshev expansion, by one of two routes.

    mode="bastin" (default) is the Kubo-Bastin formula with a double
    expansion (Garcia, Covaci and Rappoport, PRL 114, 116602 (2015)). It
    holds the Fermi sea part, so sigma_xy gives the quantized Hall
    conductivity of a Chern insulator, sigma_xy 2 pi = -C in units of
    e^2/hbar (the sign convention of conductivity.py). sigma_xx is the DC
    longitudinal conductivity, limited by the KPM resolution in a clean
    system: the Drude peak of width delta gives D/delta. Its cost grows as
    npol^2.

    mode="singleshot" is the Kubo-Greenwood formula at zero temperature,
    sigma_aa(E) = (1/pi) Tr[v_a Im G v_a Im G]/V with
    G = (E + i delta/2 - H)^-1, from one Chebyshev series per Fermi level
    with the exact coefficients of the resolvent (Ferreira and Mucciolo,
    arXiv:1507.00488). It gives the longitudinal components only (the
    off-diagonal ones are NaN), at a cost linear in the number of moments,
    so it reaches a small delta at a few Fermi levels. Each delta function
    is a Lorentzian of half width delta/2, so delta keeps its meaning, the
    half width of the line of a transition, and a clean metal gives D/delta
    as in the band modes. It never goes through the Fermi sea integral of
    the Bastin formula.

    Parameters
    ----------
    mode : str
      "bastin" or "singleshot", see above
    fermi : float or array
      Fermi level(s). The moments do not depend on it, so an array costs
      almost nothing more
    nk : int
      k-points per periodic direction (Gamma-centered mesh); the default 1
      is the Gamma point of a large cell
    T : float
      temperature of the Fermi function, default delta for "bastin"; the
      "singleshot" mode is at zero temperature only
    delta, npol, kernel, lorentz_lambda
      the resolution: npol polynomials (2 npol moments), or the npol for
      which a transition line has half width delta, and the kernel,
      "jackson" (default) or "lorentz". With "lorentz" the Kubo-Bastin
      sigma_xx is wrong (-3 in the gap of a Chern insulator, measured on
      14 November 2026, see future_development/kpm_optical_conductivity.md),
      while sigma_xy is right, so keep the default. The "singleshot" mode
      takes delta alone
    ntries, trace, seed
      number of random vectors, "stochastic" or "exact" trace, and the
      seed of the random vectors
    scale, memory, volume, ne
      half width of the rescaled spectrum (default from Lanczos, padded by
      5%), the memory budget in GB for the stored polynomials (for
      "singleshot", the stored vectors of the Fermi levels, which then run
      in batches), the volume that normalizes a finite system (default 1)
      and the number of energy nodes of the integral (default 4 times the
      moments, "bastin" only)
    component : str or None
      e.g. "xy" to return only that component ("xx", "yy" or "zz" for
      "singleshot")
    return_error : bool
      also return the standard error over the random vectors

    Returns
    -------
    sigma of shape (3,3), or (len(fermi),3,3) for an array of Fermi levels,
    real; with return_error=True the tuple (sigma, error)
    """
    _check_trace(trace)
    if mode not in _dc_modes: raise ValueError("unknown mode "+repr(mode)
            +" for the DC conductivity, accepted values are "
            +", ".join(get_dc_mode_names()))
    if mode=="singleshot" and component is not None and \
            component[0]!=component[1]: raise ValueError(
        "mode='singleshot' gives only the longitudinal components (xx, yy, "
        "zz), not "+repr(component)+"; use mode='bastin' for the Hall part")
    samples = _dc_modes[mode](h,fermi,nk,T,delta,npol,ntries,kernel,
            lorentz_lambda,trace,seed,scale,memory,volume,ne)
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
