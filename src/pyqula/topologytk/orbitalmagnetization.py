"""Orbital magnetization of a finite flake, from a Chebyshev expansion

For a finite system the position operator is well defined, and the total
orbital magnetization is the expectation value of the moment of the
current, in the commutator form used by Vidarte et al. (arXiv:2512.01575),

    M_z = Tr[f(H) m_z] / V,   m_z = -(i/2) (x H y - y H x) = -(1/2)(r x v)_z,

with v = i[H,r] the velocity (hbar = e = 1), the charge of the carriers
taken as -1 (electrons), f the Fermi function and V the volume that
normalizes the result, by default 1 so that M_z is the total moment of
the flake. Since x and y commute, r x [H,r] reduces to xHy - yHx, and
m_z has the sparsity of H, with entries -(i/2) H_ij (x_i y_j - y_i x_j).
The total is independent of the origin of the positions, since
Tr[f(H) v] = i Tr[f(H)[H,r]] = 0; the positions are taken relative to the
centroid of the orbitals, which leaves the exact value unchanged and keeps
the variance of the stochastic trace small.

The route of Vidarte et al. (arXiv:2512.01575) expands the magnetization
spectral density m(E) = Tr[m_z delta(E - H)], with one-index moments
mu_n = Tr[m_z T_n(H~)] from random vectors, so that the cost is linear in
the number of orbitals and in the number of moments, and

    M_z(E_F,T) = int dE f(E) m(E) = sum_n g_n mu_n c_n(E_F,T)

with g_n the Jackson kernel and c_n the Chebyshev coefficients of the Fermi
function. Inside a bulk gap, m(E) = dM_z/dE_F is carried by the edge states
and gives the Chern number through the Streda relation (see
orbital_magnetization). mode="ED" is the same trace from the eigenstates,
for a small flake.
"""
import numpy as np
import scipy.sparse as sp

from .. import parallel
from ..conductivitytk import kpmkubo
from ..kpmtk import kernels as kpmkernels


def _moment_operator(h,ops):
    """The sparse operator m_z = -(i/2)(x H y - y H x) of a finite
    Hamiltonian, with the positions relative to the centroid"""
    rr = kpmkubo._positions(h,ops.n) # refuses a Nambu Hamiltonian
    rr = rr - np.mean(rr,axis=0)
    x,y = rr[:,0],rr[:,1]
    i,j = ops.row,ops.col
    vals = -0.5j*ops.data*(x[i]*y[j] - y[i]*x[j])
    return sp.csr_matrix((vals,(i,j)),shape=(ops.n,ops.n))


def _check_finite(h):
    if h.dimensionality!=0: raise NotImplementedError(
        "the orbital magnetization is computed for a finite flake, where the "
        "position operator is well defined, and this Hamiltonian is periodic "
        "in "+str(h.dimensionality)+" direction(s); build a flake first, for "
        "instance with islands.get_geometry()")


def _magnetization_ed(h,fermis,T,delta,ntries,trace,seed,scale,volume):
    """sum_n f(E_n) <n|m_z|n> / V from a full diagonalization, one sample"""
    st = kpmkubo._Bounds(h,scale=scale,volume=volume)
    H = st.ops.hamiltonian(np.zeros(3)).toarray()
    mz = _moment_operator(h,st.ops).toarray()
    es,U = np.linalg.eigh(H)
    w = np.real(np.einsum("in,ij,jn->n",np.conj(U),mz,U))
    return [np.array([np.sum(kpmkubo._fermi(es,f,T)*w) for f in fermis])
            /st.V]


def _magnetization_kpm(h,fermis,T,delta,ntries,trace,seed,scale,volume):
    """One estimate of M_z(E_F) per random vector, from the moments
    Tr[m_z T_n(H~)] and the Chebyshev coefficients of the Fermi function"""
    parallel.set_num_threads()
    st = kpmkubo._Bounds(h,scale=scale,volume=volume)
    M = 2*kpmkernels.jackson_npol(st.a,delta)
    g = kpmkernels.jackson_kernel(np.ones(M))
    ne = 4*M
    x = np.cos(np.pi*(np.arange(ne)+0.5)/ne) # Chebyshev-Gauss nodes
    f = np.array([kpmkubo._fermi(st.a*x+st.b,mf,T) for mf in fermis]).T
    cf = (kpmkubo._phi(x,M)@f/ne)*g[:,None] # (M,len(fermis))
    H = kpmkubo._CSR(st.ops.hamiltonian(np.zeros(3)))
    mz = kpmkubo._CSR(_moment_operator(h,st.ops))
    rng = np.random.default_rng(seed)
    out = []
    for blocks in kpmkubo._probe_blocks(st.ops.n,trace,ntries,rng):
        mus = np.zeros(M)
        for X in blocks:
            Y = mz.dot(X)
            for n,t in enumerate(kpmkubo._chebyshev_vectors(H,X,st.a,st.b,
                    M)):
                mus[n] += np.vdot(Y,t).real
        out.append((mus@cf)/st.V)
    return out


_modes = {"kpm":_magnetization_kpm,"ED":_magnetization_ed}


def get_mode_names():
    """Accepted values of mode= in orbital_magnetization"""
    return list(_modes)


def orbital_magnetization(h,fermi=0.,T=0.,mode="kpm",delta=0.01,ntries=10,
        trace="stochastic",seed=None,scale=None,volume=None,
        return_error=False):
    """Total orbital magnetization M_z(E_F) of a finite flake, in the
    commutator form M_z = Tr[f(H) m_z]/V with m_z = -(1/2)(r x v)_z, for
    electrons (charge -1, hbar = e = 1), see the module docstring.

    Its derivative dM_z/dE_F is the magnetization spectral density
    m(E) = Tr[m_z delta(E - H)]/V. With the Fermi level inside the bulk
    gap of a Chern insulator, the edge states make it the Streda slope,
    dM_z/dE_F = C A/(2 pi) for a flake of area A and V = 1, with C the
    Chern number of h.get_chern() for the periodic system.

    Parameters
    ----------
    fermi : float or array
      Fermi level(s); the moments do not depend on it, so an array costs
      almost nothing more
    T : float
      temperature of the Fermi function, default 0
    mode : str
      "kpm" (default), the Chebyshev expansion with random vectors, at a
      cost linear in the number of orbitals, or "ED", the eigenstates of a
      small flake
    delta : float
      resolution of the Chebyshev expansion ("kpm"), the half width of the
      peak of a single level, which sets the number of polynomials. The
      Fermi step is smoothed over about delta, so delta should be well
      below the gap or T, default 0.01
    ntries, trace, seed
      number of random vectors, "stochastic" (default) or "exact" trace
      (all the basis vectors, small flakes only), and the seed ("kpm")
    scale : float
      half width of the rescaled spectrum, default from Lanczos padded
      by 5%
    volume : float
      volume that normalizes the result, default 1, the total moment
    return_error : bool
      also return the standard error over the random vectors

    Returns
    -------
    M_z, a float or an array of len(fermi); with return_error=True the
    tuple (M_z, error)
    """
    _check_finite(h)
    if mode not in _modes: raise ValueError("unknown mode "+repr(mode)
            +" for the orbital magnetization, accepted values are "
            +", ".join(get_mode_names()))
    kpmkubo._check_trace(trace)
    fermis = np.atleast_1d(np.array(fermi,dtype=np.float64))
    samples = _modes[mode](h,fermis,T,delta,ntries,trace,seed,scale,volume)
    m,err = kpmkubo._mean_and_error(samples)
    if np.ndim(fermi)==0: m,err = m[0],err[0]
    if return_error: return m,err
    return m
