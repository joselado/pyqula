"""The spin response of the local spin operators of every site, built from
the matrix elements of the sites directly.

chitk.chiAB takes each operator as a dense matrix of the whole unit cell,
one per site and per spin component, and transforms each of them into the
band basis. That is general, any operator goes through it, and it is the
cost of a large cell: with N sites it holds 3N matrices of size 2N x 2N and
does one O(N^3) transform for each. The spin operator of a site is not a
general matrix, it acts on the two spin orbitals of that site only,

    S^a_i = 1/2 sum_{ss'} c^dag_{is} sigma^a_{ss'} c_{is'} ,

so its matrix element between two Bloch states is a product of their
amplitudes on that site,

    <m,k| S^a_i |n,k+q> = 1/2 sum_{ss'} u*_m(is) sigma^a_{ss'} u_n(is') ,

and the response is the contraction of these over the pairs of states,

    chi_{(a,i),(b,j)}(q,w) = 1/Nk sum_k sum_{mn} g_mn(w) X_{mn,(a,i)} X*_{mn,(b,j)} ,

with g_mn(w) the Lindhard factor of chitk.chiAB.lindhard_pair divided by
(e_m - e_n - w + i delta). This is the same sum chitk.chiAB.chiAB_q does
with mode="matrix" for the operators (Sx,Sy,Sz)/2, in the same sign
convention, with the same k-mesh, temperature default and treatment of the
degenerate pairs, so the two agree to roundoff; what changes is that only
the pairs that contribute are kept and each frequency is one matrix
product.

When the Hamiltonian is the same for the two spins and does not mix them,
the response is the same for the three components and has no elements
between different ones, chi_{(a,i),(b,j)} = delta_ab chi^zz_ij, and chi^zz
is half of the bubble of one spin species. The bands are then diagonalized
for one spin only and one N x N block is contracted instead of a 3N x 3N
one. That is the case of a spinon mean field without a Zeeman field
(spinon.SpinonHamiltonian), which is what this module was written for.

The layout of the result is the one of chitk.rpa.chi_ops_RPA with the
operators (Sx,Sy,Sz)/2: the index of component a on site i is a*N + i.
"""

import numpy as np
from scipy.special import expit

from .. import algebra
from .chiAB import DEGENERATE_TOL, PAIR_TOL

# entries of the matrix elements held at once, which bounds the memory of
# the contraction to about 16 bytes times this
_CHUNK_ENTRIES = 2**24


def is_spin_degenerate(h, tol=1e-10):
    """True if the Hamiltonian is the same for the two spins and has no
    matrix element between them: every hopping matrix is the Kronecker
    product of an orbital matrix and the identity in spin."""
    for m in h.get_multihopping().get_dict().values():
        m = np.asarray(algebra.todense(m))
        if np.max(np.abs(m[0::2, 1::2])) > tol: return False
        if np.max(np.abs(m[1::2, 0::2])) > tol: return False
        if np.max(np.abs(m[0::2, 0::2] - m[1::2, 1::2])) > tol: return False
    return True


def _require_normal_spinful(h):
    from .. import check
    check.require_spin(h, "the spin response of the local spin operators")
    if h.has_eh:
        raise NotImplementedError("the spin response of the local spin "
                "operators is not implemented for a Hamiltonian with the "
                "electron-hole (Nambu) degree of freedom")


def _site_list(sites, n):
    """The sites whose spin operators are computed, every one by default"""
    if sites is None: return np.arange(n)
    sites = np.atleast_1d(np.asarray(sites))
    if sites.ndim != 1 or not np.issubdtype(sites.dtype, np.integer):
        raise TypeError("sites must be a list of site indices; got "
                +repr(sites))
    if len(sites) == 0 or np.min(sites) < 0 or np.max(sites) >= n:
        raise ValueError("sites must be indices of the %d sites of the "
                "Hamiltonian (0 to %d); got %s" % (n, n-1, sites.tolist()))
    return sites


def _pair_factors(e1, e2, T, delta):
    """The Lindhard factor and energy difference of every pair of a state
    of energy e1 (at k) and one of energy e2 (at k+q), as
    chitk.chiAB.lindhard_pair gives them for one pair."""
    beta = 1./T
    o1, o2 = expit(-beta*e1), expit(-beta*e2) # Fermi occupations
    de = e1[:, None] - e2[None, :]
    fac = (o1[:, None] - o2[None, :]) + 0j
    degenerate = np.abs(de) < DEGENERATE_TOL
    fp = -beta*o1*(1. - o1) # derivative of the Fermi function
    fac = np.where(degenerate, 1j*delta*fp[:, None], fac)
    de = np.where(degenerate, 0., de)
    return fac, de


def _accumulate(out, X, fac, de, energies, delta, scale=1.):
    """Add to out[w] the contraction of the matrix elements X, of shape
    (n1,n2,M), over the pairs whose factor is not negligible."""
    keep = np.abs(fac) >= PAIR_TOL*delta # as the kernels of chitk.chiAB
    if not np.any(keep): return
    Xs = X[keep] # (pairs, M)
    Xc = Xs.conj()
    XT = Xs.T
    f, d = scale*fac[keep], de[keep]
    for w in range(len(energies)):
        out[w] += (XT*(f/(d - energies[w] + 1j*delta)))@Xc


def _chi_k_degenerate(hk, k, q, energies, delta, T, n, sites):
    """One k-point of chi^zz for a spin-degenerate Hamiltonian, from the
    bands of one spin, for the spin operators of the given sites"""
    e1, u1 = algebra.eigh(algebra.todense(hk(k))[0::2, 0::2])
    e2, u2 = algebra.eigh(algebra.todense(hk(k + q))[0::2, 0::2])
    u1c, u2 = np.conj(np.asarray(u1)).T, np.asarray(u2).T # (state, site)
    u1c, u2 = u1c[:, sites], u2[:, sites] # only the sites asked for
    fac, de = _pair_factors(e1, e2, T, delta)
    m = len(sites)
    out = np.zeros((len(energies), m, m), dtype=np.complex128)
    step = max(1, _CHUNK_ENTRIES//(n*m))
    for i in range(0, n, step): # blocks of the states at k
        X = u1c[i:i+step, None, :]*u2[None, :, :]
        # two spin species, each with (1/2)^2 from the two spin operators
        _accumulate(out, X, fac[i:i+step], de[i:i+step], energies, delta,
                    scale=0.5)
    return out


def _chi_k_spinful(hk, k, q, energies, delta, T, n, sites):
    """One k-point of the full (Sx,Sy,Sz) response for any spinful
    Hamiltonian, for the spin operators of the given sites"""
    e1, u1 = algebra.eigh(algebra.todense(hk(k)))
    e2, u2 = algebra.eigh(algebra.todense(hk(k + q)))
    # (state, site, spin), only the sites asked for
    u1c = np.conj(np.asarray(u1)).T.reshape(2*n, n, 2)[:, sites]
    u2 = np.asarray(u2).T.reshape(2*n, n, 2)[:, sites]
    fac, de = _pair_factors(e1, e2, T, delta)
    m = len(sites)
    out = np.zeros((len(energies), 3*m, 3*m), dtype=np.complex128)
    step = max(1, _CHUNK_ENTRIES//(2*n*3*m))
    for i in range(0, 2*n, step): # blocks of the states at k
        c = u1c[i:i+step]
        uu = c[:, None, :, 0]*u2[None, :, :, 0]
        ud = c[:, None, :, 0]*u2[None, :, :, 1]
        du = c[:, None, :, 1]*u2[None, :, :, 0]
        dd = c[:, None, :, 1]*u2[None, :, :, 1]
        X = np.concatenate([(ud + du)/2., 1j*(du - ud)/2., (uu - dd)/2.],
                           axis=2) # Sx, Sy, Sz of every site
        _accumulate(out, X, fac[i:i+step], de[i:i+step], energies, delta)
    return out


def bare_spin_chi(h, q=None, energies=np.linspace(-3.0, 3.0, 100), nk=60,
                  delta=0.1, T=None, sites=None):
    """Return (energies,chi0): the non-interacting response of the spin
    operators (Sx,Sy,Sz)/2 of every site, one 3N x 3N matrix per energy,
    in the layout and sign convention of chitk.rpa.chi_ops_RPA.

    q is the momentum of the response in units of the reciprocal lattice
    vectors; q=None is the local response, the average over the q of the
    k-mesh. T is the temperature of the occupations, equal to delta when
    not given, as in chitk.chiAB.chiAB_q. See the module docstring for the
    formula and for when the faster spin-degenerate route is taken.

    sites restricts the response to the spin operators of those sites (a
    list of site indices, in that order): one 3M x 3M matrix per energy
    for M sites, with component a of the i-th site of the list at index
    a*M + i. The states of the whole Hamiltonian still enter the bubble.
    Default: every site."""
    return spin_chi(h, V=None, q=q, energies=energies, nk=nk, delta=delta,
                    T=T, sites=sites)


def spin_chi(h, V=None, q=None, energies=np.linspace(-3.0, 3.0, 100), nk=60,
             delta=0.1, T=None, sites=None):
    """Return (energies,chi): the response of the spin operators
    (Sx,Sy,Sz)/2 of every site, dressed with the site-basis vertex V as
    chi0 (1 - V(q) chi0)^-1, the summation chitk.rpa.chi_ops_RPA does.

    V is a real-space vertex {(n1,n2,n3): 3N x 3N matrix}, the one
    chitk.spinchi._full_spin_U builds, or None for the bare response. The
    other arguments are those of bare_spin_chi. With q=None every q of the
    mesh is dressed on its own before the average, which is the local
    response (chitk.rpa._local_rpa_response). With sites given (see
    bare_spin_chi) the response is the one of those sites only, and V is
    in their layout, 3M x 3M."""
    _require_normal_spinful(h)
    energies = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    if q is None: # the local response
        qs = h.geometry.get_kmesh(nk=nk)
        out = [spin_chi(h, V=V, q=qi, energies=energies, nk=nk, delta=delta,
                        T=T, sites=sites)[1] for qi in qs]
        return energies, np.mean(out, axis=0)
    if T is None: T = delta # as chitk.chiAB.chiAB_q
    n = len(h.geometry.r)
    sites = _site_list(sites, n)
    m = len(sites)
    q = np.array(q, dtype=np.float64)
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk)
    if is_spin_degenerate(h):
        zz = np.mean([_chi_k_degenerate(hk, np.array(k), q, energies, delta,
                                        T, n, sites) for k in ks], axis=0)
        chi = np.zeros((len(energies), 3*m, 3*m), dtype=np.complex128)
        for a in range(3): chi[:, a*m:(a+1)*m, a*m:(a+1)*m] = zz
    else:
        chi = np.mean([_chi_k_spinful(hk, np.array(k), q, energies, delta,
                                      T, n, sites) for k in ks], axis=0)
    if V is None: return energies, chi
    from .rpa import interaction_at_q
    Vq = interaction_at_q(V, h, q)
    iden = np.identity(3*m, dtype=np.complex128)
    return energies, np.array([c@algebra.inv(iden - Vq@c) for c in chi])


def static_kernel_eigenvalues(h, V, q, nk=60, delta=1e-3, T=None):
    """Return the eigenvalues, in increasing order of their real part, of
    the static kernel 1 - V(q) chi0(q,0) of the site-basis spin RPA.

    The dressed response is chi0 times the inverse of this kernel, so an
    eigenvalue that reaches zero is a static response that diverges, and
    a negative one means that the state the bands were computed on is
    unstable towards magnetic order at this q. delta is the broadening of
    the bubble, which the static limit wants small."""
    if q is None:
        raise ValueError("the static kernel is a quantity of one momentum, "
                "q=None was given; pass q in units of the reciprocal "
                "lattice vectors")
    es, chi = bare_spin_chi(h, q=q, energies=[0.0], nk=nk, delta=delta, T=T)
    from .rpa import interaction_at_q
    Vq = interaction_at_q(V, h, q)
    ev = np.linalg.eigvals(np.identity(chi[0].shape[0]) - Vq@chi[0])
    return ev[np.argsort(ev.real)]
