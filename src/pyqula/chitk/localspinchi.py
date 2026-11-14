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
A vertex with the same block for the three components and none between
them (the exchange of a spinon mean field) then dresses that block alone.

When the Hamiltonian does not mix the two spins but is different for
them (a Zeeman field along z), S_z is conserved and every Bloch state is
pure up or pure down. A pair of two up or two down states has only an S_z
matrix element, and a pair of an up and a down state only S_x and S_y, with
S_y = -i S_x when the state at k is the up one and S_y = i S_x when it is
the down one. So the bands of the two spins are diagonalized apart and
four N x N blocks are contracted, C^{ss'}_ij = sum over the pairs of a
state of spin s at k and one of spin s' at k+q, with
X = conj(u_s(i)) u_s'(i), and

    chi^zz = (C^uu + C^dd)/4 , chi^xx = chi^yy = (C^ud + C^du)/4 ,
    chi^xy = -chi^yx = i (C^ud - C^du)/4 ,

with no element between z and x or y. A field along another direction is
not rotated to z: it takes the general route.

The local response, q=None, is the average over the q of the k-mesh.
k + q is then a point of the same mesh, since H(k) is periodic in the
reduced k, so every point of the mesh is diagonalized once and the states
at k + q are taken from it.

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
    (n1,n2,M), over the pairs whose factor is not negligible. A pair whose
    factor is exactly zero (two states with the same occupation, or a
    degenerate pair, whose factor i delta f'(e) vanishes at delta=0)
    contributes nothing at any delta, so it is dropped also at delta=0,
    where a frequency equal to its energy difference would otherwise give
    0/0."""
    # as the kernels of chitk.chiAB, plus the pairs of zero factor
    keep = (np.abs(fac) >= PAIR_TOL*delta) & (fac != 0)
    if not np.any(keep): return
    Xs = X[keep] # (pairs, M)
    Xc = Xs.conj()
    XT = Xs.T
    f, d = scale*fac[keep], de[keep]
    for w in range(len(energies)):
        out[w] += (XT*(f/(d - energies[w] + 1j*delta)))@Xc


def _conserves_sz(h, tol=1e-10):
    """True if no hopping matrix has an element between the two spins, so
    that S_z is conserved and every Bloch state is pure up or pure down"""
    for m in h.get_multihopping().get_dict().values():
        m = np.asarray(algebra.todense(m))
        if np.max(np.abs(m[0::2, 1::2])) > tol: return False
        if np.max(np.abs(m[1::2, 0::2])) > tol: return False
    return True


# the three ways of building the bubble, see the module docstring
_DEGENERATE, _COLLINEAR, _SPINFUL = "degenerate", "collinear", "spinful"


def _route(h):
    if is_spin_degenerate(h): return _DEGENERATE
    if _conserves_sz(h): return _COLLINEAR
    return _SPINFUL


def _bands(hk, k, route, sites):
    """The Bloch states at k that a route needs, as a list of (energies,
    amplitudes on the given sites): the bands of one spin for the
    degenerate route and of the up and the down spin for the collinear
    one, with amplitudes of shape (state, site), and the full states for
    the spinful one, with amplitudes of shape (state, site, spin)"""
    m = np.asarray(algebra.todense(hk(k)))
    if route == _SPINFUL:
        e, u = algebra.eigh(m)
        n = m.shape[0]//2
        return [(e, np.asarray(u).T.reshape(2*n, n, 2)[:, sites])]
    if route == _DEGENERATE: blocks = [m[0::2, 0::2]]
    else: blocks = [m[0::2, 0::2], m[1::2, 1::2]]
    out = []
    for b in blocks:
        e, u = algebra.eigh(b)
        out.append((e, np.asarray(u).T[:, sites]))
    return out


def _add_product(out, b1, b2, energies, delta, T, scale):
    """Add to out[w] the contraction of X = conj(u1(i)) u2(i), site by
    site, over the pairs of a state of b1 = (e1, u1) at k and a state of
    b2 = (e2, u2) at k+q, with amplitudes of shape (state, site)"""
    (e1, u1), (e2, u2) = b1, b2
    fac, de = _pair_factors(e1, e2, T, delta)
    u1c = np.conj(u1)
    step = max(1, _CHUNK_ENTRIES//(len(e2)*u1.shape[1]))
    for i in range(0, len(e1), step): # blocks of the states at k
        X = u1c[i:i+step, None, :]*u2[None, :, :]
        _accumulate(out, X, fac[i:i+step], de[i:i+step], energies, delta,
                    scale=scale)


def _add_spinful(out, b1, b2, energies, delta, T, raising=False):
    """Add to out[w] the full (Sx,Sy,Sz) contraction over the pairs of a
    state of b1 at k and a state of b2 at k+q, with amplitudes of shape
    (state, site, spin), or with raising=True the contraction of
    S^+ = Sx + i Sy alone, whose matrix element is conj(u_up) u_down"""
    (e1, u1), (e2, u2) = b1, b2
    fac, de = _pair_factors(e1, e2, T, delta)
    u1c = np.conj(u1)
    ncol = 1 if raising else 3
    step = max(1, _CHUNK_ENTRIES//(len(e2)*ncol*u1.shape[1]))
    for i in range(0, len(e1), step): # blocks of the states at k
        c = u1c[i:i+step]
        ud = c[:, None, :, 0]*u2[None, :, :, 1]
        if raising: X = ud # S^+ of every site
        else:
            uu = c[:, None, :, 0]*u2[None, :, :, 0]
            du = c[:, None, :, 1]*u2[None, :, :, 0]
            dd = c[:, None, :, 1]*u2[None, :, :, 1]
            X = np.concatenate([(ud + du)/2., 1j*(du - ud)/2.,
                                (uu - dd)/2.], axis=2) # Sx, Sy, Sz
        _accumulate(out, X, fac[i:i+step], de[i:i+step], energies, delta)


def _zeros(route, ne, m, raising=False):
    """The bubble of a route before its first k-point: chi^zz, (energies,
    M, M), for the degenerate route, the four blocks C^uu, C^dd, C^ud and
    C^du over 4, (4, energies, M, M), for the collinear one, and the full
    (energies, 3M, 3M) response for the spinful one. With raising=True it
    is chi^{+-}, (energies, M, M), on every route."""
    if raising or route == _DEGENERATE: shape = (ne, m, m)
    elif route == _COLLINEAR: shape = (4, ne, m, m)
    else: shape = (ne, 3*m, 3*m)
    return np.zeros(shape, dtype=np.complex128)


def _add_k(acc, route, b1, b2, energies, delta, T, raising=False):
    """Add to the bubble acc of a route the pairs of the states b1 at k
    and b2 at k+q, see _bands. With raising=True only chi^{+-} is built:
    twice chi^zz on the degenerate route and C^ud on the collinear one."""
    if route == _DEGENERATE:
        # two spin species, each with (1/2)^2 from the two spin operators
        _add_product(acc, b1[0], b2[0], energies, delta, T,
                     scale=1. if raising else 0.5)
    elif route == _COLLINEAR:
        (up1, dn1), (up2, dn2) = b1, b2
        if raising: # an up state at k and a down one at k+q
            _add_product(acc, up1, dn2, energies, delta, T, scale=1.)
            return
        for c, pair in enumerate([(up1, up2), (dn1, dn2), (up1, dn2),
                                  (dn1, up2)]):
            _add_product(acc[c], pair[0], pair[1], energies, delta, T,
                         scale=0.25)
    else:
        _add_spinful(acc, b1[0], b2[0], energies, delta, T, raising)


def _spin_matrix(route, acc, m):
    """The response of (Sx,Sy,Sz)/2, one 3M x 3M matrix per energy, from
    the bubble of a route (module docstring)"""
    if route == _SPINFUL: return acc
    chi = np.zeros((acc.shape[-3], 3*m, 3*m), dtype=np.complex128)
    x, y, z = slice(0, m), slice(m, 2*m), slice(2*m, 3*m)
    if route == _DEGENERATE:
        for s in (x, y, z): chi[:, s, s] = acc
        return chi
    uu, dd, ud, du = acc # each over 4
    chi[:, z, z] = uu + dd
    chi[:, x, x] = ud + du
    chi[:, y, y] = ud + du
    chi[:, x, y] = 1j*(ud - du)
    chi[:, y, x] = -1j*(ud - du)
    return chi


def _bubble_at_q(h, route, q, energies, nk, delta, T, sites, raising=False):
    """The bubble of a route at the momentum q, a running sum over the
    k-mesh, see _zeros"""
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk)
    acc = _zeros(route, len(energies), len(sites), raising)
    for k in ks:
        k = np.array(k)
        _add_k(acc, route, _bands(hk, k, route, sites),
               _bands(hk, k + q, route, sites), energies, delta, T, raising)
    acc /= len(ks)
    return acc


def _bubbles_on_mesh(h, route, energies, nk, delta, T, sites,
                     raising=False):
    """Yield (q, bubble of a route at q) for every q of the k-mesh, with
    every point of the mesh diagonalized once: H(k) is periodic in the
    reduced k, so the states at k+q are those of the point of the mesh
    that k+q is, up to a vector of the reciprocal lattice"""
    from ..kpointstk.kmesh import nk_per_direction
    hk = h.get_hk_gen()
    ks = np.array(h.geometry.get_kmesh(nk=nk), dtype=np.float64)
    N = np.ones(3, dtype=int) # points of the mesh in each direction
    dim = h.geometry.dimensionality
    if dim > 0: N[:dim] = nk_per_direction(nk, dim)
    def point(k): return tuple(np.round(k*N).astype(int) % N)
    index = {point(k): i for i, k in enumerate(ks)}
    bands = [_bands(hk, k, route, sites) for k in ks]
    for q in ks:
        acc = _zeros(route, len(energies), len(sites), raising)
        for (k, b1) in zip(ks, bands):
            _add_k(acc, route, b1, bands[index[point(k + q)]], energies,
                   delta, T, raising)
        acc /= len(ks)
        yield q, acc


def _common_block(Vq, m):
    """The M x M block of a vertex Vq (3M x 3M) that is the same for the
    three spin components and has nothing between them, or None when Vq
    is not of that form"""
    b = Vq[:m, :m]
    for a in range(3):
        for c in range(3):
            block = Vq[a*m:(a+1)*m, c*m:(c+1)*m]
            if a == c and not np.array_equal(block, b): return None
            if a != c and np.any(block != 0.): return None
    return b


def _dyson(chi, Vq):
    """chi (1 - Vq chi)^-1 at every energy, written into a new array"""
    iden = np.identity(chi.shape[-1], dtype=np.complex128)
    out = np.empty(chi.shape, dtype=np.complex128)
    for w, c in enumerate(chi):
        # X = c (1 - Vq c)^-1 solves (1 - Vq c)^T X^T = c^T
        out[w] = np.linalg.solve((iden - Vq@c).T, c.T).T
    return out


def _dressed(h, route, acc, V, q, m):
    """The response of (Sx,Sy,Sz)/2 at q from the bubble of a route,
    dressed with the real-space vertex V when it is given. A degenerate
    bubble and a vertex with one block for the three components are
    dressed block by block, which is the same result"""
    if V is None: return _spin_matrix(route, acc, m)
    from .rpa import interaction_at_q
    Vq = np.asarray(interaction_at_q(V, h, q))
    if route == _DEGENERATE:
        b = _common_block(Vq, m)
        if b is not None: return _spin_matrix(route, _dyson(acc, b), m)
    return _dyson(_spin_matrix(route, acc, m), Vq)


def _arguments(h, energies, delta, T, sites):
    """The checked arguments of the bubble and its route"""
    _require_normal_spinful(h)
    energies = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    if T is None: T = delta # as chitk.chiAB.chiAB_q
    sites = _site_list(sites, len(h.geometry.r))
    return energies, T, sites, _route(h)


def bare_spin_chi_on_mesh(h, energies=np.linspace(-3.0, 3.0, 100), nk=60,
                          delta=0.1, T=None, sites=None):
    """Yield (q, chi0) for every q of the k-mesh, with chi0 the bare
    response of bare_spin_chi at q, for a caller that dresses each q on its
    own before the average of the local response. Every point of the mesh
    is diagonalized once (module docstring). The arguments are those of
    bare_spin_chi."""
    energies, T, sites, route = _arguments(h, energies, delta, T, sites)
    for q, acc in _bubbles_on_mesh(h, route, energies, nk, delta, T, sites):
        yield q, _spin_matrix(route, acc, len(sites))


def bare_raising_chi(h, q=None, energies=np.linspace(-3.0, 3.0, 100), nk=60,
                     delta=0.1, T=None, sites=None):
    """Return (energies,chi): the bare transverse response
    chi^{+-} = chi^xx + chi^yy - i chi^xy + i chi^yx of the spin operators
    of the sites, one M x M matrix per energy, with S^+ = Sx + i Sy on the
    first index and S^- on the second. It is the combination of the blocks
    of bare_spin_chi, built from the matrix element of S^+ alone, which is
    a third of the columns of (Sx,Sy,Sz). The arguments are those of
    bare_spin_chi."""
    energies, T, sites, route = _arguments(h, energies, delta, T, sites)
    if q is None: # the local response
        out, nq = 0., 0
        for qi, acc in _bubbles_on_mesh(h, route, energies, nk, delta, T,
                                        sites, raising=True):
            out = out + acc
            nq += 1
        return energies, out/nq
    acc = _bubble_at_q(h, route, np.array(q, dtype=np.float64), energies, nk,
                       delta, T, sites, raising=True)
    return energies, acc


def bare_spin_chi(h, q=None, energies=np.linspace(-3.0, 3.0, 100), nk=60,
                  delta=0.1, T=None, sites=None):
    """Return (energies,chi0): the non-interacting response of the spin
    operators (Sx,Sy,Sz)/2 of every site, one 3N x 3N matrix per energy,
    in the layout and sign convention of chitk.rpa.chi_ops_RPA.

    q is the momentum of the response in units of the reciprocal lattice
    vectors; q=None is the local response, the average over the q of the
    k-mesh. T is the temperature of the occupations, equal to delta when
    not given, as in chitk.chiAB.chiAB_q. See the module docstring for the
    formula and for the faster routes of a Hamiltonian that does not mix
    the two spins. With delta=0 the bubble is evaluated on the real axis:
    a pair of zero factor (two states with the same occupation) is left
    out, so an energy equal to its energy difference is finite, and an
    energy equal to the difference of a pair that does contribute is a
    pole, where the result is not finite.

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
    energies, T, sites, route = _arguments(h, energies, delta, T, sites)
    m = len(sites)
    if q is None: # the local response, every q dressed on its own
        out, nq = None, 0
        for qi, acc in _bubbles_on_mesh(h, route, energies, nk, delta, T,
                                        sites):
            chi = _dressed(h, route, acc, V, qi, m) # a new array at each q
            if out is None: out = chi
            else: out += chi
            nq += 1
        out /= nq
        return energies, out
    q = np.array(q, dtype=np.float64)
    acc = _bubble_at_q(h, route, q, energies, nk, delta, T, sites)
    return energies, _dressed(h, route, acc, V, q, m)


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
    energies, T, sites, route = _arguments(h, [0.0], delta, T, None)
    n = len(sites)
    q = np.array(q, dtype=np.float64)
    acc = _bubble_at_q(h, route, q, energies, nk, delta, T, sites)
    from .rpa import interaction_at_q
    Vq = np.asarray(interaction_at_q(V, h, q))
    b = _common_block(Vq, n) if route == _DEGENERATE else None
    if b is not None: # the eigenvalues of one block, once per component
        ev = np.repeat(np.linalg.eigvals(np.identity(n) - b@acc[0]), 3)
    else:
        chi = _spin_matrix(route, acc, n)
        ev = np.linalg.eigvals(np.identity(3*n) - Vq@chi[0])
    return ev[np.argsort(ev.real)]
