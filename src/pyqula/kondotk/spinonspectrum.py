"""Tunneling spectrum of a tip above one site of a spinon mean field, to
second and third order in the exchange between the tunneling electron and
the spin of the site.

The formulas are those of Ternes, New J. Phys. 17, 063016 (2015),
arXiv:1505.04430, in the form the dmrgpy package evaluates them for the
eigenstates i, f, m of a spin model, with energies E and thermal weights
p_i (both tunneling directions are summed, v = +eV and v = -eV):

    second order   2 pi T0^2 sum_v sum_if p_i sum_a |<f|S_a|i>|^2
                       Theta(v - (E_f - E_i))

    third order    4 pi T0^2 Jrho_s sum_v sum_ifm p_i Im[C_ifm]/2
                       Theta(v - (E_f - E_i))
                       [ F(v - (E_m - E_i)) + F(v - (E_f - E_m)) ]

    C_ifm = sum_jkl eps_jkl <i|S_l|f> <f|S_k|m> <m|S_j|i>

The second order is the steps of inelastic tunneling. The third order is
the Kondo term: a logarithm F at the energy of the virtual state m, in
the direct and in the exchange diagram.

Here the spin is the one of the spinons of a site, S_a = 1/2 f^dag
sigma_a f, and the states are those of a quadratic Hamiltonian, so the
sums over many-body states are done with Wick's theorem. A final state f
is one particle-hole pair. A virtual state m is a pair that shares its
hole with f (the particle is scattered) or its particle with f (the hole
is scattered, with a minus sign from the exchange of two fermions). Only
the amplitudes of the single-particle states on the site of the tip
enter, through the two local spectral matrices of the site,

    O(e) = sum_n occupation_n     u_n u_n^dag delta(e - e_n)
    E(e) = sum_n (1-occupation_n) u_n u_n^dag delta(e - e_n) ,

2 x 2 matrices in spin, with u_n the two amplitudes of state n on the
site. With them

    S(w) = 1/4 sum_a int de Tr[ sigma_a E(e+w) sigma_a O(e) ]

is the spin spectrum of the second order, and the third order has

    particle:  1/8 eps_jkl Tr[ sigma_l E(e_b) sigma_k E(e_d) sigma_j O(e_a) ]
               E_f - E_i = e_b - e_a ,  E_m - E_i = e_d - e_a
    hole:     -1/8 eps_jkl Tr[ sigma_l E(e_b) sigma_j O(e_c) sigma_k O(e_a) ]
               E_f - E_i = e_b - e_a ,  E_m - E_i = e_b - e_c .

Each of them is a product of two functions of one energy, summed over a
third one, so the whole spectrum is a few convolutions on an energy grid
for each site. When the site has a moment <S>, the terms where one of
the three spin operators is replaced by its average are added: they are
the same formulas with the state i in the place of f or m, or with f = m.

Everything is local, so the cost is one diagonalization for each k-point
plus the convolutions of each site, and a cell of a few hundred
inequivalent sites is as easy as a small one.

For one decoupled site (a level at the Fermi energy, half filled) this
gives exactly half of the spectrum of a spin 1/2, at both orders: the
free fermion has an empty and a doubly occupied state, which carry no
spin, and its local moment S^2 is 3/8 against 3/4.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix
from scipy.special import expit

from .. import algebra
from .thermal import step_kernel, log_kernel, _log_antiderivative

# orders of perturbation theory in the exchange with the tunneling
# electron, order= of kondo_spectrum, and whether each has the Kondo term
_ORDERS = {2: False, 3: True}

# largest number of points of the energy grid chosen when de is not given
_MAX_GRID = 16384

# a component of a local spectral matrix, or a moment, below this is zero
_TOL = 1e-12

_SIGMA = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                  dtype=np.complex128) # identity, sigma_x, sigma_y, sigma_z

_EPS = np.zeros((3, 3, 3))
for _i, _j, _k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    _EPS[_i, _j, _k] = 1.
    _EPS[_i, _k, _j] = -1.

_PAULI = _SIGMA[1:]
# traces of the module docstring, for the components (identity, x, y, z)
# of the three spectral matrices, in the order they appear in the trace
_T_PARTICLE = np.einsum('jkl,lab,mbc,kcd,nde,jef,rfa->mnr', _EPS, _PAULI,
                        _SIGMA, _PAULI, _SIGMA, _PAULI, _SIGMA)/64.
_T_HOLE = -np.einsum('jkl,lab,mbc,jcd,nde,kef,rfa->mnr', _EPS, _PAULI,
                     _SIGMA, _PAULI, _SIGMA, _PAULI, _SIGMA)/64.
# two-spin spectrum R_lk(w) = 1/4 int Tr[sigma_l E(e+w) sigma_k O(e)]
_T_PAIR = np.einsum('lab,mbc,kcd,nda->lkmn', _PAULI, _SIGMA, _PAULI,
                    _SIGMA)/16.


def get_order_names():
    """Values accepted by order= of kondo_spectrum"""
    return list(_ORDERS)


# a function on the energy grid is a pair (i0, values): values[j] is the
# function at the energy (i0 + j)*de

def _conv(f, g):
    """(f*g)(x) = sum_n f(n) g(x - n)"""
    return f[0] + g[0], fftconvolve(f[1], g[1])


def _reflect(f):
    """f(-x)"""
    return -(f[0] + len(f[1]) - 1), f[1][::-1]


def _crop(f, lo, hi):
    """The values of f at the points lo..hi, which it must hold"""
    i0, a = f
    if lo < i0 or hi > i0 + len(a) - 1:
        raise RuntimeError("internal error of the energy grid: points "
                "%d..%d asked of a function held on %d..%d"
                % (lo, hi, i0, i0 + len(a) - 1))
    return a[lo - i0:hi - i0 + 1]


def _spectral_bound(h):
    """An upper bound of |energy| of every state, the sum of the spectral
    norms of the hopping matrices"""
    return sum(np.linalg.norm(np.asarray(algebra.todense(m)), 2)
               for m in h.get_multihopping().get_dict().values())


def _occupations(e, T):
    if T > 0.: return expit(-e/T)
    return np.where(e < -1e-9, 1., np.where(e > 1e-9, 0., 0.5))


def local_spectral_components(h, sites, nk, T, de, degenerate=False):
    """Return (n0, occ, emp): the occupied and the empty local spectral
    matrix of every site of sites on the energy grid n*de, n = n0, n0+1,
    ..., as arrays [site, component, energy]. The components are those of
    O = (o_0 + o_x sigma_x + o_y sigma_y + o_z sigma_z)/2, so o_0 is the
    occupied local density of states of the two spins together and o_a
    twice the spin density along a. Each array holds the weight in each
    bin, not a density, so its sum over the energies is the integral.

    With degenerate=True only the first component is computed, which is
    the case of a Hamiltonian that is the same for the two spins. The bands
    of one spin are then enough: every orbital state u comes twice, once
    for each spin, with the same energy, so it has the weight 2|u_i|^2.

    Every state is shared between its two neighboring points of the grid,
    in proportion to the distance, which keeps its energy to second order
    in de."""
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=nk)
    half = int(np.ceil(_spectral_bound(h)/de)) + 2
    n0, L = -half, 2*half + 1
    sites = np.asarray(sites)
    ncomp = 1 if degenerate else 4
    occ = np.zeros((len(sites)*ncomp, L))
    emp = np.zeros((len(sites)*ncomp, L))
    for k in ks:
        m = np.asarray(algebra.todense(hk(np.array(k))))
        if degenerate: # the bands of one spin, each one twice
            es, vs = algebra.eigh(m[0::2, 0::2])
            es, vs = np.asarray(es).real, np.asarray(vs)
            comps = 2.*np.abs(vs[sites, :][:, None, :])**2
        else:
            es, vs = algebra.eigh(m)
            es, vs = np.asarray(es).real, np.asarray(vs)
            a, b = vs[2*sites, :], vs[2*sites + 1, :] # [site, state], up/down
            ab = np.conj(a)*b
            comps = np.stack([np.abs(a)**2 + np.abs(b)**2, 2*ab.real,
                              2*ab.imag, np.abs(a)**2 - np.abs(b)**2],
                             axis=1)
        comps = comps.reshape(len(sites)*ncomp, len(es))
        x = es/de - n0
        j = np.floor(x).astype(int)
        frac = x - j
        ns = np.arange(len(es))
        # share of each state in the two points of the grid next to it
        B = csr_matrix((np.concatenate([1. - frac, frac]),
                        (np.concatenate([ns, ns]),
                         np.concatenate([j, j + 1]))), shape=(len(es), L))
        f = _occupations(es, T)
        occ += (B.T@(comps*f[None, :]).T).T
        emp += (B.T@(comps*(1. - f)[None, :]).T).T
    occ, emp = occ/len(ks), emp/len(ks)
    used = np.nonzero(np.any(occ != 0., axis=0) | np.any(emp != 0., axis=0))[0]
    lo, hi = used[0], used[-1] + 1
    shape = (len(sites), ncomp, hi - lo)
    return n0 + lo, occ[:, lo:hi].reshape(shape), emp[:, lo:hi].reshape(shape)


class _Kernels():
    """The step, the logarithm and their product at the points -K..K"""
    def __init__(self, K, de, T, omega0, Gamma0):
        self.step = (-K, step_kernel(K, de, T))
        self.log = (-K, log_kernel(K, de, T, omega0, Gamma0))
        if T == 0.:
            # the average of step times logarithm over a bin: the
            # logarithm integrated over the positive part of the bin
            y = np.arange(-K, K + 1)*de
            up = np.maximum(y + de/2., 0.)
            down = np.maximum(y - de/2., 0.)
            prod = (_log_antiderivative(up, omega0, Gamma0)
                    - _log_antiderivative(down, omega0, Gamma0))/de
        else:
            prod = self.step[1]*self.log[1]
        self.product = (-K, prod)


def _site_spectrum(n0, occ, emp, Q, ker, third):
    """Second-order and third-order sums of one site at the bias points
    -Q..Q, for the tunneling direction v = +eV; the other direction is
    the same arrays reversed. occ and emp are [component, energy]. The
    third-order one is complex, and its imaginary part is the one used."""
    ncomp, L = occ.shape
    e = [(n0, emp[m]) for m in range(ncomp)]
    o = [(n0, occ[m]) for m in range(ncomp)]
    orefl = [_reflect(f) for f in o]
    # components that are not zero
    act = [m for m in range(ncomp)
           if max(np.max(np.abs(occ[m])), np.max(np.abs(emp[m]))) > _TOL]
    moment = np.zeros(3) # <S_a> of the site
    if ncomp == 4: moment = 0.5*occ[1:].sum(axis=1)
    # spin spectrum: (3/8) e_0 o_0 - (1/8) e_a o_a, from the trace
    pair = {(m, n): _conv(e[m], orefl[n]) for m in act for n in act}
    S = sum((0.375 if m == 0 else -0.125)*pair[(m, m)][1] for m in act)
    S = (-(L - 1), S)
    second = _crop(_conv(S, ker.step), -Q, Q)
    second = second + np.sum(moment**2)*_crop(ker.step, -Q, Q)
    if not third: return second, None
    plus = (n0 - Q, n0 + L - 1 + Q) # range of v + e
    minus = (-Q - (n0 + L - 1), Q - n0) # range of v - e
    def on_plus(f, kernel): return _crop(_conv(f, kernel), *plus)
    def on_minus(f, kernel): return _crop(_conv(_reflect(f), kernel), *minus)
    AE = {m: on_plus(e[m], ker.step) for m in act} # particle of f
    BE = {m: on_plus(e[m], ker.log) for m in act} # particle of m, direct
    CO = {m: on_plus(o[m], ker.log) for m in act} # hole of m, exchange
    AO = {m: on_minus(o[m], ker.step) for m in act} # hole of f
    BO = {m: on_minus(o[m], ker.log) for m in act} # hole of m, direct
    CE = {m: on_minus(e[m], ker.log) for m in act} # particle of m, exchange
    third = np.zeros(2*Q + 1, dtype=np.complex128)
    for r in act: # outer sum over the hole of f, at v + e
        G = sum(_T_PARTICLE[m, n, r]*AE[m]*BE[n]
                + _T_HOLE[m, n, r]*AE[m]*CO[n] for m in act for n in act)
        third += _crop(_conv((plus[0], G), orefl[r]), -Q, Q)
    for m in act: # outer sum over the particle of f, at v - e
        G = sum(_T_PARTICLE[m, n, r]*CE[n]*AO[r]
                + _T_HOLE[m, n, r]*BO[n]*AO[r] for n in act for r in act)
        third += _crop(_conv(e[m], (minus[0], G)), -Q, Q)
    if np.max(np.abs(moment)) > _TOL:
        third += _moment_terms(pair, act, moment, L, Q, ker)
    return second, third


def _moment_terms(pair, act, moment, L, Q, ker):
    """The third-order terms where one of the three spin operators is
    replaced by its average, for a site with a moment"""
    # R[l, k] at the pair energies -(L-1)..(L-1)
    R = sum(_T_PAIR[:, :, m, n][:, :, None]*pair[(m, n)][1][None, None, :]
            for m in act for n in act)
    i0 = -(L - 1)
    W1 = (i0, np.einsum('jkl,l,kjw->w', _EPS, moment, R)) # f = i
    W2 = (i0, np.einsum('jkl,k,ljw->w', _EPS, moment, R)) # f = m
    W3 = (i0, np.einsum('jkl,j,lkw->w', _EPS, moment, R)) # m = i
    step, log = _crop(ker.step, -Q, Q), _crop(ker.log, -Q, Q)
    def at(f, kernel): return _crop(_conv(f, kernel), -Q, Q)
    out = step*(at(W1, ker.log) + at(_reflect(W1), ker.log))
    out = out + at(W2, ker.product) + log*at(W2, ker.step)
    out = out + log*at(W3, ker.step) + at(W3, ker.product)
    return out


def kondo_spectrum(h, energies, site=None, Jrho_s=-0.05, T=0.01,
                   omega0=1.0, Gamma0=1e-5, T0=1.0, order=3, nk=20,
                   de=None, Ts=None):
    """Return (energies, dIdV): the differential conductance of a tip
    above a site of the spinon Hamiltonian h, in units of e^2 T0^2/hbar.

    energies is the bias, in the units of the Hamiltonian, and the Fermi
    energy of the spinons is zero, as a converged mean field has it. site
    is the index of the site under the tip, a list of them, or None for
    every site of the cell; dIdV is one row per site, or one array when
    site is a single index.

    Jrho_s is the dimensionless exchange with the electrons of the
    substrate, in the sign convention of arXiv:1505.04430 (negative for
    an antiferromagnetic one). T is the temperature of the electrodes,
    omega0 the band cutoff and Gamma0 the lifetime broadening of the
    Kondo logarithm, T0 the tunneling strength, which only sets the
    scale. order is 2 for the inelastic steps alone and 3 to add the
    Kondo term. Ts is the temperature of the spinon occupations, T when
    not given. nk is the k-mesh, and de the step of the energy grid, by
    default T/4, made larger if the grid would pass 16384 points; with
    T=0 the functions are averaged over each bin."""
    from .. import check
    from ..chitk.localspinchi import is_spin_degenerate
    what = "the Kondo tunneling spectrum of the spinons"
    check.require_spin(h, what)
    if h.has_eh:
        raise NotImplementedError(what + " is not implemented for a "
                "Hamiltonian with the electron-hole (Nambu) degree of "
                "freedom")
    if order not in _ORDERS:
        raise ValueError("order=%r is not an order of the Kondo tunneling "
                "spectrum, the accepted values are %s"
                % (order, ", ".join(repr(i) for i in get_order_names())))
    if T < 0. or (Ts is not None and Ts < 0.):
        raise ValueError("the temperatures T and Ts cannot be negative, "
                "got T=%r and Ts=%r" % (T, Ts))
    if omega0 <= 0. or Gamma0 <= 0.:
        raise ValueError("the band cutoff omega0 and the broadening Gamma0 "
                "of the Kondo logarithm must be positive, got omega0=%r "
                "and Gamma0=%r" % (omega0, Gamma0))
    if Ts is None: Ts = T
    energies = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    n = len(h.geometry.r)
    single = site is not None and np.ndim(site) == 0
    sites = np.arange(n) if site is None else np.atleast_1d(site).astype(int)
    if np.any(sites < 0) or np.any(sites >= n):
        raise ValueError("site=%r is not a site of this Hamiltonian, which "
                "has the sites 0 to %d" % (site, n - 1))
    vmax = np.max(np.abs(energies))
    if de is None:
        span = 2.*_spectral_bound(h) + 2.*vmax
        de = max(T/4., span/_MAX_GRID)
    if de <= 0.:
        raise ValueError("the step de of the energy grid must be positive, "
                "got de=%r" % (de,))
    n0, occ, emp = local_spectral_components(h, sites, nk, Ts, de,
            degenerate=is_spin_degenerate(h))
    L = occ.shape[2]
    Q = int(np.ceil(vmax/de)) + 1
    third = _ORDERS[order]
    ker = _Kernels(Q + L - 1, de, T, omega0, Gamma0)
    grid = np.arange(-Q, Q + 1)*de
    out = np.zeros((len(sites), len(energies)))
    for i in range(len(sites)):
        s2, s3 = _site_spectrum(n0, occ[i], emp[i], Q, ker, third)
        y = 2.*np.pi*T0**2*(s2 + s2[::-1]) # both tunneling directions
        if third:
            s3 = s3.imag/2.
            y = y + 4.*np.pi*T0**2*Jrho_s*(s3 + s3[::-1])
        out[i] = np.interp(energies, grid, y)
    if single: return energies, out[0]
    return energies, out
