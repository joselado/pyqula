import warnings

import numpy as np
from numba import jit
from scipy.integrate import quad

from .. import algebra
from ..algebra import dagger
from ..transporttk.smatrix import enlarge_hlist
from ..transporttk.fermidirac import fermidirac

# Floquet-Keldysh DC current between two (possibly superconducting) leads,
# following San-Jose, Cayao, Prada, Aguado, New J. Phys. 15, 075019 (2013)
# (arXiv:1301.4408), Appendix A. See also src/pyqula/keldysh.py.
#
# Only heterostructures with NO explicit central region (heterostructures.
# build(h1,h2), the two leads directly weak-linked via set_coupling) are
# supported. Testing found a confirmed, unresolved systematic error (a few
# percent, growing at low transparency) for junctions with an explicit
# central Hamiltonian (a single dense central site or several block-diagonal
# central sites) whenever that region is not structurally identical to a
# lead; _check_supported rejects that case until it's root-caused.
#
# Rather than assembling the dense (block x sideband)-space Floquet
# Hamiltonian (see floquet.py:floquet_hamiltonian, still used by
# tests/keldysh/test_floquet_hamiltonian_assembly.py and kept as the
# reference construction) and inverting it whole (O((2*ns)^3)), this module
# exploits its structure directly: with no explicit central region there is
# exactly one spatial bond (the AC-carrying weak link), so block 0's
# sideband n only couples to block 1's sidebands n+1 and n-1 (never to
# another block-0 site or a same-sideband block-1 site). Following that
# coupling path visits every sideband exactly once, alternating which
# block owns it -- so the whole (block, sideband) lattice splits into
# exactly two independent 1D chains of `ns` sites each (verified: zero
# cross-coupling between the two chains). Each chain is then solved with a
# standard O(ns) recursive Green's function sweep (_rgf_chain) instead of
# an O(ns^3) dense inversion. Both the chain decomposition and the RGF
# sweep were validated against the dense construction to machine precision
# before being wired in here (see the PR/commit description).


def lesser_from_retarded(sigma_r, energy, temperature=0.):
    """Sigma^<(energy) = i*f(energy)*A(energy) = -f(energy)*[Sigma^r-Sigma^r^dagger],
    with f the Fermi function (equilibrium leads, chemical potential 0)."""
    f = fermidirac(np.array([energy]), temp=temperature)[0]
    return -f*(sigma_r - dagger(sigma_r))


def _check_supported(ht):
    if getattr(ht, "dimensionality", 1) != 1:
        raise NotImplementedError(
            "keldysh.dc_current only supports 1D leads")
    if not ht.has_eh:
        raise NotImplementedError(
            "keldysh.dc_current needs a Nambu (BdG) heterostructure; "
            "call h.turn_nambu() on both leads first (even with zero pairing)")
    if not (ht.block_diagonal and len(ht.central_intra) == 0):
        raise NotImplementedError(
            "keldysh.dc_current only supports heterostructures with no "
            "explicit central region (heterostructures.build(h1,h2), i.e. "
            "the two leads directly weak-linked via set_coupling); a "
            "confirmed, unresolved systematic error was found during "
            "testing for junctions with an explicit central Hamiltonian, so "
            "that case is rejected until it is root-caused")


def _prepare_system(ht):
    """Build the 2-block chain (one extra unit cell of each lead, directly
    weak-linked) and the electron/hole projectors for the right lead, whose
    unit cell is the target of the AC-carrying bond."""
    _check_supported(ht)
    hlist = enlarge_hlist(ht).central_intra
    proje = algebra.todense(ht.Hr.get_operator("electron").get_matrix())
    projh = algebra.todense(ht.Hr.get_operator("hole").get_matrix())
    dim = algebra.todense(hlist[0][0]).shape[0]
    if proje.shape[0] != dim:
        raise ValueError("dimension mismatch between the lead Hamiltonian "
                          "and the heterostructure's unit cell")
    return hlist, proje, projh, dim


def _cached_selfenergy(ht, e, lead, delta, cache):
    """Static lead self-energies only depend on (lead, energy); memoize them
    since the same energies recur across sideband/quadrature/adaptive-nmax
    evaluations within a single dc_current call, and green_renormalization
    (the underlying Sancho-Rubio iteration) is not cheap. `numba=True`
    routes it through the compiled Sancho-Rubio kernel (greentk.rg.
    green_renormalization_jit) instead of the plain-Python default used
    elsewhere in the library -- this call site alone recomputes lead
    selfenergies tens of thousands of times per dc_current call, where the
    per-call Python overhead dominates; the tolerance is the same as the
    Python path (see green_renormalization_jit), so this only changes
    speed, never the result."""
    key = (lead, round(e, 10))
    out = cache.get(key)
    if out is None:
        out = algebra.todense(ht.get_selfenergy(e, lead=lead, delta=delta,
                                                 pristine=True, numba=True))
        cache[key] = out
    return out


def _prefetch_selfenergies_batch(ht, es, lead, delta, cache):
    """Batch-compute every not-yet-cached selfenergy of one lead across a
    whole set of sideband energies at once (see transporttk/selfenergy.py:
    get_selfenergy_batch and greentk/rg.py:green_renormalization_jit_batch)
    instead of one sideband at a time: for a fixed quasienergy, the
    `2*nmax+1` sidebands only differ in energy for the same fixed lead, so
    they are embarrassingly parallel and are run over a numba `prange`
    loop across threads. `_cached_selfenergy` below then just hits the
    cache this fills in; the tolerance matches the non-batched path
    exactly, so this only changes speed, never the result."""
    keys = [(lead, round(e, 10)) for e in es]
    miss = [i for i, k in enumerate(keys) if k not in cache]
    if not miss: return
    me = np.array([es[i] for i in miss])
    outs = ht.get_selfenergy_batch(me, lead=lead, delta=delta, pristine=True)
    for i, out in zip(miss, outs):
        cache[keys[i]] = algebra.todense(out)


def _chain_sites(nmax):
    """The two independent 1D chains the (block, sideband) Floquet lattice
    decomposes into (see module docstring): each is a list of `ns =
    2*nmax+1` (block, n) pairs, one entry per sideband n, in physical
    chain order (consecutive entries are the actual nearest-neighbor
    bonds). The two chains partition the block-0 sites between them --
    together they cover every sideband n exactly once at block 0."""
    ns = 2*nmax+1
    chainA = [(0 if k % 2 == 0 else 1, -nmax+k) for k in range(ns)]
    chainB = [(1 if k % 2 == 0 else 0, -nmax+k) for k in range(ns)]
    return chainA, chainB


def _rgf_chain(Es, taus, SigLess):
    """Diagonal blocks of the retarded and lesser Green's functions of a
    1D block-tridiagonal chain, exact, via the standard O(N) two-sweep
    recursive Green's function algorithm (N = len(Es)) instead of one
    O(N^3) dense inversion (see e.g. Datta, "Electronic Transport in
    Mesoscopic Systems"; Lake & Datta, PRB 45, 6670 (1992) for the Keldysh
    extension). `taus[i]` is the hopping from site i to site i+1 (matrix
    convention H_{i+1,i} = -taus[i]), `Es[i]` is (energy+i*delta)*I - h_i -
    Sigma^r_i (the site's own onsite term already dressed by its local
    retarded selfenergy), `SigLess[i]` is Sigma^<_i.

    A forward sweep builds "left-connected" retarded/lesser Green's
    functions (site i dressed only by the embedding from sites 0..i-1), a
    backward sweep builds the mirror "right-connected" ones, and the two
    are combined at each site to get the true, fully-embedded diagonal
    block. Validated against dense np.linalg.inv to machine precision, on
    both generic random (non-Hermitian) test chains and the actual Floquet
    chains built by _floquet_green_functions below. The recursion itself
    (_rgf_chain_jit) is numba-compiled: like the selfenergy computation,
    this call site makes many np.linalg.inv calls on small (dim x dim)
    matrices, where per-call Python/LAPACK-dispatch overhead dominates the
    actual flop count -- compiling removes that overhead, not the math."""
    N = len(Es)
    dim = Es[0].shape[0]
    Es_arr = np.asarray(Es, dtype=np.complex128)
    SigLess_arr = np.asarray(SigLess, dtype=np.complex128)
    if N > 1:
        taus_arr = np.asarray(taus, dtype=np.complex128)
    else:
        taus_arr = np.empty((0, dim, dim), dtype=np.complex128)
    G, Gless = _rgf_chain_jit(Es_arr, taus_arr, SigLess_arr)
    return list(G), list(Gless)


@jit(nopython=True, cache=True)
def _rgf_chain_jit(Es, taus, SigLess):
    N = Es.shape[0]
    dim = Es.shape[1]
    gL = np.empty((N, dim, dim), dtype=np.complex128)
    gLessL = np.empty((N, dim, dim), dtype=np.complex128)
    gL[0] = np.linalg.inv(Es[0])
    gLessL[0] = gL[0]@SigLess[0]@np.conjugate(gL[0]).T
    for i in range(1, N):
        t = taus[i-1]
        td = np.conjugate(t).T
        sigl_r = t@gL[i-1]@td
        sigl_less = t@gLessL[i-1]@td
        gL[i] = np.linalg.inv(Es[i]-sigl_r)
        gLd = np.conjugate(gL[i]).T
        gLessL[i] = gL[i]@(SigLess[i]+sigl_less)@gLd
    gR = np.empty((N, dim, dim), dtype=np.complex128)
    gRless = np.empty((N, dim, dim), dtype=np.complex128)
    gR[N-1] = np.linalg.inv(Es[N-1])
    gRless[N-1] = gR[N-1]@SigLess[N-1]@np.conjugate(gR[N-1]).T
    for i in range(N-2, -1, -1):
        t = taus[i]
        td = np.conjugate(t).T
        sigr_r = td@gR[i+1]@t
        sigr_less = td@gRless[i+1]@t
        gR[i] = np.linalg.inv(Es[i]-sigr_r)
        gRd = np.conjugate(gR[i]).T
        gRless[i] = gR[i]@(SigLess[i]+sigr_less)@gRd
    G = np.empty((N, dim, dim), dtype=np.complex128)
    Gless = np.empty((N, dim, dim), dtype=np.complex128)
    for i in range(N):
        Eeff = Es[i].copy()
        SLtot = SigLess[i].copy()
        if i > 0:
            t = taus[i-1]
            td = np.conjugate(t).T
            Eeff = Eeff - t@gL[i-1]@td
            SLtot = SLtot + t@gLessL[i-1]@td
        if i < N-1:
            t = taus[i]
            td = np.conjugate(t).T
            Eeff = Eeff - td@gR[i+1]@t
            SLtot = SLtot + td@gRless[i+1]@t
        G[i] = np.linalg.inv(Eeff)
        Gd = np.conjugate(G[i]).T
        Gless[i] = G[i]@SLtot@Gd
    return G, Gless


def _floquet_green_functions(ht, voltage, quasienergy, nmax, delta,
                              temperature, cache):
    """Retarded and lesser Green's function diagonal blocks at every
    block-0 (left-lead-type) sideband, together with the left lead's
    lesser/advanced self-energies (needed by the current trace). Builds
    the two decoupled Floquet chains (_chain_sites) directly instead of
    assembling the dense (2*ns*dim)^2 Hamiltonian, and solves each with
    the O(ns) recursive sweep (_rgf_chain) -- exact, not an approximation
    (see module docstring)."""
    hlist, proje, projh, dim = _prepare_system(ht)
    v0 = algebra.todense(hlist[1][0])  # hopping <lead1 unit cell|H|lead0 unit cell>
    ve = proje@v0  # electron-projected AC bond, couples sideband n -> n+1
    vh = projh@v0  # hole-projected AC bond, couples sideband n -> n-1
    hii = [algebra.todense(hlist[0][0]), algebra.todense(hlist[1][1])]
    iden = np.eye(dim, dtype=np.complex128)
    ns = 2*nmax+1

    es = [quasienergy+(isb-nmax)*voltage for isb in range(ns)]
    _prefetch_selfenergies_batch(ht, es, 0, delta, cache)
    _prefetch_selfenergies_batch(ht, es, 1, delta, cache)

    Gr00, Gless00, sigL_less, sigL_a = {}, {}, {}, {}
    for chain in _chain_sites(nmax):
        Es, SigLess, taus = [], [], []
        for k, (b, n) in enumerate(chain):
            e = quasienergy+n*voltage
            sig_r = _cached_selfenergy(ht, e, b, delta, cache)
            Es.append((quasienergy+n*voltage+1j*delta)*iden - hii[b] - sig_r)
            sl = lesser_from_retarded(sig_r, e, temperature=temperature)
            SigLess.append(sl)
            if b == 0:
                sigL_less[n] = sl
                sigL_a[n] = dagger(sig_r)
            if k < len(chain)-1:
                taus.append(ve if b == 0 else dagger(vh))
        G, Gless = _rgf_chain(Es, taus, SigLess)
        for k, (b, n) in enumerate(chain):
            if b == 0:
                Gr00[n] = G[k]
                Gless00[n] = Gless[k]
    return Gr00, Gless00, sigL_less, sigL_a, dim, ns


def current_integrand(ht, voltage, quasienergy, nmax, tauz,
                       delta=1e-6, temperature=0., cache=None):
    """Integrand Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} of the paper's
    Eq. for I_dc, at a fixed quasienergy. `tauz` is the electron/hole
    grading operator matching the left lead's unit-cell dimension."""
    if cache is None:
        cache = {}
    Gr00, Gless00, sigL_less, sigL_a, dim, ns = _floquet_green_functions(
        ht, voltage, quasienergy, nmax, delta, temperature, cache)
    total = 0.0+0.0j
    for isb in range(ns):
        n = isb-nmax
        M = Gr00[n]@sigL_less[n] + Gless00[n]@sigL_a[n]
        total += np.trace(M@tauz)
    return total.real


def dc_current(ht, voltage, nmax=6, nmax_max=40, tol=1e-3, temperature=0.,
               delta=None):
    """Time-averaged (DC) current through a two-terminal junction (two
    leads, no explicit central region) under a bias `voltage`, computed
    with the Floquet-Keldysh formalism of San-Jose, Cayao, Prada, Aguado,
    NJP 15, 075019 (2013).

    The number of Floquet sidebands is increased adaptively (as in the
    paper) until the result changes by less than `tol`, capped at
    `nmax_max` to guarantee termination (a warning is issued if the cap is
    hit before convergence)."""
    if voltage == 0.:
        return 0.0
    _check_supported(ht)
    if delta is None:
        delta = ht.delta
    tauz = algebra.todense(ht.Hl.get_operator("tauz").get_matrix())
    cache = {}

    def integral(nmax):
        f = lambda e: current_integrand(ht, voltage, e, nmax, tauz,
                                         delta=delta, temperature=temperature,
                                         cache=cache)
        val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=1e-3)
        return val

    prev = integral(nmax)
    converged = False
    while nmax < nmax_max:
        nmax += 2
        cur = integral(nmax)
        denom = max(abs(cur), abs(prev), 1e-12)
        converged = abs(cur-prev)/denom < tol
        prev = cur
        if converged:
            break
    if not converged:
        warnings.warn(
            f"keldysh.dc_current: sidebands did not converge to tol={tol} "
            f"by nmax_max={nmax_max} at voltage={voltage}; result may be "
            "inaccurate, try a larger nmax_max")
    return prev


def iv_curve(ht, voltages, **kwargs):
    """Convenience wrapper: dc_current evaluated over an array of voltages,
    in parallel (see parallel.pcall)."""
    from ..parallel import pcall
    return np.array(pcall(lambda v: dc_current(ht, v, **kwargs), voltages))
