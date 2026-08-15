"""EXPERIMENTAL, not wired into dc_current yet -- see
documentation/keldysh_sideband_decimation_plan.md.

Absorbing-boundary (retarded + lesser) closure for the semi-infinite tail of
a Floquet sideband chain (see keldyshtk/current.py's module docstring for
the chain decomposition this operates on), computed by extending the SAME
forward/backward recursive-Green's-function recursion current.py's
_rgf_chain_jit already uses, out to convergence, instead of truncating the
chain at a hard, adaptively-grown `nmax`.

KNOWN BROKEN, DO NOT USE `dc_current_boundary`/`floquet_green_functions_boundary` FOR A REAL
CALCULATION: they silently drop most of the current's value. The DC current is a direct SUM
over every sideband's own local contribution (see current._integrand_trace_sum_jit), which,
via the Floquet-unfolding identity `energy = quasienergy + n*voltage`, is a discretized
integral over an unbounded real-energy range -- `nmax` in current.py's dc_current is
truncating that range, not providing a boundary condition around a localized region. Measured
directly: at nmax0=8 (voltage=0.1*delta, T=0.5, quasienergy=0.005), 80% of the current's value
comes from sidebands with |n|>8, entirely outside the window this module's fixed-window
wrapper solves. See documentation/keldysh_sideband_decimation_plan.md's "small-fixed-window
absorbing-boundary design ... FAILS" section for the full analysis.

`converged_boundary`/`_rgf_chain_boundary_jit` themselves (the actual absorbing-boundary
closure primitive) ARE correct -- validated to ~1e-16 against a large-nmax hard-truncation
reference (`validate_against_truncation` below) across several voltage regimes and
quasienergies. What's broken is using them to permanently truncate the sideband SUM; they
remain valid building blocks for a genuinely incremental-sum redesign (extend the summation
range and evaluate each new term cheaply via this closure, rather than dropping terms outside
a fixed window) -- not yet attempted, see the plan doc's process notes.
"""
import numpy as np
from numba import jit

from .current import (_assemble_chain_jit, _rgf_chain_jit, _batch_selfenergy,
                       _integrand_trace_sum_jit)


@jit(nopython=True, cache=True)
def _forward_seed_jit(Es, SigLess, taus):
    """Fresh-start forward (left-connected) RGF sweep over one chain
    segment (open boundary at its own first site): returns only the
    FINAL (gL, gLessL), at the segment's last site -- the boundary state
    to inject as an absorbing left-boundary condition for whatever comes
    after this segment. Same recursion as current._rgf_chain_jit's
    forward sweep, just discarding every intermediate site."""
    N = Es.shape[0]
    gL = np.linalg.inv(Es[0])
    gLessL = gL @ SigLess[0] @ np.conjugate(gL).T
    for i in range(1, N):
        t = taus[i-1]
        td = np.conjugate(t).T
        sigl_r = t @ gL @ td
        sigl_less = t @ gLessL @ td
        gL = np.linalg.inv(Es[i] - sigl_r)
        gLd = np.conjugate(gL).T
        gLessL = gL @ (SigLess[i] + sigl_less) @ gLd
    return gL, gLessL


@jit(nopython=True, cache=True)
def _backward_seed_jit(Es, SigLess, taus):
    """Mirror of _forward_seed_jit: fresh-start backward (right-connected)
    RGF sweep over one chain segment, returning the FINAL (gR, gRless) at
    the segment's FIRST site (the loop's last update)."""
    N = Es.shape[0]
    gR = np.linalg.inv(Es[N-1])
    gRless = gR @ SigLess[N-1] @ np.conjugate(gR).T
    for i in range(N-2, -1, -1):
        t = taus[i]
        td = np.conjugate(t).T
        sigr_r = td @ gR @ t
        sigr_less = td @ gRless @ t
        gR = np.linalg.inv(Es[i] - sigr_r)
        gRd = np.conjugate(gR).T
        gRless = gR @ (SigLess[i] + sigr_less) @ gRd
    return gR, gRless


@jit(nopython=True, cache=True)
def _rgf_chain_boundary_jit(Es, taus, SigLess, tau_left, gL_seed, gLessL_seed,
                             tau_right, gR_seed, gRless_seed):
    """Same recursion/combine structure as current._rgf_chain_jit, but the
    chain's two ends are dressed with precomputed absorbing-boundary
    states (gL_seed,gLessL_seed)/(gR_seed,gRless_seed) -- the converged
    embedding from everything beyond this (small, fixed-size) window --
    instead of a hard open boundary (inv(Es[0])/inv(Es[N-1])). `tau_left`/
    `tau_right` are the hops connecting the boundary states into the
    window's first/last site respectively. Exact given a converged seed."""
    N = Es.shape[0]
    dim = Es.shape[1]
    gL = np.empty((N, dim, dim), dtype=np.complex128)
    gLessL = np.empty((N, dim, dim), dtype=np.complex128)
    td = np.conjugate(tau_left).T
    sigl_r = tau_left @ gL_seed @ td
    sigl_less = tau_left @ gLessL_seed @ td
    gL[0] = np.linalg.inv(Es[0] - sigl_r)
    gLd = np.conjugate(gL[0]).T
    gLessL[0] = gL[0] @ (SigLess[0] + sigl_less) @ gLd
    for i in range(1, N):
        t = taus[i-1]
        td = np.conjugate(t).T
        sigl_r = t @ gL[i-1] @ td
        sigl_less = t @ gLessL[i-1] @ td
        gL[i] = np.linalg.inv(Es[i] - sigl_r)
        gLd = np.conjugate(gL[i]).T
        gLessL[i] = gL[i] @ (SigLess[i] + sigl_less) @ gLd

    gR = np.empty((N, dim, dim), dtype=np.complex128)
    gRless = np.empty((N, dim, dim), dtype=np.complex128)
    td = np.conjugate(tau_right).T
    sigr_r = td @ gR_seed @ tau_right
    sigr_less = td @ gRless_seed @ tau_right
    gR[N-1] = np.linalg.inv(Es[N-1] - sigr_r)
    gRd = np.conjugate(gR[N-1]).T
    gRless[N-1] = gR[N-1] @ (SigLess[N-1] + sigr_less) @ gRd
    for i in range(N-2, -1, -1):
        t = taus[i]
        td = np.conjugate(t).T
        sigr_r = td @ gR[i+1] @ t
        sigr_less = td @ gRless[i+1] @ t
        gR[i] = np.linalg.inv(Es[i] - sigr_r)
        gRd = np.conjugate(gR[i]).T
        gRless[i] = gR[i] @ (SigLess[i] + sigr_less) @ gRd

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
        else:
            t = tau_left
            td = np.conjugate(t).T
            Eeff = Eeff - t@gL_seed@td
            SLtot = SLtot + t@gLessL_seed@td
        if i < N-1:
            t = taus[i]
            td = np.conjugate(t).T
            Eeff = Eeff - td@gR[i+1]@t
            SLtot = SLtot + td@gRless[i+1]@t
        else:
            t = tau_right
            td = np.conjugate(t).T
            Eeff = Eeff - td@gR_seed@t
            SLtot = SLtot + td@gRless_seed@t
        G[i] = np.linalg.inv(Eeff)
        Gd = np.conjugate(G[i]).T
        Gless[i] = G[i]@SLtot@Gd
    return G, Gless


def _stretch_arrays(ht, quasienergy, voltage, n_start, depth, hii0, hii1, ve, vhd,
                     delta, temperature, start_block, cache, selfenergy_qtci):
    """Build (Es, SigLess, taus) for `depth` consecutive sideband sites
    n = n_start, n_start+1, ..., n_start+depth-1 (increasing n), whose
    block ownership starts at `start_block` and alternates -- same
    convention _assemble_chain_jit uses, just re-anchored at an arbitrary
    `n_start` instead of always -nmax."""
    es = np.array([quasienergy+n*voltage for n in range(n_start, n_start+depth)])
    sigR0 = _batch_selfenergy(ht, es, 0, delta, cache, selfenergy_qtci=selfenergy_qtci)
    sigR1 = _batch_selfenergy(ht, es, 1, delta, cache, selfenergy_qtci=selfenergy_qtci)
    Es, SigLess, taus, _, _ = _assemble_chain_jit(
        es, sigR0, sigR1, hii0, hii1, ve, vhd, delta, temperature, start_block)
    return Es, SigLess, taus


def converged_boundary(ht, quasienergy, voltage, nmax0, hii0, hii1, ve, vhd, delta,
                        temperature, side, start_block, cache, selfenergy_qtci=None,
                        tol=1e-10, depth0=64, depth_max=8192):
    """Converged absorbing-boundary (g, gless, tau_connect) for one side of
    a fixed window [-nmax0, nmax0] of one chain (start_block in {0,1}, see
    current.py's module docstring for the chain decomposition).

    `side='left'`: closes off sidebands n < -nmax0, returns (gL, gLessL)
    at n=-nmax0-1 plus the hop `tau_connect` linking it into the window's
    first site (n=-nmax0).
    `side='right'`: closes off sidebands n > nmax0, returns (gR, gRless)
    at n=nmax0+1 plus the hop linking it into the window's last site
    (n=nmax0).

    Found by extending the seed stretch's depth (64, 128, 256, ...,
    doubling) until the boundary state stops changing by more than `tol`
    (relative, max-norm) -- the seed sweep itself is cheap (small dim x
    dim matrix ops over `depth` sites, not a full chain solve with self-
    energy evaluation overhead), so redoing it from scratch at each
    doubling is negligible next to what this is meant to replace. `depth`
    is kept even throughout so a fixed `start_block_stretch` correctly
    describes every doubling (see this function's own derivation in
    documentation/keldysh_sideband_decimation_plan.md's implementation
    notes)."""
    import warnings
    if side == 'left':
        # block at n=-nmax0-1 (site closest to the window) is
        # 1-start_block (window's own convention: block(n) = start_block
        # xor ((n+nmax0) % 2), and (-nmax0-1+nmax0) % 2 == 1).
        tau_connect = ve if start_block == 1 else vhd
    elif side == 'right':
        # block at n=nmax0 (window's own last site) is start_block; the
        # hop out of it is the same alternation _assemble_chain_jit uses.
        tau_connect = ve if start_block == 0 else vhd
    else:
        raise ValueError("side must be 'left' or 'right'")

    depth = depth0
    prev = None
    g = gless = None
    while depth <= depth_max:
        if side == 'left':
            n_start = -nmax0-depth
            # block at n_start relative to window anchor: start_block xor
            # ((n_start+nmax0) % 2) == start_block xor (depth % 2); depth
            # is kept even, so this is just start_block.
            stretch_start_block = start_block
            Es, SigLess, taus = _stretch_arrays(
                ht, quasienergy, voltage, n_start, depth, hii0, hii1, ve, vhd,
                delta, temperature, stretch_start_block, cache, selfenergy_qtci)
            g, gless = _forward_seed_jit(Es, SigLess, taus)
        else:
            n_start = nmax0+1
            # block at n_start relative to window anchor: start_block xor
            # ((n_start+nmax0) % 2) == start_block xor 1.
            stretch_start_block = 1-start_block
            Es, SigLess, taus = _stretch_arrays(
                ht, quasienergy, voltage, n_start, depth, hii0, hii1, ve, vhd,
                delta, temperature, stretch_start_block, cache, selfenergy_qtci)
            g, gless = _backward_seed_jit(Es, SigLess, taus)
        if prev is not None:
            num = max(np.max(np.abs(g-prev[0])), np.max(np.abs(gless-prev[1])))
            den = max(np.max(np.abs(g)), np.max(np.abs(gless)), 1e-12)
            if num/den < tol:
                return g, gless, tau_connect
        prev = (g, gless)
        depth *= 2
    warnings.warn(
        f"keldyshtk.boundary.converged_boundary: {side} boundary did not "
        f"converge to tol={tol} by depth_max={depth_max} at quasienergy="
        f"{quasienergy}, voltage={voltage}; result may be inaccurate")
    return g, gless, tau_connect


def floquet_green_functions_boundary(ht, voltage, quasienergy, nmax0, delta, temperature,
                                      cache, system, selfenergy_qtci=None, tol=1e-10):
    """Drop-in analogue of current._floquet_green_functions, but solving a
    FIXED, small window [-nmax0, nmax0] dressed with converged absorbing
    boundaries (see converged_boundary) on both ends of both chains,
    instead of a hard-truncated chain re-grown adaptively. EXPERIMENTAL:
    not yet validated against the production path -- see
    validate_against_truncation below and documentation/
    keldysh_sideband_decimation_plan.md."""
    from .. import algebra
    from ..algebra import dagger
    hlist, proje, projh, dim = system
    if len(hlist) != 2:
        # Same two-chain-decomposition assumption as current.py's fast path;
        # a junction with an explicit central region has no such chain (see
        # current._dense_floquet_integrand) and is not handled here.
        raise NotImplementedError(
            "the absorbing-boundary path only supports a two-block chain")
    v0 = algebra.todense(hlist[1][0])
    ve = (proje@v0).astype(np.complex128)
    vh = projh@v0
    vhd = dagger(vh).astype(np.complex128)
    hii0 = algebra.todense(hlist[0][0]).astype(np.complex128)
    hii1 = algebra.todense(hlist[1][1]).astype(np.complex128)
    ns = 2*nmax0+1

    es = np.array([quasienergy+(isb-nmax0)*voltage for isb in range(ns)])
    sigR0 = _batch_selfenergy(ht, es, 0, delta, cache, selfenergy_qtci=selfenergy_qtci)
    sigR1 = _batch_selfenergy(ht, es, 1, delta, cache, selfenergy_qtci=selfenergy_qtci)

    Gr00 = np.empty((ns, dim, dim), dtype=np.complex128)
    Gless00 = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((ns, dim, dim), dtype=np.complex128)
    for start_block in (0, 1):
        Es, SigLess, taus, sl_less, sl_a = _assemble_chain_jit(
            es, sigR0, sigR1, hii0, hii1, ve, vhd, delta, temperature, start_block)
        gL, gLessL, tau_left = converged_boundary(
            ht, quasienergy, voltage, nmax0, hii0, hii1, ve, vhd, delta, temperature,
            'left', start_block, cache, selfenergy_qtci=selfenergy_qtci, tol=tol)
        gR, gRless, tau_right = converged_boundary(
            ht, quasienergy, voltage, nmax0, hii0, hii1, ve, vhd, delta, temperature,
            'right', start_block, cache, selfenergy_qtci=selfenergy_qtci, tol=tol)
        G, Gless = _rgf_chain_boundary_jit(Es, taus, SigLess, tau_left, gL, gLessL,
                                            tau_right, gR, gRless)
        Gr00[start_block::2] = G[start_block::2]
        Gless00[start_block::2] = Gless[start_block::2]
        sigL_less[start_block::2] = sl_less[start_block::2]
        sigL_a[start_block::2] = sl_a[start_block::2]
    return Gr00, Gless00, sigL_less, sigL_a, dim, ns


def current_integrand_boundary(ht, voltage, quasienergy, nmax0, tauz, delta=1e-6,
                                temperature=0., cache=None, system=None,
                                selfenergy_qtci=None, tol=1e-10):
    """Analogue of current.current_integrand, using
    floquet_green_functions_boundary (fixed window + converged absorbing
    boundaries) instead of current._floquet_green_functions (hard
    truncation at an adaptively-grown nmax)."""
    if cache is None:
        cache = {}
    if system is None:
        from .current import _prepare_system
        system = _prepare_system(ht)
    Gr00, Gless00, sigL_less, sigL_a, dim, ns = floquet_green_functions_boundary(
        ht, voltage, quasienergy, nmax0, delta, temperature, cache, system,
        selfenergy_qtci=selfenergy_qtci, tol=tol)
    if tauz.dtype != np.complex128:
        tauz = tauz.astype(np.complex128)
    return _integrand_trace_sum_jit(Gr00, sigL_less, Gless00, sigL_a, tauz).real


def dc_current_boundary(ht, voltage, nmax0=8, delta=None, temperature=0.,
                         selfenergy_qtci=None, tol=1e-10, epsrel=1e-3):
    """EXPERIMENTAL analogue of current.dc_current: same quasienergy
    quadrature (scipy.integrate.quad over [0,|voltage|]), but each point
    is solved on a FIXED, small window [-nmax0,nmax0] dressed with
    converged absorbing boundaries instead of an adaptively-grown
    open-boundary chain -- no nmax/nmax_max/adaptive-sideband-loop
    parameters, since the boundary closure replaces that convergence
    axis entirely. Not yet wired in as dc_current's default path -- see
    documentation/keldysh_sideband_decimation_plan.md. `nmax0` should be
    picked generously enough to resolve the near-gap MAR/Andreev
    resonance structure explicitly (validate_against_truncation is the
    tool to check this for a given system) -- convergence in nmax0 itself
    is NOT yet automated here, unlike current.dc_current's nmax."""
    raise RuntimeError(
        "dc_current_boundary is known broken: it silently drops most of "
        "the current's value (measured ~80% at nmax0=8, see this module's "
        "docstring and documentation/keldysh_sideband_decimation_plan.md). "
        "Use keldyshtk.current.dc_current instead. The underlying "
        "floquet_green_functions_boundary/converged_boundary primitives "
        "are correct and remain usable directly (see "
        "validate_against_truncation) -- only this fixed-window current "
        "assembly is disabled.")
    from .current import _prepare_bias_target, _check_supported, _prepare_system
    from scipy.integrate import quad
    from .. import algebra
    if voltage == 0.:
        return 0.0
    ht = _prepare_bias_target(ht)
    _check_supported(ht)
    if delta is None:
        delta = ht.delta
    from .current import _is_localprobe
    lead0 = ht.lead if _is_localprobe(ht) else ht.Hl
    tauz = algebra.todense(lead0.get_operator("tauz").get_matrix()).astype(np.complex128)
    cache = {}
    system = _prepare_system(ht)
    f = lambda e: current_integrand_boundary(
        ht, voltage, e, nmax0, tauz, delta=delta, temperature=temperature,
        cache=cache, system=system, selfenergy_qtci=selfenergy_qtci, tol=tol)
    val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=epsrel)
    return val


def validate_against_truncation(ht, voltage, quasienergy, nmax0, nmax_reference, delta,
                                 temperature=0., selfenergy_qtci=None, tol=1e-10):
    """Standalone correctness check: compare floquet_green_functions_boundary's
    Gr00/Gless00 at the FIXED window [-nmax0,nmax0] against
    current._floquet_green_functions run at a much larger `nmax_reference`
    (hard truncation, already validated -- see current.py's own docstrings/
    tests), restricted to the same window of sidebands. Returns the max
    relative error over the window -- use this before trusting
    floquet_green_functions_boundary for anything."""
    from .current import _prepare_system, _floquet_green_functions
    system = _prepare_system(ht)
    cache_ref = {}
    Gr00_ref, Gless00_ref, _, _, dim, ns_ref = _floquet_green_functions(
        ht, voltage, quasienergy, nmax_reference, delta, temperature, cache_ref, system,
        selfenergy_qtci=selfenergy_qtci)
    cache_bd = {}
    Gr00_bd, Gless00_bd, _, _, dim2, ns_bd = floquet_green_functions_boundary(
        ht, voltage, quasienergy, nmax0, delta, temperature, cache_bd, system,
        selfenergy_qtci=selfenergy_qtci, tol=tol)
    offset = nmax_reference-nmax0
    Gr00_ref_win = Gr00_ref[offset:offset+ns_bd]
    Gless00_ref_win = Gless00_ref[offset:offset+ns_bd]
    denom_r = max(np.max(np.abs(Gr00_ref_win)), 1e-12)
    denom_l = max(np.max(np.abs(Gless00_ref_win)), 1e-12)
    err_r = np.max(np.abs(Gr00_bd-Gr00_ref_win))/denom_r
    err_l = np.max(np.abs(Gless00_bd-Gless00_ref_win))/denom_l
    return max(err_r, err_l)
