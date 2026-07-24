import warnings

import numpy as np
from numba import jit
from scipy.integrate import quad

from .. import algebra
from ..algebra import dagger
from ..transporttk.smatrix import enlarge_hlist

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


def _fermi_scalar(e, temperature=0.):
    """Scalar Fermi-Dirac occupation, equivalent to
    transporttk.fermidirac.fermidirac(np.array([e]),temp=temperature)[0]
    but without that function's array-wrap/extract overhead -- this
    module's hot loop (_floquet_green_functions) calls it once per
    (block,sideband) site, tens of thousands of times per dc_current
    call, where that overhead was measured to matter (~0.2s of a 3.4s
    call, see _rgf_chain_jit's docstring for the profiling context)."""
    if temperature == 0.:
        if e < 0.: return 1.0
        if e > 0.: return 0.0
        return 0.5
    return 1.0/(1.0+np.exp(e/temperature))


def lesser_from_retarded(sigma_r, energy, temperature=0.):
    """Sigma^<(energy) = i*f(energy)*A(energy) = -f(energy)*[Sigma^r-Sigma^r^dagger],
    with f the Fermi function (equilibrium leads, chemical potential 0)."""
    f = _fermi_scalar(energy, temperature)
    return -f*(sigma_r - dagger(sigma_r))


def _is_localprobe(ht):
    from ..transporttk.localprobe import LocalProbe
    return isinstance(ht, LocalProbe)


def _check_supported(ht):
    if getattr(ht, "dimensionality", 1) != 1:
        raise NotImplementedError(
            "keldysh.dc_current only supports 1D leads")
    if not ht.has_eh:
        raise NotImplementedError(
            "keldysh.dc_current needs a Nambu (BdG) heterostructure; "
            "call h.turn_nambu() on both leads first (even with zero pairing)")
    if _is_localprobe(ht):
        return  # a LocalProbe has no explicit central region by construction
    if not (ht.block_diagonal and len(ht.central_intra) == 0):
        raise NotImplementedError(
            "keldysh.dc_current only supports heterostructures with no "
            "explicit central region (heterostructures.build(h1,h2), i.e. "
            "the two leads directly weak-linked via set_coupling); a "
            "confirmed, unresolved systematic error was found during "
            "testing for junctions with an explicit central Hamiltonian, so "
            "that case is rejected until it is root-caused")


def _prepare_system_localprobe(lp):
    """Build the 2-block (probe, sample-site) chain and the electron/hole
    projectors for a LocalProbe, mirroring `_prepare_system` below: block 0
    is the probe's unit cell (`lp.lead`), block 1 is the single bulk site
    being probed (`lp.i` of `lp.H`), and the AC-carrying bond is the
    probe-sample tunneling link scaled by the transparency `lp.T` -- the
    same raw Hamiltonian block structure `localprobe.get_central_gmatrix`
    dresses with self-energies and (energy+i*delta) for the smatrix method."""
    from ..htk.extract import local_hamiltonian
    from ..transporttk.localprobe import get_intra
    oi = algebra.todense(local_hamiltonian(lp.H, get_intra(lp.H), i=lp.i))
    h01 = algebra.todense(lp.lead.inter)*lp.T  # probe -> sample-site bond
    hlist = [[algebra.todense(lp.lead.intra), h01],
             [dagger(h01), oi]]
    proje = algebra.todense(local_hamiltonian(
        lp.H, lp.H.get_operator("electron").get_matrix(), i=lp.i))
    projh = algebra.todense(local_hamiltonian(
        lp.H, lp.H.get_operator("hole").get_matrix(), i=lp.i))
    dim = hlist[0][0].shape[0]
    if proje.shape[0] != dim:
        raise ValueError("dimension mismatch between the probe lead and "
                          "the local sample site")
    return hlist, proje, projh, dim


def _prepare_system(ht):
    """Build the 2-block chain (one extra unit cell of each lead, directly
    weak-linked) and the electron/hole projectors for the right lead, whose
    unit cell is the target of the AC-carrying bond."""
    _check_supported(ht)
    if _is_localprobe(ht):
        return _prepare_system_localprobe(ht)
    hlist = enlarge_hlist(ht).central_intra
    proje = algebra.todense(ht.Hr.get_operator("electron").get_matrix())
    projh = algebra.todense(ht.Hr.get_operator("hole").get_matrix())
    dim = algebra.todense(hlist[0][0]).shape[0]
    if proje.shape[0] != dim:
        raise ValueError("dimension mismatch between the lead Hamiltonian "
                          "and the heterostructure's unit cell")
    return hlist, proje, projh, dim


def _cached_selfenergy(ht, e, lead, delta, cache, selfenergy_qtci=None):
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
    speed, never the result.

    `selfenergy_qtci`, if given, is a {lead: SelfenergyQTCI} dict (see
    qtcitk.selfenergy_qtci and keldyshtk.current.build_selfenergy_qtci):
    an interpolant built once (from far fewer true solves) and evaluated
    here instead of a fresh solve -- this dict-based memoization cache is
    local to one dc_current call, but the interpolant itself can be built
    once and shared across many dc_current calls (e.g. both sides of
    keldysh_didv's finite-difference derivative), unlike this cache."""
    key = (lead, round(e, 10))
    out = cache.get(key)
    if out is None:
        if selfenergy_qtci is not None and lead in selfenergy_qtci:
            out = selfenergy_qtci[lead](e)
        else:
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
    if not hasattr(ht, "get_selfenergy_batch"):
        return  # e.g. LocalProbe: fall back to per-energy _cached_selfenergy
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


@jit(nopython=True, cache=True)
def _rgf_chain_jit(Es, taus, SigLess):
    """Diagonal blocks of the retarded and lesser Green's functions of a
    1D block-tridiagonal chain, exact, via the standard O(N) two-sweep
    recursive Green's function algorithm (N = Es.shape[0]) instead of one
    O(N^3) dense inversion (see e.g. Datta, "Electronic Transport in
    Mesoscopic Systems"; Lake & Datta, PRB 45, 6670 (1992) for the Keldysh
    extension). `taus[i]` is the hopping from site i to site i+1 (matrix
    convention H_{i+1,i} = -taus[i]), `Es[i]` is (energy+i*delta)*I - h_i -
    Sigma^r_i (the site's own onsite term already dressed by its local
    retarded selfenergy), `SigLess[i]` is Sigma^<_i. Callers (see
    _floquet_green_functions) build these as (N,dim,dim)/(N-1,dim,dim)
    numpy arrays directly rather than Python lists later converted with
    np.asarray -- that conversion, when this used to be wrapped in a
    separate _rgf_chain(Es, taus, SigLess) taking lists, was measured to
    cost more than the recursion below (1.1s of a 3.4s dc_current call,
    ~1680 calls -- cProfile'd after the self-energy path itself had
    already been optimized away as the bottleneck, see aaatk/
    selfenergy_aaa.py's module docstring for that earlier round).

    A forward sweep builds "left-connected" retarded/lesser Green's
    functions (site i dressed only by the embedding from sites 0..i-1), a
    backward sweep builds the mirror "right-connected" ones, and the two
    are combined at each site to get the true, fully-embedded diagonal
    block. Validated against dense np.linalg.inv to machine precision, on
    both generic random (non-Hermitian) test chains and the actual Floquet
    chains built by _floquet_green_functions below. This recursion is
    numba-compiled: like the selfenergy computation, this call site makes
    many np.linalg.inv calls on small (dim x dim) matrices, where per-call
    Python/LAPACK-dispatch overhead dominates the actual flop count --
    compiling removes that overhead, not the math."""
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
                              temperature, cache, system,
                              selfenergy_qtci=None):
    """Retarded and lesser Green's function diagonal blocks at every
    block-0 (left-lead-type) sideband, together with the left lead's
    lesser/advanced self-energies (needed by the current trace). Builds
    the two decoupled Floquet chains (_chain_sites) directly instead of
    assembling the dense (2*ns*dim)^2 Hamiltonian, and solves each with
    the O(ns) recursive sweep (_rgf_chain_jit) -- exact, not an
    approximation (see module docstring). `system` is the (hlist, proje,
    projh, dim) tuple from `_prepare_system(ht)`, precomputed once per
    `dc_current` call by the caller: it only depends on `ht` (never on
    quasienergy, voltage, nmax or delta), but this function is called once
    per quadrature point of the current integral, so recomputing it here
    -- which involves building electron/hole projector operators over
    `ht` and extracting local Hamiltonian blocks -- would redo the same
    work tens to hundreds of times per `dc_current` call for no benefit."""
    hlist, proje, projh, dim = system
    v0 = algebra.todense(hlist[1][0])  # hopping <lead1 unit cell|H|lead0 unit cell>
    ve = proje@v0  # electron-projected AC bond, couples sideband n -> n+1
    vh = projh@v0  # hole-projected AC bond, couples sideband n -> n-1
    vhd = dagger(vh)  # precomputed once, not per-site (it's a loop constant)
    hii = [algebra.todense(hlist[0][0]), algebra.todense(hlist[1][1])]
    iden = np.eye(dim, dtype=np.complex128)
    ns = 2*nmax+1

    es = [quasienergy+(isb-nmax)*voltage for isb in range(ns)]
    if selfenergy_qtci is None:
        # batch-prefetching amortizes the per-call solve cost across the
        # sideband set (see _prefetch_selfenergies_batch); with a qtci
        # interpolant there's no solve left to amortize, only a cheap
        # tensor-train evaluation, so it's skipped when qtci is in use.
        _prefetch_selfenergies_batch(ht, es, 0, delta, cache)
        _prefetch_selfenergies_batch(ht, es, 1, delta, cache)

    # Indexed by isb=n+nmax (0..ns-1), not a dict keyed by n: the sideband
    # index n ranges over a small contiguous [-nmax,nmax], so a dict here
    # only ever paid Python hashing/lookup overhead for what's really a
    # plain array index -- and this shape lets current_integrand's trace
    # sum below run as one jitted call over that per its enumerate below.
    Gr00 = np.empty((ns, dim, dim), dtype=np.complex128)
    Gless00 = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((ns, dim, dim), dtype=np.complex128)
    for chain in _chain_sites(nmax):
        # Built directly as (N,dim,dim) arrays, not Python lists later
        # converted with np.asarray -- see _rgf_chain_jit's docstring for
        # why that conversion mattered once the self-energy path itself
        # was no longer the bottleneck.
        N = len(chain)
        Es = np.empty((N, dim, dim), dtype=np.complex128)
        SigLess = np.empty((N, dim, dim), dtype=np.complex128)
        taus = np.empty((max(N-1, 0), dim, dim), dtype=np.complex128)
        for k, (b, n) in enumerate(chain):
            e = quasienergy+n*voltage
            sig_r = _cached_selfenergy(ht, e, b, delta, cache,
                                        selfenergy_qtci=selfenergy_qtci)
            sig_r_dag = dagger(sig_r)  # computed once, reused below (was twice)
            Es[k] = (e+1j*delta)*iden - hii[b] - sig_r
            f = _fermi_scalar(e, temperature)
            sl = -f*(sig_r - sig_r_dag)
            SigLess[k] = sl
            if b == 0:
                sigL_less[n+nmax] = sl
                sigL_a[n+nmax] = sig_r_dag
            if k < N-1:
                taus[k] = ve if b == 0 else vhd
        G, Gless = _rgf_chain_jit(Es, taus, SigLess)
        for k, (b, n) in enumerate(chain):
            if b == 0:
                Gr00[n+nmax] = G[k]
                Gless00[n+nmax] = Gless[k]
    return Gr00, Gless00, sigL_less, sigL_a, dim, ns


def current_integrand(ht, voltage, quasienergy, nmax, tauz,
                       delta=1e-6, temperature=0., cache=None, system=None,
                       selfenergy_qtci=None):
    """Integrand Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} of the paper's
    Eq. for I_dc, at a fixed quasienergy. `tauz` is the electron/hole
    grading operator matching the left lead's unit-cell dimension.
    `system` is the precomputed `_prepare_system(ht)` tuple, see
    `_floquet_green_functions`; computed on demand if not given (e.g. for
    standalone callers/tests) so this stays a valid entry point on its
    own. `selfenergy_qtci`: see _cached_selfenergy/build_selfenergy_qtci."""
    if cache is None:
        cache = {}
    if system is None:
        system = _prepare_system(ht)
    Gr00, Gless00, sigL_less, sigL_a, dim, ns = _floquet_green_functions(
        ht, voltage, quasienergy, nmax, delta, temperature, cache, system,
        selfenergy_qtci=selfenergy_qtci)
    # _integrand_trace_sum_jit (numba) requires matching operand dtypes for
    # @; dc_current's internal call site already passes a complex128 tauz
    # (cast once there, not on every quasienergy evaluation), so this only
    # copies for a standalone caller passing a real-valued tauz.
    if tauz.dtype != np.complex128:
        tauz = tauz.astype(np.complex128)
    return _integrand_trace_sum_jit(Gr00, sigL_less, Gless00, sigL_a, tauz).real


@jit(nopython=True, cache=True)
def _integrand_trace_sum_jit(Gr00, sigL_less, Gless00, sigL_a, tauz):
    """sum_isb Tr{[Gr00[isb]@sigL_less[isb] + Gless00[isb]@sigL_a[isb]]@tauz},
    the paper's Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} summed over
    every sideband -- compiled since the per-sideband matmul+trace was, at
    Python/numpy dispatch level, a meaningful fraction of the per-
    quasienergy-point cost (np.trace alone has real generic-dispatch
    overhead for a 4x4 array; ns can be up to ~2*nmax_max+1 sidebands,
    called once per quadrature point)."""
    total = 0j
    ns = Gr00.shape[0]
    for isb in range(ns):
        M = Gr00[isb]@sigL_less[isb] + Gless00[isb]@sigL_a[isb]
        MT = M@tauz
        tr = 0j
        for d in range(MT.shape[0]):
            tr += MT[d, d]
        total += tr
    return total


def _prepare_bias_target(ht):
    """For a LocalProbe whose probe lead is normal (no/negligible pairing),
    ground it: force `frozen_lead=True` so its self-energy is evaluated at
    absolute energy 0 for every bias, with the entire bias dropped across
    the sample instead -- the same convention `didv(method="smatrix")`
    uses for the probe, so that `dc_current`/`method="keldysh"` on such a
    LocalProbe is directly consistent with its own `method="smatrix"`
    result (see transporttk.didv.didv's docstring for measurements of the
    O(1) disagreement this closes).

    For a LocalProbe whose probe lead genuinely is superconducting, do the
    opposite: force `frozen_lead=False` (unfreeze), so the probe's own
    Floquet sideband physics is evaluated at its actual, varying energy.
    Grounding it instead would pin every evaluation at the probe's own gap
    center -- confirmed to suppress the AC-Josephson/MAR current there by
    over an order of magnitude (0.025 vs the correct 0.38 for
    examples/transport/decay_constant_keldysh's parameters), since a
    grounded probe can never sample its own gap edge, where the
    quasiparticle transport that current depends on actually lives. This
    is the library's originally-validated LocalProbe Keldysh use case
    (both probe and sample superconducting, no normal lead for "smatrix"
    to reflect against) and must keep working exactly as before.

    Two-lead Heterostructures have no `frozen_lead` concept and are
    always returned untouched -- their `dc_current`/`method="keldysh"`
    behavior (validated against a normal-normal rigid two-terminal bias
    reference and an equilibrium Andreev linear-response check, see
    tests/keldysh) is unaffected by this function and must stay that
    way."""
    if not _is_localprobe(ht):
        return ht
    from ..transporttk.didv import _lead_is_superconducting
    ht = ht.copy()
    ht.frozen_lead = not _lead_is_superconducting(ht.lead)
    return ht


def build_selfenergy_qtci(ht, voltage, nmax_max, delta=None, margin=4,
                           tolerance=1e-6, **kwargs):
    """Build one qtcitk.selfenergy_qtci.SelfenergyQTCI interpolant per lead
    (0 and 1), covering every Floquet sideband energy dc_current's
    adaptive nmax loop could ever reach for this `voltage`, capped at
    `nmax_max`: the quasienergy integral ranges over [0,|voltage|] and
    sidebands add up to nmax_max more steps of |voltage| in either
    direction, so |energy| <= (nmax_max+1)*|voltage| covers it with room
    to spare. Pass the result as dc_current's `selfenergy_qtci` argument;
    building it once and sharing it across multiple dc_current calls
    (e.g. both sides of keldysh_didv's finite-difference derivative, or a
    whole iv_curve sweep) is the point -- a single dc_current call alone
    was measured to solve ~28500 distinct (lead,energy) self-energies
    from scratch with essentially no reuse, self-energy computation
    alone accounting for ~78% of total wall time.

    Measured NOT to help for a LocalProbe's Sancho-Rubio lead
    self-energy specifically (see qtcitk.selfenergy_qtci's module
    docstring for the full benchmark): building the interpolants can end
    up needing *more* true solves than the direct per-energy approach,
    because that self-energy isn't compressible enough over the energy
    range multiple Andreev reflection needs. Kept as tested, documented,
    opt-in infrastructure (not used unless a caller explicitly builds and
    passes selfenergy_qtci) for a self-energy that compresses better."""
    ht = _prepare_bias_target(ht)
    _check_supported(ht)
    if delta is None: delta = ht.delta
    system = _prepare_system(ht)
    hlist, proje, projh, dim = system
    erange = (nmax_max+1)*abs(voltage)
    from ..qtcitk.selfenergy_qtci import SelfenergyQTCI
    out = {}
    for lead in (0, 1):
        def get_se(e, lead=lead): # default arg freezes the loop variable
            return ht.get_selfenergy(e, lead=lead, delta=delta,
                                     pristine=True, numba=True)
        out[lead] = SelfenergyQTCI(get_se, dim, -erange, erange, delta,
                                    margin=margin, tolerance=tolerance,
                                    **kwargs)
    return out


def build_selfenergy_aaa(ht, voltage, nmax_max, delta=None,
                          tolerance=1e-6, **kwargs):
    """Build one aaatk.selfenergy_aaa.SelfenergyAAA interpolant per lead (0
    and 1), covering the same energy window as build_selfenergy_qtci
    (|energy| <= (nmax_max+1)*|voltage|) and returned in the same {lead:
    interpolant} form -- a drop-in replacement for build_selfenergy_qtci
    wherever that is used (dc_current's `selfenergy_qtci` argument,
    keldysh_didv's shared-interpolant plumbing), since both interpolant
    types share the same `interp(energy) -> matrix` call contract.

    Unlike build_selfenergy_qtci, this one *does* pay off for a
    LocalProbe's Sancho-Rubio self-energy: AAA represents a narrow
    resonance directly as a pole rather than having to bisect its width
    bit-by-bit on a quantics grid, so a rational fit to the same target
    needs on the order of hundreds of true solves rather than the tens of
    thousands qtci needed. That solve-count reduction is real, but only
    part of the story -- see aaatk/selfenergy_aaa.py's module docstring
    for the measured net wall-clock effect through the actual dc_current
    pipeline (modest for a cheap-per-solve 1D target, since evaluating the
    interpolant many times isn't free either; larger for an expensive-
    per-solve target), which is why dc_current uses this by default
    (selfenergy_method="aaa") but with a bounded build budget and a
    fallback to direct solves if that budget isn't enough to converge."""
    ht = _prepare_bias_target(ht)
    _check_supported(ht)
    if delta is None: delta = ht.delta
    system = _prepare_system(ht)
    hlist, proje, projh, dim = system
    erange = (nmax_max+1)*abs(voltage)
    from ..aaatk.selfenergy_aaa import SelfenergyAAA
    out = {}
    for lead in (0, 1):
        def get_se(e, lead=lead): # default arg freezes the loop variable
            return ht.get_selfenergy(e, lead=lead, delta=delta,
                                     pristine=True, numba=True)
        out[lead] = SelfenergyAAA(get_se, dim, -erange, erange, delta,
                                   tolerance=tolerance, **kwargs)
    return out


def dc_current(ht, voltage, nmax=6, nmax_max=40, tol=1e-3, temperature=0.,
               delta=None, min_consecutive=2, selfenergy_qtci=None,
               selfenergy_method="aaa"):
    """Time-averaged (DC) current through a two-terminal junction under a
    bias `voltage`, computed with the Floquet-Keldysh formalism of
    San-Jose, Cayao, Prada, Aguado, NJP 15, 075019 (2013). The junction is
    either a two-lead heterostructure with no explicit central region
    (heterostructures.build(h1,h2)) or a LocalProbe (probe tip weakly
    coupled to a single site of a bulk sample) whose probe and sample are
    both superconducting.

    The number of Floquet sidebands is increased adaptively (as in the
    paper) until the result changes by less than `tol` for
    `min_consecutive` sideband increments in a row, capped at `nmax_max`
    to guarantee termination (a warning is issued if the cap is hit
    before convergence). Requiring more than one consecutive agreeing
    step (rather than just the last pair) guards against sub-gap
    energies where the integral is not monotonic in nmax -- confirmed to
    dip through a near-zero crossing and even change sign before
    settling, which can otherwise make a single lucky pair of nearby
    values look converged while the sequence is still far from its true
    limit (observed: nmax=60->62 already satisfies tol=1e-3 on its own,
    while the true limit only stabilizes to machine precision by
    nmax~64).

    `selfenergy_method` picks how lead self-energies are obtained:
    "aaa" (default) builds one aaatk.selfenergy_aaa.SelfenergyAAA
    interpolant per lead internally (see build_selfenergy_aaa), covering
    this call's voltage/nmax_max window, and evaluates it instead of
    solving Sancho-Rubio/bloch_selfenergy from scratch at every one of the
    (tens of thousands of) distinct (lead,energy) pairs the adaptive
    sideband sweep visits. Measured through the actual pipeline (see
    aaatk/selfenergy_aaa.py's module docstring for the performance bugs
    found and fixed while measuring it, and _rgf_chain_jit/
    _integrand_trace_sum_jit's docstrings below for a second, separate
    optimization round of the shared RGF-chain/trace-sum machinery that
    benefits the direct path too): a consistent win even within a single
    call for a cheap-per-solve target like a 1D Sancho-Rubio self-energy
    (roughly break-even to ~40% faster), and should be substantially
    larger for an expensive-per-solve target (e.g. a 2D sample's
    green_kchain-based self-energy) where a single direct solve alone
    costs much more than the interpolant's whole per-energy evaluation.
    Sharing one interpolant across several calls
    (e.g. keldysh_didv's finite difference, or an iv_curve sweep) helps
    further since the one-time build cost is paid only once. If the
    interpolant doesn't converge within its (deliberately modest, single-
    call-sized) budget, this falls back to "direct" automatically rather
    than risking a large, possibly-losing build -- see build_selfenergy_aaa.
    "direct" restores the old per-energy behavior unconditionally.

    `selfenergy_qtci`, if given explicitly, overrides `selfenergy_method`
    entirely: pass a {lead: interpolant} dict (from build_selfenergy_aaa
    or build_selfenergy_qtci) built to cover at least this call's
    voltage/nmax_max, to reuse one interpolant across several dc_current
    calls instead of each one building (and immediately discarding) its
    own.

    A wide sideband window (large nmax_max/voltage packing many Andreev/
    MAR resonances into the interpolated energy range) can need enough
    AAA support points that a single, not-reused dc_current call breaks
    even with -- or even loses to -- direct per-energy solves; the
    interpolant built automatically here therefore uses a deliberately
    modest budget (build_selfenergy_aaa's defaults), and if either lead's
    fit doesn't converge within that budget, this call falls back to
    `selfenergy_method="direct"` rather than paying for an expensive,
    possibly-losing build. A caller that KNOWS it will reuse the
    interpolant across many calls (e.g. an iv_curve sweep) should build
    one explicitly with a larger budget (build_selfenergy_aaa(...,
    ncand_max=...)) and pass it as `selfenergy_qtci` instead of relying on
    this automatic, single-call-sized default."""
    if voltage == 0.:
        return 0.0
    if selfenergy_qtci is None and selfenergy_method == "aaa":
        selfenergy_qtci = build_selfenergy_aaa(ht, voltage, nmax_max, delta=delta)
        if not all(s.converged for s in selfenergy_qtci.values()):
            selfenergy_qtci = None  # fall back to direct per-energy solves
    elif selfenergy_qtci is None and selfenergy_method != "direct":
        raise ValueError("selfenergy_method must be 'aaa' or 'direct', "
                          f"got {selfenergy_method!r}")
    ht = _prepare_bias_target(ht)
    _check_supported(ht)
    if delta is None:
        delta = ht.delta
    lead0 = ht.lead if _is_localprobe(ht) else ht.Hl
    # complex128, matching Gr00/sigL_less's dtype: _integrand_trace_sum_jit
    # (numba) requires matching operand dtypes for @, and casting once
    # here (tauz is a per-call constant) avoids paying an array copy on
    # every one of the (up to tens of thousands of) quasienergy evaluations.
    tauz = algebra.todense(lead0.get_operator("tauz").get_matrix()).astype(np.complex128)
    cache = {}
    # _prepare_system(ht) only depends on ht (never on quasienergy, nmax or
    # voltage) but current_integrand is called once per quadrature point
    # below (tens to hundreds of times per dc_current call, across the
    # adaptive nmax loop too) -- compute it once here instead of redoing
    # the electron/hole-projector and local-Hamiltonian extraction work on
    # every single evaluation.
    system = _prepare_system(ht)

    def integral(nmax):
        f = lambda e: current_integrand(ht, voltage, e, nmax, tauz,
                                         delta=delta, temperature=temperature,
                                         cache=cache, system=system,
                                         selfenergy_qtci=selfenergy_qtci)
        val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=1e-3)
        return val

    prev = integral(nmax)
    streak = 0
    converged = False
    while nmax < nmax_max:
        nmax += 2
        cur = integral(nmax)
        denom = max(abs(cur), abs(prev), 1e-12)
        streak = streak+1 if abs(cur-prev)/denom < tol else 0
        prev = cur
        if streak >= min_consecutive:
            converged = True
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
