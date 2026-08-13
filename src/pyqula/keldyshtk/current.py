import warnings
from functools import lru_cache

import numpy as np
from numba import jit, prange
from scipy.integrate import quad

from .. import algebra
from ..algebra import dagger
from ..transporttk.smatrix import enlarge_hlist
# sets numba.config.THREADING_LAYER = 'workqueue' (fork-safe) before any
# parallel=True numba function in this module gets compiled/run -- must be
# imported ahead of _assemble_chain_batch_jit/_rgf_chain_batch_jit/
# _integrand_trace_sum_batch_jit below (see greentk/rg.py, same pattern).
from .. import parallel

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


def _prefetch_selfenergies_batch(ht, es, lead, delta, cache, keys):
    """Batch-compute every not-yet-cached selfenergy of one lead across a
    whole set of sideband energies at once (see transporttk/selfenergy.py:
    get_selfenergy_batch and greentk/rg.py:green_renormalization_jit_batch)
    instead of one sideband at a time: for a fixed quasienergy, the
    `2*nmax+1` sidebands only differ in energy for the same fixed lead, so
    they are embarrassingly parallel and are run over a numba `prange`
    loop across threads. `keys[i]` is the caller's precomputed `(lead,
    round(es[i],10))` cache key for `es[i]` -- passed in rather than
    recomputed here since the caller (_batch_selfenergy) needs the same
    keys again afterward to gather results, and round() was measured to
    be a meaningful fraction (~20%) of a dc_current call's total time
    (214200 calls in one profiled case) when computed twice per energy."""
    if not hasattr(ht, "get_selfenergy_batch"):
        return  # e.g. LocalProbe: fall back to per-energy solves in the caller
    miss = [i for i, k in enumerate(keys) if k not in cache]
    if not miss: return
    me = np.array([es[i] for i in miss])
    outs = ht.get_selfenergy_batch(me, lead=lead, delta=delta, pristine=True)
    for i, out in zip(miss, outs):
        cache[keys[i]] = algebra.todense(out)


def _batch_selfenergy(ht, es, lead, delta, cache, selfenergy_qtci=None):
    """Self-energy of `lead` at every energy in `es` (1D array), returned
    as one (len(es),dim,dim) array -- replaces a Python loop of per-energy
    calls (each a dict lookup + round() cache key + dispatch) with a
    single batched call wherever the underlying evaluator supports one.
    This matters once the per-energy evaluation itself is cheap (an AAA
    interpolant: ~2us) -- profiling a dc_current call showed the
    surrounding Python-level per-site dispatch, not that evaluation, had
    become the dominant cost (see _floquet_green_functions's docstring).

    An AAA interpolant (aaatk.selfenergy_aaa.SelfenergyAAA, `selfenergy_
    qtci`'s default contents -- see dc_current's `selfenergy_method="aaa"`)
    exposes `call_batch`, evaluating every energy in one compiled call;
    anything else sharing the `interp(energy)->matrix` contract (e.g. a
    qtcitk.selfenergy_qtci.SelfenergyQTCI interpolant, which only defines
    scalar __call__) falls back to a plain Python loop over it -- still
    correct, just without the batching win.

    With no interpolant at all (`selfenergy_qtci` not covering `lead`),
    `_prefetch_selfenergies_batch` fills `cache` with one batched Sancho-
    Rubio solve per lead where available (falls back to solving lazily,
    one at a time below, for e.g. a LocalProbe -- see that function)."""
    es = np.asarray(es)
    if selfenergy_qtci is not None and lead in selfenergy_qtci:
        interp = selfenergy_qtci[lead]
        if hasattr(interp, "call_batch"):
            return interp.call_batch(es)
        return np.array([interp(e) for e in es], dtype=np.complex128)
    keys = [(lead, round(e, 10)) for e in es]
    _prefetch_selfenergies_batch(ht, es, lead, delta, cache, keys)
    out = []
    for i, k in enumerate(keys):
        v = cache.get(k)
        if v is None:
            v = algebra.todense(ht.get_selfenergy(es[i], lead=lead, delta=delta,
                                                    pristine=True, numba=True))
            cache[k] = v
        out.append(v)
    return np.array(out, dtype=np.complex128)


@jit(nopython=True, cache=True)
def _assemble_chain_jit(es, sigR0, sigR1, hii0, hii1, ve, vhd, delta,
                         temperature, start_block):
    """Build one Floquet chain's per-site (Es, SigLess, taus) arrays --
    the inputs _rgf_chain_jit needs -- plus its block-0-owned (sigL_less,
    sigL_a) entries, directly from precomputed self-energy arrays. This
    replaces a Python-level loop over (block,sideband) sites that called
    dagger()/_fermi_scalar()/a dict-cache lookup once per site: profiling
    a dc_current call on a deep-subgap junction (large nmax) found that
    loop's own body -- not the RGF sweep, not the self-energy solve --
    was the dominant cost (~58% of total wall time), because per-call
    Python/numba dispatch overhead, paid ~150000 times per call, exceeded
    the actual arithmetic once self-energies were already cheap to obtain
    (see aaatk/selfenergy_aaa.py's _eval_matrix_batch_jit docstring for
    the same finding from the self-energy side). Compiling the whole
    per-site assembly removes that overhead instead of only the self-
    energy evaluation's own share of it.

    `sigR0`/`sigR1` are (ns,dim,dim) arrays of each lead's self-energy at
    every quasienergy `es[k]`, shared by BOTH chains: chainA and chainB
    visit the exact same sideband energies `quasienergy+n*voltage` (see
    the module docstring's chain decomposition), just with which block
    owns each site swapped between them, so both chains index into the
    same two precomputed arrays rather than needing their own. `hii0`/
    `hii1` are the two blocks' onsite Hamiltonians, `ve`/`vhd` the AC bond
    hopping from a block-0 site to the next (electron-projected) and from
    a block-1 site to the next (hole-projected, dagger'd). `start_block`
    (0 or 1) is which block owns site k=0; block ownership then alternates
    with k, matching the n=-nmax+k, b=(start_block if k even else
    1-start_block) convention the removed _chain_sites helper used to
    build explicitly."""
    ns = es.shape[0]
    dim = hii0.shape[0]
    iden = np.eye(dim, dtype=np.complex128)
    Es = np.empty((ns, dim, dim), dtype=np.complex128)
    SigLess = np.empty((ns, dim, dim), dtype=np.complex128)
    taus = np.empty((ns-1, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((ns, dim, dim), dtype=np.complex128)
    for k in range(ns):
        e = es[k]
        is_block0 = (k % 2 == 0) if start_block == 0 else (k % 2 == 1)
        if is_block0:
            sig_r = sigR0[k]
            hii = hii0
        else:
            sig_r = sigR1[k]
            hii = hii1
        sig_r_dag = np.conjugate(sig_r).T
        Es[k] = (e+1j*delta)*iden - hii - sig_r
        if temperature == 0.:
            if e < 0.: f = 1.0
            elif e > 0.: f = 0.0
            else: f = 0.5
        else:
            f = 1.0/(1.0+np.exp(e/temperature))
        sl = -f*(sig_r - sig_r_dag)
        SigLess[k] = sl
        if is_block0:
            sigL_less[k] = sl
            sigL_a[k] = sig_r_dag
        if k < ns-1:
            taus[k] = ve if is_block0 else vhd
    return Es, SigLess, taus, sigL_less, sigL_a


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


def _prepare_chain_consts(system):
    """The per-`system` (not per-quasienergy/nmax) pieces of a Floquet
    chain -- the AC-bond hoppings `ve`/`vhd` and the two blocks' onsite
    Hamiltonians `hii0`/`hii1` -- factored out of `_floquet_green_
    functions` so they can be computed ONCE per `dc_current` call (like
    `system` itself) instead of rebuilt (a `todense`/projector matmul
    each) on every one of its tens-to-hundreds of quasienergy x nmax-step
    calls: profiling a slow deep-subgap case found this rebuild was a
    real, avoidable share of `_floquet_green_functions`'s own per-call
    overhead once the self-energy and chain-assembly costs it originally
    targeted were already batched away."""
    hlist, proje, projh, dim = system
    v0 = algebra.todense(hlist[1][0])  # hopping <lead1 unit cell|H|lead0 unit cell>
    ve = (proje@v0).astype(np.complex128)  # electron-projected AC bond, sideband n -> n+1
    vh = projh@v0  # hole-projected AC bond, couples sideband n -> n-1
    vhd = dagger(vh).astype(np.complex128)
    hii0 = algebra.todense(hlist[0][0]).astype(np.complex128)
    hii1 = algebra.todense(hlist[1][1]).astype(np.complex128)
    return ve, vhd, hii0, hii1, dim


def _floquet_green_functions(ht, voltage, quasienergy, nmax, delta,
                              temperature, cache, system,
                              selfenergy_qtci=None, chain_consts=None):
    """Retarded and lesser Green's function diagonal blocks at every
    block-0 (left-lead-type) sideband, together with the left lead's
    lesser/advanced self-energies (needed by the current trace). Builds
    the two decoupled Floquet chains directly instead of assembling the
    dense (2*ns*dim)^2 Hamiltonian, and solves each with the O(ns)
    recursive sweep (_rgf_chain_jit) -- exact, not an approximation (see
    module docstring). `system` is the (hlist, proje, projh, dim) tuple
    from `_prepare_system(ht)`, precomputed once per `dc_current` call by
    the caller: it only depends on `ht` (never on quasienergy, voltage,
    nmax or delta), but this function is called once per quadrature point
    of the current integral, so recomputing it here -- which involves
    building electron/hole projector operators over `ht` and extracting
    local Hamiltonian blocks -- would redo the same work tens to hundreds
    of times per `dc_current` call for no benefit.

    Self-energies and the per-site (Es,SigLess,taus) chain arrays are both
    built as single batched/compiled calls (_batch_selfenergy,
    _assemble_chain_jit) rather than a Python loop over (block,sideband)
    sites: profiling a dc_current call on a deep-subgap junction (large
    nmax) found that loop's own body accounted for ~58% of total wall
    time, dwarfing both the RGF sweep and the self-energy solve/evaluation
    it was calling -- ~150000 tiny per-site Python calls (a dict-cache
    lookup+round() per self-energy, a dagger()/conjugate-transpose, a
    scalar Fermi evaluation) whose combined dispatch overhead exceeded the
    actual arithmetic once the self-energy side of this pipeline was
    already optimized (AAA interpolation, batched Sancho-Rubio) in an
    earlier round. chainA and chainB (the two decoupled chains the
    (block,sideband) Floquet lattice splits into -- see module docstring)
    visit the exact same set of sideband energies `quasienergy+n*voltage`,
    just with which block owns each site swapped between them, so one
    pair of (lead,ns)-sized self-energy arrays here is shared by both
    chains rather than each solving its own.

    `chain_consts`, if given, is `_prepare_chain_consts(system)` --
    precomputed once by the caller (see that function's docstring) instead
    of rebuilt on every call here; computed on demand if not given (e.g.
    a standalone caller, or boundary.py's validate_against_truncation)."""
    if chain_consts is None:
        chain_consts = _prepare_chain_consts(system)
    ve, vhd, hii0, hii1, dim = chain_consts
    ns = 2*nmax+1

    # Indexed by isb=n+nmax (0..ns-1), not a dict keyed by n: the sideband
    # index n ranges over a small contiguous [-nmax,nmax], so a dict here
    # only ever paid Python hashing/lookup overhead for what's really a
    # plain array index.
    es = np.array([quasienergy+(isb-nmax)*voltage for isb in range(ns)])
    sigR0 = _batch_selfenergy(ht, es, 0, delta, cache, selfenergy_qtci=selfenergy_qtci)
    sigR1 = _batch_selfenergy(ht, es, 1, delta, cache, selfenergy_qtci=selfenergy_qtci)

    Gr00 = np.empty((ns, dim, dim), dtype=np.complex128)
    Gless00 = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((ns, dim, dim), dtype=np.complex128)
    for start_block in (0, 1):  # chainA (0) and chainB (1), see module docstring
        Es, SigLess, taus, sl_less, sl_a = _assemble_chain_jit(
            es, sigR0, sigR1, hii0, hii1, ve, vhd, delta, temperature,
            start_block)
        G, Gless = _rgf_chain_jit(Es, taus, SigLess)
        # This chain owns block 0 at every other site, starting at
        # k=start_block (see _assemble_chain_jit's docstring); those are
        # exactly the entries Gr00/Gless00/sigL_less/sigL_a need, indexed
        # by isb=k directly since both chains share the same n=-nmax+k
        # sideband-energy assignment (only which block owns each k
        # differs between them).
        Gr00[start_block::2] = G[start_block::2]
        Gless00[start_block::2] = Gless[start_block::2]
        sigL_less[start_block::2] = sl_less[start_block::2]
        sigL_a[start_block::2] = sl_a[start_block::2]
    return Gr00, Gless00, sigL_less, sigL_a, dim, ns


def current_integrand(ht, voltage, quasienergy, nmax, tauz,
                       delta=1e-6, temperature=0., cache=None, system=None,
                       selfenergy_qtci=None, chain_consts=None):
    """Integrand Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} of the paper's
    Eq. for I_dc, at a fixed quasienergy. `tauz` is the electron/hole
    grading operator matching the left lead's unit-cell dimension.
    `system` is the precomputed `_prepare_system(ht)` tuple, see
    `_floquet_green_functions`; computed on demand if not given (e.g. for
    standalone callers/tests) so this stays a valid entry point on its
    own. `chain_consts`: see _prepare_chain_consts, likewise computed on
    demand if not given. `selfenergy_qtci`: see _batch_selfenergy/
    build_selfenergy_qtci."""
    if cache is None:
        cache = {}
    if system is None:
        system = _prepare_system(ht)
    Gr00, Gless00, sigL_less, sigL_a, dim, ns = _floquet_green_functions(
        ht, voltage, quasienergy, nmax, delta, temperature, cache, system,
        selfenergy_qtci=selfenergy_qtci, chain_consts=chain_consts)
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


@jit(nopython=True, parallel=True, cache=True)
def _assemble_chain_batch_jit(es2d, sigR0, sigR1, hii0, hii1, ve, vhd, delta,
                               temperature, start_block):
    """Batched version of _assemble_chain_jit: same per-site assembly, but
    over a whole quadrature's worth of quasienergy NODES at once (a leading
    `nq` axis added to every array), one independent chain per node, run
    over a numba `prange` loop across threads -- the same batch-over-an-
    independent-axis pattern already used for the sideband axis inside
    _batch_selfenergy's Sancho-Rubio fallback (greentk/rg.py:
    green_renormalization_jit_batch_core). `es2d[iq,k]` is quadrature
    node `iq`'s k-th sideband quasienergy (`quasienergies[iq]+(k-nmax)*
    voltage`); `sigR0`/`sigR1` are (nq,ns,dim,dim), one self-energy per
    node per sideband -- batched once, up front, for the whole node set by
    the caller (`_floquet_green_functions_batch`), not recomputed here.
    Only exists for `quadrature="fixed"` (see dc_current), whose node set
    is known before any integrand evaluation; `quadrature="adaptive"`'s
    node set is discovered one callback at a time by scipy.integrate.quad
    and cannot be batched this way."""
    nq = es2d.shape[0]
    ns = es2d.shape[1]
    dim = hii0.shape[0]
    iden = np.eye(dim, dtype=np.complex128)
    Es = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    SigLess = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    taus = np.empty((nq, ns-1, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    for iq in prange(nq):  # quadrature nodes are independent -> parallel
        for k in range(ns):
            e = es2d[iq, k]
            is_block0 = (k % 2 == 0) if start_block == 0 else (k % 2 == 1)
            if is_block0:
                sig_r = sigR0[iq, k]
                hii = hii0
            else:
                sig_r = sigR1[iq, k]
                hii = hii1
            sig_r_dag = np.conjugate(sig_r).T
            Es[iq, k] = (e+1j*delta)*iden - hii - sig_r
            if temperature == 0.:
                if e < 0.: f = 1.0
                elif e > 0.: f = 0.0
                else: f = 0.5
            else:
                f = 1.0/(1.0+np.exp(e/temperature))
            sl = -f*(sig_r - sig_r_dag)
            SigLess[iq, k] = sl
            if is_block0:
                sigL_less[iq, k] = sl
                sigL_a[iq, k] = sig_r_dag
            if k < ns-1:
                taus[iq, k] = ve if is_block0 else vhd
    return Es, SigLess, taus, sigL_less, sigL_a


@jit(nopython=True, parallel=True, cache=True)
def _rgf_chain_batch_jit(Es, taus, SigLess):
    """Batched version of _rgf_chain_jit: identical per-node O(ns) RGF
    sweep (see that function's docstring for the algorithm itself and its
    machine-precision validation against dense inversion), just run once
    per quadrature node over a `prange` loop instead of once per node via
    a separate Python-level call -- each node's chain is fully independent
    of every other node's (different quasienergy, no coupling between
    them), so this changes nothing about the math, only how many separate
    numba dispatches/Python-callback round trips it costs to get all of
    them. `Es`/`taus`/`SigLess` carry a leading `nq` axis (node index),
    all else as in _rgf_chain_jit."""
    nq = Es.shape[0]
    N = Es.shape[1]
    dim = Es.shape[2]
    G = np.empty((nq, N, dim, dim), dtype=np.complex128)
    Gless = np.empty((nq, N, dim, dim), dtype=np.complex128)
    for iq in prange(nq):  # quadrature nodes are independent -> parallel
        gL = np.empty((N, dim, dim), dtype=np.complex128)
        gLessL = np.empty((N, dim, dim), dtype=np.complex128)
        gL[0] = np.linalg.inv(Es[iq, 0])
        gLessL[0] = gL[0]@SigLess[iq, 0]@np.conjugate(gL[0]).T
        for i in range(1, N):
            t = taus[iq, i-1]
            td = np.conjugate(t).T
            sigl_r = t@gL[i-1]@td
            sigl_less = t@gLessL[i-1]@td
            gL[i] = np.linalg.inv(Es[iq, i]-sigl_r)
            gLd = np.conjugate(gL[i]).T
            gLessL[i] = gL[i]@(SigLess[iq, i]+sigl_less)@gLd
        gR = np.empty((N, dim, dim), dtype=np.complex128)
        gRless = np.empty((N, dim, dim), dtype=np.complex128)
        gR[N-1] = np.linalg.inv(Es[iq, N-1])
        gRless[N-1] = gR[N-1]@SigLess[iq, N-1]@np.conjugate(gR[N-1]).T
        for i in range(N-2, -1, -1):
            t = taus[iq, i]
            td = np.conjugate(t).T
            sigr_r = td@gR[i+1]@t
            sigr_less = td@gRless[i+1]@t
            gR[i] = np.linalg.inv(Es[iq, i]-sigr_r)
            gRd = np.conjugate(gR[i]).T
            gRless[i] = gR[i]@(SigLess[iq, i]+sigr_less)@gRd
        for i in range(N):
            Eeff = Es[iq, i].copy()
            SLtot = SigLess[iq, i].copy()
            if i > 0:
                t = taus[iq, i-1]
                td = np.conjugate(t).T
                Eeff = Eeff - t@gL[i-1]@td
                SLtot = SLtot + t@gLessL[i-1]@td
            if i < N-1:
                t = taus[iq, i]
                td = np.conjugate(t).T
                Eeff = Eeff - td@gR[i+1]@t
                SLtot = SLtot + td@gRless[i+1]@t
            G[iq, i] = np.linalg.inv(Eeff)
            Gd = np.conjugate(G[iq, i]).T
            Gless[iq, i] = G[iq, i]@SLtot@Gd
    return G, Gless


def _floquet_green_functions_batch(ht, voltage, quasienergies, nmax, delta,
                                    temperature, cache, system,
                                    selfenergy_qtci=None, chain_consts=None):
    """Batched version of _floquet_green_functions: solves EVERY quadrature
    node's pair of Floquet chains in one shot, over the leading node axis,
    instead of once per node via a Python callback (`current_integrand`
    called once per `scipy.integrate.quad` evaluation). Only meaningful
    with a node set known in advance (`quadrature="fixed"`, see dc_current
    and `_fixed_quasienergy_nodes`).

    Self-energies are still funneled through `_batch_selfenergy` exactly as
    before -- flattened over (node, sideband) into one (nq*ns,) array per
    lead per call, so the per-energy `cache` dict (shared across the whole
    dc_current call, including every nmax growth step) is populated/reused
    identically to the unbatched path; batching the chain solve does not
    change what gets cached or when. `quasienergies` is the fixed node
    array (nq,); returns the same five-tuple as _floquet_green_functions,
    each array carrying a new leading `nq` axis."""
    if chain_consts is None:
        chain_consts = _prepare_chain_consts(system)
    ve, vhd, hii0, hii1, dim = chain_consts
    ns = 2*nmax+1
    quasienergies = np.asarray(quasienergies)
    nq = quasienergies.shape[0]

    # es2d[iq,isb] = quasienergies[iq]+(isb-nmax)*voltage -- same convention
    # as _floquet_green_functions' `es`, one row per quadrature node.
    offsets = (np.arange(ns) - nmax) * voltage
    es2d = quasienergies[:, None] + offsets[None, :]
    es_flat = es2d.reshape(-1)
    sigR0 = _batch_selfenergy(ht, es_flat, 0, delta, cache,
                               selfenergy_qtci=selfenergy_qtci).reshape(nq, ns, dim, dim)
    sigR1 = _batch_selfenergy(ht, es_flat, 1, delta, cache,
                               selfenergy_qtci=selfenergy_qtci).reshape(nq, ns, dim, dim)

    Gr00 = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    Gless00 = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    sigL_less = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    sigL_a = np.empty((nq, ns, dim, dim), dtype=np.complex128)
    for start_block in (0, 1):  # chainA (0) and chainB (1), see module docstring
        Es, SigLess, taus, sl_less, sl_a = _assemble_chain_batch_jit(
            es2d, sigR0, sigR1, hii0, hii1, ve, vhd, delta, temperature,
            start_block)
        G, Gless = _rgf_chain_batch_jit(Es, taus, SigLess)
        Gr00[:, start_block::2] = G[:, start_block::2]
        Gless00[:, start_block::2] = Gless[:, start_block::2]
        sigL_less[:, start_block::2] = sl_less[:, start_block::2]
        sigL_a[:, start_block::2] = sl_a[:, start_block::2]
    return Gr00, Gless00, sigL_less, sigL_a, dim, ns


@jit(nopython=True, parallel=True, cache=True)
def _integrand_trace_sum_batch_jit(Gr00, sigL_less, Gless00, sigL_a, tauz):
    """Batched version of _integrand_trace_sum_jit: same per-sideband
    trace sum, over a leading `nq` (quadrature node) axis, one independent
    sum per node via `prange`. Returns a (nq,) complex array (one value per
    quadrature node) instead of a scalar."""
    nq = Gr00.shape[0]
    ns = Gr00.shape[1]
    out = np.empty(nq, dtype=np.complex128)
    for iq in prange(nq):  # quadrature nodes are independent -> parallel
        total = 0j
        for isb in range(ns):
            M = Gr00[iq, isb]@sigL_less[iq, isb] + Gless00[iq, isb]@sigL_a[iq, isb]
            MT = M@tauz
            tr = 0j
            for d in range(MT.shape[0]):
                tr += MT[d, d]
            total += tr
        out[iq] = total
    return out


# Node-chunk size for current_integrand_batch below. Peak memory of one
# unchunked call is ~13-18 live (nq,ns,dim,dim) complex128 arrays across
# _floquet_green_functions_batch/_assemble_chain_batch_jit/_rgf_chain_batch_jit
# (sigR0/sigR1/Gr00/Gless00/sigL_less/sigL_a/Es/SigLess/taus/sl_less/sl_a/
# G/Gless, briefly ~18 while the second start_block's set is built before
# the first is released) -- nq*ns*dim^2*16 bytes each. Left unchunked, a
# large nmax_max/voltage combination (e.g. nmax_max=64, voltage~1.0 ->
# nq~2700, ns~129) is order 1GB at dim=4 and scales up further with dim
# (more orbitals, or a LocalProbe). Chunking the *solve* over node groups
# of this size (not the final weighted sum, which stays one np.dot call
# over the whole array so the result is bit-identical to an unchunked
# call) bounds peak memory to a small, roughly dim-independent-at-typical-
# sizes constant while leaving _assemble_chain_batch_jit/_rgf_chain_batch_
# jit's own per-chunk numba dispatch overhead amortized across enough
# nodes to still batch effectively (prange saturates well below this).
_BATCH_CHUNK_NODES = 256


def current_integrand_batch(ht, voltage, quasienergies, nmax, tauz,
                             delta=1e-6, temperature=0., cache=None,
                             system=None, selfenergy_qtci=None,
                             chain_consts=None, chunk_size=_BATCH_CHUNK_NODES):
    """Batched version of current_integrand: the same per-quasienergy
    integrand `Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz}`, evaluated at
    every quasienergy in `quasienergies` (nq,) via `_floquet_green_
    functions_batch`, returned as an (nq,) real array instead of one
    scalar per call. Used by dc_current's `quadrature="fixed"` path (see
    `_fixed_quasienergy_nodes`) to replace nq separate Python-callback
    evaluations (one per scipy.integrate.quad node under
    `quadrature="adaptive"`) with batched/compiled calls.

    Solved in groups of at most `chunk_size` nodes (see
    `_BATCH_CHUNK_NODES`'s comment for the memory reasoning), not all `nq`
    nodes in one shot: each node's chain is fully independent of every
    other node's, so splitting the node axis into chunks does not change
    any individual node's result -- only how much peak memory solving them
    costs. Every element of the returned array is computed by exactly one
    chunk, in the same order as `quasienergies`, so the result is
    bit-identical to an unchunked call regardless of `chunk_size` (verified
    in tests/keldysh/test_batched_fixed_quadrature.py)."""
    if cache is None:
        cache = {}
    if system is None:
        system = _prepare_system(ht)
    if tauz.dtype != np.complex128:
        tauz = tauz.astype(np.complex128)
    quasienergies = np.asarray(quasienergies)
    nq = quasienergies.shape[0]
    out = np.empty(nq, dtype=np.float64)
    for start in range(0, nq, chunk_size):
        end = min(start+chunk_size, nq)
        Gr00, Gless00, sigL_less, sigL_a, dim, ns = _floquet_green_functions_batch(
            ht, voltage, quasienergies[start:end], nmax, delta, temperature,
            cache, system, selfenergy_qtci=selfenergy_qtci,
            chain_consts=chain_consts)
        out[start:end] = _integrand_trace_sum_batch_jit(
            Gr00, sigL_less, Gless00, sigL_a, tauz).real
    return out


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


def _leads_share_selfenergy(ht, delta, erange):
    """Cheap empirical check (a handful of true solves, not a full AAA
    build) for whether lead 0 and lead 1 have the IDENTICAL self-energy
    as a function of energy -- so build_selfenergy_aaa can build ONE
    interpolant and reuse it for both instead of paying for two
    independent builds (each hundreds to thousands of true solves).

    Comparing the raw intra/inter/extra_delta attributes directly would
    be unsafe: heterostructures.create_leads_and_central_list stores the
    left lead's inter as dagger(h_left.inter) while the right lead's is
    h_right.inter unchanged (see heterostructures.py) -- so even a
    literally-identical physical lead built on both sides (e.g. two
    separately-constructed but equal Hamiltonians passed to
    heterostructures.build) has non-identical raw matrices whenever
    `inter` isn't Hermitian (multi-orbital, SOC, or any directional
    hopping). Only a numerical comparison of the actual self-energy
    catches the symmetric case correctly without either false negatives
    (physically-identical leads, differently-stored) or false positives.

    Also correctly returns False for a LocalProbe without any special
    case: LocalProbe.get_selfenergy's lead=0 (probe surface GF) and
    lead=1 (bulk sample-site GF, a completely different Green's-function
    calculation, `local_selfenergy`) are different physics by
    construction and will not match numerically at any sampled energy."""
    probes = (0.0, 0.37*erange, -0.61*erange)
    for e in probes:
        s0 = algebra.todense(ht.get_selfenergy(e, lead=0, delta=delta,
                                                pristine=True, numba=True))
        s1 = algebra.todense(ht.get_selfenergy(e, lead=1, delta=delta,
                                                pristine=True, numba=True))
        if s0.shape != s1.shape or not np.allclose(s0, s1, rtol=1e-10, atol=1e-12):
            return False
    return True


def build_selfenergy_aaa(ht, voltage, nmax_max, delta=None,
                          tolerance=1e-3, **kwargs):
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
    fallback to direct solves if that budget isn't enough to converge.

    When `ht` exposes `get_selfenergy_batch` (Heterostructure does; a
    LocalProbe does not), every candidate/validation round's true solves
    are routed through it -- the numba prange-parallel Sancho-Rubio
    iteration (transporttk.selfenergy.get_selfenergy_batch, greentk.rg.
    green_renormalization_jit_batch) instead of one Python-level call per
    energy -- via SelfenergyAAA's own `get_selfenergy_batch` argument.
    This is a pure speedup of the *build* (same solves, same rounds, same
    resulting fit -- see SelfenergyAAA.__init__'s docstring), not a change
    to the interpolant's accuracy or the number of true solves needed.

    If the two leads turn out to have the identical self-energy as a
    function of energy (checked empirically, see _leads_share_selfenergy
    -- a symmetric junction, e.g. the same physical lead on both sides of
    heterostructures.build), only ONE interpolant is actually built and
    `out[1] is out[0]` (the same object, not a copy) -- halving the build
    cost for that common case. Never true for a LocalProbe (lead 0 is the
    probe's own surface GF, lead 1 is the bulk sample-site GF it couples
    to -- different physics by construction), so this never risks handing
    a LocalProbe the wrong lead's self-energy.

    When both leads DO need independent builds, they run concurrently in
    two threads rather than one after the other: each build's actual work
    (the numba prange-parallel batched Sancho-Rubio solve above, and the
    SVD inside aaa()'s LAPACK call) releases the GIL, so two builds
    genuinely overlap instead of just interleaving Python bytecode -- the
    two SelfenergyAAA objects are fully independent (their own local
    `solved` cache, no shared mutable state), and the only shared object
    they both read from, `ht`, is not mutated by get_selfenergy/
    get_selfenergy_batch for a Heterostructure, and touches disjoint cache
    keys (keyed on `lead`) for a LocalProbe's optional reuse_selfenergy
    cache -- so this is safe with no locking needed."""
    ht = _prepare_bias_target(ht)
    _check_supported(ht)
    if delta is None: delta = ht.delta
    system = _prepare_system(ht)
    hlist, proje, projh, dim = system
    erange = (nmax_max+1)*abs(voltage)
    from ..aaatk.selfenergy_aaa import SelfenergyAAA
    shared = _leads_share_selfenergy(ht, delta, erange)
    leads = (0,) if shared else (0, 1)

    def build_one(lead):
        def get_se(e, lead=lead): # default arg freezes the loop variable
            return ht.get_selfenergy(e, lead=lead, delta=delta,
                                     pristine=True, numba=True)
        get_se_batch = None
        if hasattr(ht, "get_selfenergy_batch"):
            def get_se_batch(es, lead=lead): # default arg freezes the loop variable
                return ht.get_selfenergy_batch(es, lead=lead, delta=delta,
                                                pristine=True)
        return SelfenergyAAA(get_se, dim, -erange, erange, delta,
                              tolerance=tolerance,
                              get_selfenergy_batch=get_se_batch, **kwargs)

    if len(leads) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(leads)) as ex:
            out = dict(zip(leads, ex.map(build_one, leads)))
    else:
        out = {leads[0]: build_one(leads[0])}
    if shared: out[1] = out[0]
    return out


def build_shared_selfenergy(ht, vmax, nmax_max=40, delta=None, dv=None, **kwargs):
    """Build one aaatk.selfenergy_aaa.SelfenergyAAA interpolant per lead
    (see build_selfenergy_aaa), sized to cover every voltage magnitude up
    to `vmax` that an upcoming SWEEP of dc_current/didv calls could reach,
    for sharing across that whole sweep instead of each call independently
    building (and discarding) its own default fit.

    Returns None (never raises) if `ht` is not a Floquet-Keldysh-eligible
    junction (both leads, or a LocalProbe's probe+sample, superconducting
    -- see transporttk.didv._both_leads_superconducting) or if the AAA fit
    doesn't converge within its default build budget; callers should fall
    back to their ordinary per-call default in either case, exactly like
    dc_current's own selfenergy_method="aaa" fallback contract. Otherwise
    returns the raw {0: interpolant, 1: interpolant} dict build_selfenergy_
    aaa itself returns -- pass it as the `selfenergy_qtci` kwarg to every
    dc_current/didv call in the sweep.

    `dv`, if the caller knows it will pass an explicit dv through to
    keldysh_didv (rather than keldysh_didv's own default dv formula), must
    be given here too: the window must cover every call's voltage+-dv, not
    just its voltage, or SelfenergyAAA -- which enforces no domain, only
    warns once -- would silently extrapolate for any call whose dv pushes
    it past vmax.
    With dv=None (the common case) this assumes keldysh_didv's own default,
    max(voltage*1e-2,1e-3), evaluated at the worst case (vmax itself),
    which safely covers every smaller voltage in the sweep too.

    This factors out the pattern kappa.py's _shared_selfenergy_for_branch
    originally grew for its own finite-temperature coupling/energy sweep
    (see that function, now a thin wrapper around this one); iv_curve and
    thermaldidv.finite_T_didv use it directly. It is worth doing even for
    a SINGLE finite_T_didv call: that call's own internal thermal
    quadrature alone was measured to make 147 independent didv evaluations
    (temp=0.02), each previously building and discarding its own fit --
    almost entirely redundant since all 147 share the same two leads and
    only need self-energies over one common, boundable energy window. A
    smaller-scale check sharing one build across just 8 independent
    energies (examples/transport-scale settings, nmax_max=6) still showed
    a ~1.65x wall-clock win even at that modest a multiplier; the payoff
    grows with sweep size since the fixed build cost amortizes further.
    The built interpolant is a small, plain-numpy-backed object (dict of
    SelfenergyAAA instances) that pickles cleanly through both stdlib
    pickle and the `multiprocess`/dill pickler pcall's worker pool uses,
    so it can be shared into parallel sweeps too, not just serial ones."""
    from ..transporttk.didv import _both_leads_superconducting
    if not _both_leads_superconducting(ht):
        return None
    if delta is None: delta = ht.delta
    margin = dv if dv is not None else max(vmax*1e-2, 1e-3)
    shared = build_selfenergy_aaa(ht, vmax+margin, nmax_max, delta=delta, **kwargs)
    if not all(s.converged for s in shared.values()):
        return None
    return shared


# Default composite-Gauss-Legendre design for `quadrature="fixed"` below:
# equal-width panels of ABSOLUTE width `_FIXED_QUAD_PANEL_WIDTH` (not a
# fixed panel COUNT -- see below for why that first design was rejected),
# each with a 16-point Gauss-Legendre rule, covering [0,|voltage|].
# `npanels = max(_FIXED_QUAD_MIN_PANELS, ceil(|voltage|/_FIXED_QUAD_PANEL_WIDTH))`
# so the node count scales with the domain size instead of staying fixed.
#
# Chosen empirically (see documentation/keldysh_sideband_decimation_plan.md
# for the fuller investigation this summarizes -- item 2b, "replace the
# adaptive quasienergy quadrature with a fixed deterministic node set") in
# two stages:
#
# 1. A first design with a FIXED panel COUNT (60 panels regardless of
#    voltage, order 16, ~960 nodes always -- panel width ~0.0092, close to
#    this constant) was validated for ACCURACY against a sweep spanning
#    delta_sc in {0.1,0.3}, transparency in {0.3,0.6,1.0}, voltage in
#    {0.05,...,1.0} (both SC-SC and normal-normal junctions, 33/33 cases
#    within 4e-4 of adaptive quad), plus three robustness sweeps holding
#    one quantity fixed at a time: panel/singularity phase alignment
#    (voltage scanned across a full panel width at fixed delta_sc/
#    transparency/nmax -- worst relative error ~4e-4 over the period, the
#    binding accuracy constraint, since a fixed uniform grid cannot choose
#    where its panel boundaries fall relative to the delta_sc-mod-voltage
#    gap-edge feature); nmax (6 to 64 -- relative error flat to <1e-8
#    variation, so the node set does not need to depend on nmax); and
#    ht.delta broadening (1e-3 down to 1e-6 -- relative error saturates
#    rather than growing, confirming the dominant error is resolving the
#    singularity's algebraic tail at the panel-width scale, not its narrow
#    core, so the design does not degrade for a smaller delta). All three
#    are evidence for why a panel-width-scaled design should generalize,
#    not a re-run against the exact constant below.
#
# 2. That fixed-count design was then rejected on COST, not accuracy:
#    since node count does not shrink for a small |voltage|, it made the
#    doc's own deep-subgap representative case (voltage=0.1*delta_sc,
#    where "adaptive" needs very few quadrature points because the
#    integration domain itself is tiny) ~20x SLOWER than "adaptive" per
#    dc_current call, and a cheap normal-normal case ~31x slower --
#    regressions on exactly the cases this whole optimization effort
#    targets. Scaling panel count with |voltage| (the design actually
#    shipped here) fixes the small-voltage blowup: re-validated on the
#    same 33-case sweep plus one de-commensurated doc-case voltage (34/34
#    within 6e-4 of adaptive quad, worst case delta_sc=0.1/transparency=
#    0.3/voltage=0.15, a ~1.7x margin), and a clean (uncontended) wall-
#    clock benchmark of dc_current itself (selfenergy_method="direct")
#    gave: ~1.4x slower on the deep-subgap case (down from ~20x), ~1.9x
#    slower on the hardest SC-SC case in the sweep, and ~45x slower on the
#    cheap normal-normal case (a normal junction has no gap-edge
#    singularity at all, so a grid sized to resolve one is pure overkill
#    there, and telling the two apart at runtime would require inspecting
#    the pairing -- the gap-introspection this design deliberately avoids;
#    this regression is a design boundary, not a tuning miss). See the
#    plan doc for the full per-case node-count and wall-clock comparison
#    of both designs, and dc_current's own `quadrature` docstring for the
#    `keldysh_didv` finite-difference amplification check.
_FIXED_QUAD_PANEL_WIDTH = 0.006
_FIXED_QUAD_MIN_PANELS = 6
_FIXED_QUAD_ORDER = 16


@lru_cache(maxsize=8)
def _gauss_legendre_rule(order):
    """np.polynomial.legendre.leggauss(order), memoized -- called once per
    (order) the first time `_fixed_quasienergy_nodes` needs it, not
    recomputed on every dc_current call."""
    return np.polynomial.legendre.leggauss(order)


def _fixed_quasienergy_nodes(voltage, panel_width=_FIXED_QUAD_PANEL_WIDTH,
                              min_panels=_FIXED_QUAD_MIN_PANELS,
                              order=_FIXED_QUAD_ORDER):
    """Deterministic composite Gauss-Legendre quadrature nodes and weights
    covering [0, |voltage|], the same domain `dc_current`'s adaptive
    scipy.integrate.quad call integrates current_integrand over:
    `npanels = max(min_panels, ceil(|voltage|/panel_width))` equal-width
    panels, each with its own `order`-point Gauss-Legendre rule. A pure
    function of `(voltage, panel_width, min_panels, order)` only -- NOT of
    nmax, of the integrand's values, or of anything about `ht` (e.g. no
    lead-gap lookup) -- so the exact same node set is visited every time
    `dc_current` is called at the same voltage, satisfying the hard
    determinism requirement `quadrature="fixed"` exists for (see
    documentation/keldysh_sideband_decimation_plan.md): no per-call
    adaptive refinement, and (unlike scipy.integrate.quad's adaptive
    subdivision, whose node set can differ across nmax steps of the same
    dc_current call) identical nodes at every nmax the caller's adaptive
    sideband loop visits. On the selfenergy_method="direct" path (where
    _batch_selfenergy's dict cache is actually consulted -- "aaa" bypasses
    it via interp.call_batch), this means every nmax step after the first
    re-visits nodes already in that cache, where adaptive quad's per-step
    subdivision can land on different nodes and re-solve them instead;
    both quadratures' caches are already live across the whole dc_current
    call regardless, so the difference is hit RATE, not whether caching
    happens at all. This is a plausible structural difference, not a
    measured one -- the wall-clock benchmarks in dc_current's own
    `quadrature` docstring show "fixed" slower overall despite it.

    Returns (nodes, weights) as flat (npanels*order,) arrays; the integral
    is `float(np.dot(weights, f(nodes)))` for whatever integrand `f`."""
    x, w = _gauss_legendre_rule(order)
    V = abs(voltage)
    npanels = max(min_panels, int(np.ceil(V/panel_width)))
    edges = np.linspace(0., V, npanels+1)
    half = 0.5*(edges[1:]-edges[:-1])  # constant, all panels equal width
    mid = 0.5*(edges[1:]+edges[:-1])
    nodes = (mid[:, None] + half[:, None]*x[None, :]).ravel()
    weights = (half[:, None]*w[None, :]).ravel()
    return nodes, weights


def dc_current(ht, voltage, nmax=6, nmax_max=40, tol=1e-3, temperature=0.,
               delta=None, min_consecutive=2, selfenergy_qtci=None,
               selfenergy_method="direct", nmax_growth=1.5,
               fixed_nmax=None, return_nmax=False, quadrature="adaptive",
               quad_panel_width=_FIXED_QUAD_PANEL_WIDTH,
               quad_min_panels=_FIXED_QUAD_MIN_PANELS,
               quad_order=_FIXED_QUAD_ORDER):
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

    `nmax_growth` (default 1.5) grows each step geometrically
    (`nmax = max(nmax+2, ceil(nmax*nmax_growth))`) instead of a fixed
    `nmax += 2`: since `_floquet_green_functions` re-solves the whole
    `2*nmax+1`-site chain from scratch at every step (no incremental
    reuse across nmax -- see documentation/
    keldysh_sideband_decimation_plan.md for why that idea was
    investigated and shelved), a fixed increment makes the total
    resolved-chain-size summed over every step scale like nmax_max^2 (a
    deep-subgap case needing nmax~64 was measured to redo ~16x the work
    of a single nmax=64 solve). Geometric growth cuts that to a small
    constant factor (~3x for the same case) by visiting O(log nmax_max)
    steps instead of O(nmax_max) -- at the cost of coarser sampling of
    the nmax->current curve, which is not a robustness regression: the
    min_consecutive agreement check (previous paragraph) is if anything a
    *stronger* test of true convergence with widely-spaced steps, since a
    coincidental near-agreement between two nearby, small-Delta-nmax
    values (the failure mode min_consecutive guards against) is far less
    likely between two much-more-different windows. Pass nmax_growth<=1
    to recover the old fixed +=2 stepping exactly.

    `selfenergy_method` picks how lead self-energies are obtained:
    "direct" (default) solves Sancho-Rubio/bloch_selfenergy from scratch
    at every one of the (tens of thousands of) distinct (lead,energy)
    pairs the adaptive sideband sweep visits -- unconditionally correct,
    at the cost of not reusing work across nearby energies.

    "aaa" instead builds one aaatk.selfenergy_aaa.SelfenergyAAA
    interpolant per lead internally (see build_selfenergy_aaa), covering
    this call's voltage/nmax_max window, and evaluates that instead of
    solving from scratch at every energy -- a real speedup for a
    cheap-per-solve target like a 1D Sancho-Rubio self-energy (roughly
    break-even to ~40% faster for a single call, more when the
    interpolant is shared across several calls, e.g. keldysh_didv's
    finite difference or an iv_curve sweep -- see aaatk/selfenergy_aaa.py's
    module docstring for the performance measurements). NOT the default:
    documentation/keldysh_sideband_decimation_plan.md's "shared-nmax
    finite difference" update found a real accuracy gap (up to ~10%
    relative error in the current, growing with the sideband window
    size/nmax_max in the pre-fix implementation), root-caused in
    documentation/keldysh_aaa_selfenergy_accuracy_plan.md to
    under-resolved candidate points -- both at a lead's own gap-edge
    singularities and, more importantly for the current-error trend,
    across the fit's broader "bulk" domain -- NOT error compounding
    through the RGF chain (measured directly to attenuate, not amplify,
    a local self-energy error). SelfenergyAAA's held-out validation was
    itself confined to points near existing candidates and so did not
    detect this; both the validation sampling and the grid-refinement
    strategy were fixed accordingly (see that document's update log and
    `aaatk/selfenergy_aaa.py`'s `_refine_grid`). Still opt-in: even with
    the fix, a `converged=True` fit is only as good as `tolerance` for the
    specific system/parameters it was built for, and a hard enough target
    can still legitimately report `converged=False` (safe fallback to
    "direct" happens automatically in that case, see below) rather than
    ship a wrong answer.

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
    this automatic, single-call-sized default.

    `fixed_nmax`, if given, skips the adaptive nmax search entirely and
    solves the chain once at exactly this nmax (nmax/nmax_max/tol/
    min_consecutive/nmax_growth are all ignored). For sharing across a
    finite-difference pair (see transporttk.didv.keldysh_didv): the two
    biases differ by only `2*dv` (~1-2% of voltage), so their converged
    nmax is almost always identical, and same-nmax differencing also
    cancels systematic truncation error that the adaptive loop's own
    tol/min_consecutive guard does not otherwise control for.

    `return_nmax=True` returns `(current, nmax)` instead of just
    `current` -- `nmax` is the converged (or fixed_nmax's own) value,
    meant to be fed into a following call's `fixed_nmax`.

    `quadrature` picks how the outer quasienergy integral over
    `[0,|voltage|]` (the paper's Floquet-zone integral, evaluated once per
    adaptive-nmax step by `current_integrand`) is carried out. `"adaptive"`
    (default, unchanged behavior) uses `scipy.integrate.quad`'s adaptive
    21-point-Gauss-Kronrod-with-subdivision rule (`limit=50, epsrel=1e-3`);
    its node set can differ across nmax steps of the same call and is not
    reproducible without re-running the exact same call.

    `"fixed"` instead uses a deterministic composite Gauss-Legendre rule
    (`_fixed_quasienergy_nodes`): equal-width panels of absolute width
    `quad_panel_width` over `[0,|voltage|]` (at least `quad_min_panels` of
    them), each with its own `quad_order`-point Gauss-Legendre rule --
    defaulting to panel_width=0.006, min_panels=6, order=16, so e.g.
    |voltage|=0.55 uses 92 panels (1472 nodes) while |voltage|=0.03 uses
    the 6-panel floor (96 nodes); see `_FIXED_QUAD_PANEL_WIDTH`'s own
    comment for how this was chosen (and why panel width, not a fixed
    panel COUNT, is what scales with voltage). The node set is a pure
    function of `voltage` (and `quad_panel_width`/`quad_min_panels`/
    `quad_order`) only: identical across every nmax the adaptive sideband
    loop visits within one call, and reproducible call-to-call with no
    dependence on the integrand's runtime values -- what makes item 2c's
    batched-solver work (`current_integrand_batch`, see below) possible,
    since it needs the full node set in advance. Validated (documentation/
    keldysh_sideband_decimation_plan.md) to agree with "adaptive" to
    within its own epsrel=1e-3 tolerance (typically much better, ~1e-4 to
    1e-9; worst case over a 34-point sweep of SC-SC and normal-normal
    junctions, delta_sc in {0.1,0.3}, transparency in {0.3,0.6,1.0},
    voltage in {0.05,...,1.0}, was 5.8e-4, a ~1.7x margin). Robustness to
    nmax (6 to 64, flat to <1e-8 variation) and to the self-energy
    broadening `delta` (1e-3 to 1e-6, saturating rather than growing) was
    established on an earlier fixed-PANEL-COUNT design that shares this
    one's panel/order mechanism but not its exact width constant (see
    `_FIXED_QUAD_PANEL_WIDTH`'s comment for why panel width, not count,
    was the fix that shipped) -- supporting evidence for why the design
    should generalize, not a re-run of those two sweeps against this exact
    constant.

    NOT the default. Every quasienergy node's chain solve is now batched
    over a leading node axis (`current_integrand_batch` ->
    `_floquet_green_functions_batch` -> `_assemble_chain_batch_jit`/
    `_rgf_chain_batch_jit`, one numba `prange` call across nodes instead of
    one Python callback per node -- item 2c of documentation/
    keldysh_sideband_decimation_plan.md) rather than the one-callback-per-
    quad-evaluation loop item 2b originally shipped this mode with, solved
    in node chunks (`current_integrand_batch`'s `chunk_size`, default
    `_BATCH_CHUNK_NODES`) to bound peak memory rather than materializing
    every quadrature node's chain arrays at once. Batched per-node
    integrand values were checked bit-identical to the pre-batching
    per-node loop, and independent of `chunk_size`, in
    tests/keldysh/test_batched_fixed_quadrature.py.

    Speed, measured together (old per-node loop vs new batched-and-chunked
    vs "adaptive", same run, `selfenergy_method="direct"`, median of 5
    uncontended runs each -- the three numbers must come from one
    measurement session to be comparable; item 2b's own "fixed" vs
    "adaptive" ratios recorded elsewhere in this docstring/the plan doc do
    NOT reproduce under this session's load, e.g. its ~45x normal-normal
    slowdown measured ~8x here even before batching, so treat the two
    sessions' numbers as separate data points, not a single before/after
    series): on the doc's deep-subgap case, old-fixed 0.757s -> new-fixed
    0.606s (1.25x) vs adaptive's own 0.920s in the same run -- fixed is now
    faster than adaptive here. On the hardest SC-SC case in the validation
    sweep, old-fixed 5.566s -> new-fixed 1.466s (3.80x) vs adaptive's
    1.068s -- batching closes nearly all of the gap, fixed now ~1.4x
    slower. On a cheap normal-normal case (no gap-edge singularity, so
    "adaptive" solves it with as few as 21 points while "fixed" still pays
    for its full voltage-scaled panel count), old-fixed 1.726s -> new-fixed
    1.195s (1.44x) vs adaptive's 0.211s -- still ~5.7x slower, a real,
    structural gap for that case (telling it apart from a singular one at
    runtime would mean inspecting the pairing -- exactly the gap-
    introspection this design deliberately avoids), not a batching
    shortfall. Net: "fixed" is no longer a clear wall-clock loss on every
    case the way it was pre-batching, but "adaptive" (the default) remains
    faster on the normal-junction and hardest-SC-SC cases, so there is
    still no case where switching away from the default is a clear win by
    itself; "fixed"+batching mainly exists as infrastructure (a
    deterministic, cacheable node set) for callers that need
    reproducibility across calls, not as a general speed upgrade. A
    `keldysh_didv` finite-difference check on the deep-subgap case still
    found "fixed" and "adaptive" agreeing to 2.0e-4 relative in the
    resulting dI/dV -- no blow-up.

    The `prange` parallelism inside `_assemble_chain_batch_jit`/
    `_rgf_chain_batch_jit` only helps when numba can actually use multiple
    threads: inside a `parallel.pcall` worker (e.g. `iv_curve` with
    `cores>1`), `parallel.set_num_threads()` clamps numba to 1 thread per
    worker to avoid oversubscription, so the batching's speedup there comes
    only from fewer Python/numba dispatch round trips (still real, since
    that was a meaningful share of the old per-node-loop cost), not from
    the multi-threaded scaling the single-process benchmarks above show.

    `quad_panel_width`/`quad_min_panels`/`quad_order` are exposed for
    tuning/testing that node set, not meant to be hand-tuned per call."""
    if voltage == 0.:
        return (0.0, nmax) if return_nmax else 0.0
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
    # Same rationale as `system` above, one level down: chain_consts only
    # depends on `system`, not on quasienergy/nmax, but _floquet_green_
    # functions is called once per quadrature point x adaptive-nmax step.
    chain_consts = _prepare_chain_consts(system)

    if quadrature not in ("adaptive", "fixed"):
        raise ValueError(f"quadrature must be 'adaptive' or 'fixed', got {quadrature!r}")

    def integral(nmax):
        if quadrature == "fixed":
            nodes, weights = _fixed_quasienergy_nodes(
                voltage, panel_width=quad_panel_width,
                min_panels=quad_min_panels, order=quad_order)
            vals = current_integrand_batch(
                ht, voltage, nodes, nmax, tauz, delta=delta,
                temperature=temperature, cache=cache, system=system,
                selfenergy_qtci=selfenergy_qtci, chain_consts=chain_consts)
            return float(np.dot(weights, vals))
        f = lambda e: current_integrand(ht, voltage, e, nmax, tauz,
                                         delta=delta, temperature=temperature,
                                         cache=cache, system=system,
                                         selfenergy_qtci=selfenergy_qtci,
                                         chain_consts=chain_consts)
        val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=1e-3)
        return val

    if fixed_nmax is not None:
        prev = integral(fixed_nmax)
        return (prev, fixed_nmax) if return_nmax else prev

    prev = integral(nmax)
    streak = 0
    converged = False
    while nmax < nmax_max:
        nmax = min(nmax_max, max(nmax+2, int(np.ceil(nmax*nmax_growth))))
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
    return (prev, nmax) if return_nmax else prev


def iv_curve(ht, voltages, **kwargs):
    """Convenience wrapper: dc_current evaluated over an array of voltages,
    in parallel (see parallel.pcall).

    Unlike a single dc_current call (whose own default is
    selfenergy_method="direct" -- see its docstring for the AAA accuracy
    gap this used to raise, since fixed), a voltage sweep is exactly the
    workload the AAA fit's build cost (7-31s) is meant to amortize: this
    builds one shared AAA self-energy interpolant up front
    (build_shared_selfenergy), sized to cover every voltage in
    `voltages`, and reuses it for every dc_current call in the sweep
    instead of each call independently building (and discarding) its
    own -- so "aaa" is the default HERE, opposite of dc_current's own
    default, unless the caller explicitly passes selfenergy_method=
    "direct" to opt back out. Skipped if the caller already passed
    selfenergy_qtci explicitly (an explicit opt-out, so building a shared
    fit here would silently override the caller's own choice), or if the
    shared fit doesn't converge within budget (falls back to "direct"
    automatically, same safe-fallback contract as a plain dc_current
    call)."""
    from ..parallel import pcall
    if ("selfenergy_qtci" not in kwargs
            and kwargs.get("selfenergy_method", "aaa") == "aaa"
            and len(voltages)):
        nmax_max = kwargs.get("nmax_max", 40)
        vmax = max(abs(v) for v in voltages)
        shared = build_shared_selfenergy(ht, vmax, nmax_max=nmax_max,
                                          delta=kwargs.get("delta"))
        if shared is not None:
            kwargs["selfenergy_qtci"] = shared
    return np.array(pcall(lambda v: dc_current(ht, v, **kwargs), voltages))
