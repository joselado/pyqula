# JAX-differentiable counterpart of keldyshtk/current.py's Floquet-Keldysh
# DC current, for the workload that motivated it: zero-temperature dI/dV
# of a junction whose leads (or, for a LocalProbe, probe+sample) are both
# superconducting -- current.py's keldysh_didv gets dI/dV as a central
# finite difference of two independent dc_current() calls, each running
# its own scipy.integrate.quad adaptive quasienergy quadrature (with its
# own adaptive Floquet-sideband/nmax loop on top). This module instead
# vmaps the ENTIRE quasienergy quadrature for a FIXED nmax into one
# batched, jitted XLA computation and differentiates it directly with
# jax.grad -- no finite difference, no per-quadrature-node Python
# dispatch.
#
# Deliberately narrower in scope than current.py's dc_current/keldysh_didv:
#  - zero temperature only (didv(0.) short-circuits to 0.0 as a
#    documented physical fact, current() at any temperature!=0. is not
#    implemented and raises rather than silently ignoring it)
#  - nmax is FIXED, not adaptively grown: the caller picks it (as with
#    dc_current's own *starting* nmax, just without the growth loop --
#    each distinct nmax needs its own JIT trace since it changes every
#    array shape in the computation, so adaptively growing nmax here
#    would mean repeatedly paying JAX's compile cost, which is the one
#    genuinely expensive part of this whole approach -- see JaxKeldyshCurrent's
#    docstring)
#  - only reuses an already-built aaatk.selfenergy_aaa.SelfenergyAAA lead
#    self-energy interpolant (via current.build_selfenergy_aaa), never
#    solves Sancho-Rubio itself -- self-energy is re-expressed here purely
#    as an evaluation of that already-fitted barycentric rational form,
#    which is what makes it JAX-differentiable at no extra solve cost
#
# WHY A FIXED, MASKED QUASIENERGY GRID (not one rescaled to [0,V]):
# The first attempt used Gauss-Legendre nodes rescaled to the integration
# domain [0,V] (the natural choice, since the domain depends on V). The
# *value* I(V) came out fine, but jax.grad(I)(V) was numerical garbage --
# 0.38, 1.93, 0.77, -0.03 across quadrature orders 48/96/128/256 for a
# representative LocalProbe test case (SC probe+sample, T=0.3, nmax=8,
# voltage=0.25), not converging and occasionally the wrong sign, even
# though jax.grad matched a plain finite difference of that SAME
# discretized function at every order (so autodiff itself was not the
# bug). The cause: rescaling quadrature nodes with V means differentiating
# w.r.t. V also differentiates through node motion past this system's
# sharp MAR/Andreev resonances, which is numerically unstable. Switching
# to a grid whose node POSITIONS never depend on V -- a fixed
# Gauss-Legendre grid on the caller-supplied [0,vmax] window, masked to
# e<=|V| -- fixed this completely: both the value and the gradient
# converge cleanly as the grid is refined, and jax.grad again matches a
# finite difference of the (now well-behaved) discretized function.
#
# A MASKED QUADRATURE SUM SILENTLY DROPS THE LEIBNIZ BOUNDARY TERM:
# I(V) = int_0^|V| g(V,e)de has a voltage-dependent integration limit, so
# d/dV I(V) = sign(V)*g(V,|V|) + int_0^|V| d_V[g(V,e)]de (Leibniz's rule)
# -- but jax.grad of `sum(w * where(e_grid<=|V|,1,0) * g(V,e_grid))`
# ONLY ever produces the second (interior) term. Both branches of
# jnp.where are already-computed arrays that do not themselves depend on
# V, so autodiff sees no V-dependence through the mask and the
# moving-boundary contribution is not approximated -- it is silently
# absent, with no error or warning. This was caught by a plain
# normal-normal junction (turn_nambu, zero pairing; dc_current is exactly
# validated there against a non-Floquet static-bias reference in
# tests/keldysh/test_normal_junction_gauge_invariance.py): omitting the
# boundary term gave dI/dV off by 2 orders of magnitude and the wrong
# sign (-0.018 vs a reference of 1.53), even though current(V) itself
# (which needs no boundary term) matched to machine precision throughout
# -- current()'s agreement alone gave false confidence, and only checking
# didv() on a system simple enough to make a 100x error obvious surfaced
# this. It had been silently corrupting the SC-probe LocalProbe numbers
# this module was first developed against too, just by a smaller,
# easy-to-rationalize-away amount (see below) rather than an obvious one.
# _make_I_and_didv now adds sign(V)*g(V,|V|) back explicitly, evaluated
# directly (one more call to the same validated integrand, not through
# the quadrature sum).
#
# THE BOUNDARY TERM'S OWN EVALUATION POINT IS A DOUBLE NUMERICAL EDGE CASE:
# g(V,|V|) means quasienergy=|voltage| exactly, which (a) puts the
# outermost sideband's energy exactly at +-(nmax+1)*voltage, exactly the
# edge of the window build_selfenergy_aaa fits when sized to exactly that
# same value (confirmed: NaN self-energy exactly there) -- fixed with a
# small margin on the fitted window; and (b) since the sideband ladder's
# spacing is voltage and the probe energy is exactly |voltage|, ALWAYS
# puts some other sideband's energy at exactly e=0, a genuine self-energy
# singularity for some lead geometries already documented elsewhere in
# this codebase (tests/keldysh/test_andreev_linear_response.py) --
# confirmed here too (NaN at e=0 exactly, finite at even e=1e-6) and
# fixed by evaluating the boundary term at quasienergy=|voltage|-eps
# instead (eps=max(delta,|voltage|*1e-4), a negligible O(eps)
# approximation) -- the same fix scipy.integrate.quad gets "for free"
# from its Kronrod nodes never landing exactly on an interval endpoint.
# JaxKeldyshCurrent.__init__ verifies the boundary term is actually
# finite at both +vmax and -vmax after building, rather than trusting the
# margin/eps blindly -- both failures above were found exactly that way,
# not by reasoning about it in advance.
#
# WHY THE QUADRATURE ORDER MUST SCALE WITH nmax, NOT BE HARDCODED, AND WHY
# IT MUST BE VALIDATED WITH min_consecutive AND AT BOTH SIGNS OF VOLTAGE:
# current() converges fast and independently of the boundary-term issues
# above (no boundary term needed) -- e.g. nmax=16, voltage=0.25: gl_order
# 800 already matches a tightened-tolerance scipy reference to 2e-6
# relative, and each warm call takes under 1ms. didv() needs far more:
# a single-pair convergence check (comparing only two consecutive
# doublings, only at +vmax) was found to accept gl_order=1600 there --
# but two more doublings move it by another ~35%, the same false-
# convergence risk keldyshtk.current.dc_current's own adaptive nmax loop
# documents and guards against with min_consecutive=2. Checking only
# +vmax was also not enough: -vmax needed independent convergence, found
# converged at a different order than +vmax was. JaxKeldyshCurrent's
# search therefore requires min_consecutive agreeing doublings (default
# 2) at BOTH +vmax and -vmax before accepting an order, mirroring
# dc_current's own contract as closely as this fixed-order (not
# fixed-tolerance) setting allows.
#
# MEASURED NET EFFECT (LocalProbe, SC probe+sample, T=0.3, delta=1e-3,
# voltage=0.25, fixed nmax, direct method also at fixed nmax -- reproduced
# by examples/transport/keldysh_jax_benchmark/main.py, not hand-picked
# numbers; see this module's tests too). Once correctly implemented
# (boundary term + both numerical-edge-case fixes + a false-convergence-
# resistant search), the picture is LESS favorable than an earlier,
# incomplete version of this module suggested, and current() is NOT the
# large win a value-only microbenchmark (isolated from didv(), at a
# smaller, independently-tuned gl_order) can make it look like:
#   nmax= 8: current()  warm ~1.4s vs direct's ~1.1s (0.8x -- SLOWER)
#            didv()     warm ~1.4s vs direct's ~1.6s (1.1x -- about even)
#   nmax=16: current()  warm ~5.6s vs direct's ~3.7s (0.7x -- SLOWER)
#            didv()     warm ~5.5s vs direct's ~2.6s (0.5x -- 2x SLOWER)
# JIT compilation adds ~35-80s on top (mostly the gl_order search's
# several doublings, each its own compile), paid once per JaxKeldyshCurrent
# instance and amortized over every later call on it.
#
# WHY current() DOESN'T GET THE SPEED A VALUE-ONLY MICROBENCHMARK SHOWS:
# current() alone, at an independently-chosen, much smaller gl_order (800
# at both nmax=8 and 16, since the plain integral converges far faster
# than its derivative -- see above), IS sub-millisecond per warm call,
# thousands of times faster than dc_current. But JaxKeldyshCurrent.
# current() and .didv() are two views onto the SAME compiled function
# (_make_I_and_didv computes both together, since didv() needs current()'s
# machinery anyway and value_and_grad computes both in one pass at
# whatever cost the gradient's own accuracy demands) -- so as built here,
# current() pays the SAME large gl_order didv() needs, not the smaller one
# it would need on its own. Splitting these into two independently-tuned
# compiled paths (a cheap current()-only one, an expensive current()+
# didv() one) would recover current()'s standalone speed, but is not
# implemented; a caller who only ever wants I(V) and not dI/dV should
# consider that a known, understood, currently-unrealized opportunity
# rather than something this module already provides -- see the
# with-boundary-term/without-boundary-term microbenchmark path in this
# module's own development history (the finding above) if picking this up.
#
# WHAT THIS MEANS FOR A REAL WORKLOAD: for the resonance-rich system this
# module was developed and tested against, at the fixed-nmax settings
# above, NEITHER current() nor didv() as JaxKeldyshCurrent actually
# provides them today is a clear win over the direct numba path -- the
# honest range measured is roughly break-even to ~2x slower, the opposite
# of what an earlier round of this same investigation (before the boundary
# term bug and the two self-energy edge cases were found and fixed)
# reported. This joins qtcitk.selfenergy_qtci (see that module's own
# extensive docstring) as tested, documented infrastructure that did not
# pay off for the specific workload it was built for once measured
# rigorously, kept because a different system/regime -- one that
# converges at a smaller nmax, or where splitting current()/didv() into
# separately-tuned compiled paths is worth building -- may fare better;
# benchmark before assuming either way, and do not assume the module
# docstring's own earlier draft (visible in git history) was correct --
# it wasn't, in a way only caught by testing a system simple enough to
# make a 100x error obvious.
#
# ACCURACY AT SMALL nmax IS LIMITED BY nmax ITSELF, NOT BY THIS MODULE:
# with the boundary term now correctly included, current() and didv()
# both match the direct method (at the same fixed nmax) tightly for a
# well-behaved system (the normal-normal case above: both to machine/
# near-machine precision) and to ~0.1-1.2% for the harder SC-probe one.
#
# This sensitivity gets much more severe once you look for global
# properties rather than one point: I(-V)=-I(V) is an EXACT physical
# symmetry (validated for current.py's dc_current on an easier,
# normal-normal system in tests/keldysh/test_normal_junction_gauge_
# invariance.py, using adaptive nmax up to nmax_max=30) but at a FIXED,
# small nmax on this harder SC-probe LocalProbe system, checking it
# directly against current.dc_current itself (not against this module)
# gives I(+0.25)+I(-0.25) = -0.0247, ~40% of I(+0.25) -- the SAME
# violation (to 4 significant figures) this module's own JaxKeldyshCurrent
# shows at the same nmax. That rules out a JAX-specific bug: nmax=8 is
# simply too small to be quantitatively trustworthy for this particular
# system (matches the user guide's own warning that the sideband sum
# "converges slowly, especially deep below the combined gap at low
# transparency" -- see also keldyshtk.current.dc_current's docstring on
# non-monotonic nmax convergence). Both this module and the direct method
# are equally exposed to this; it is a property of the physical
# system/nmax choice, not of either implementation. Pick nmax generously
# (and prefer current.dc_current's adaptive-nmax path, or benchmark this
# module at several nmax and confirm current()/didv() have stopped moving,
# before trusting either method's absolute accuracy on a system like this
# one -- convergence in nmax is this module's known, not-yet-built,
# open piece; see JaxKeldyshCurrent's docstring for why it is fixed here.
#
# Requires the optional "jax" extra (pip install pyqula[jax]); only
# imported lazily by transporttk.didv/keldyshtk.current when a caller
# explicitly asks for method="keldysh_jax" or imports this module
# directly -- never a hard dependency of the core package.
import warnings

import numpy as np

import jax
# JAX defaults to float32/complex64, which would silently truncate this
# module's complex128 self-energy/Green's-function arithmetic -- must be
# set before any jax array is created (mirrors kpmtk/kpmjax.py's own
# x64 opt-in for the identical reason).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from .. import algebra


def _pack_selfenergy_aaa(selfenergy_aaa):
    """Extract a built aaatk.selfenergy_aaa.SelfenergyAAA's fitted
    barycentric coefficients into JAX arrays -- re-solving Sancho-Rubio
    in JAX is out of scope (its iteration count is data-dependent, an
    awkward fit for JAX's static-shape tracing); reusing an already-fitted
    rational interpolant sidesteps that entirely and is what makes this
    whole pipeline differentiable at no extra solve cost."""
    return dict(zj=jnp.array(selfenergy_aaa._zj_pad),
                wf=jnp.array(selfenergy_aaa._wf_pad),
                w=jnp.array(selfenergy_aaa._w_pad),
                ii=np.array(selfenergy_aaa._ii),
                jj=np.array(selfenergy_aaa._jj),
                dim=selfenergy_aaa.dim)


def _eval_selfenergy(packed, e):
    """Barycentric rational evaluation at (scalar, possibly traced) energy
    e, matching aaatk.selfenergy_aaa._eval_matrix_jit's numba kernel --
    except for its exact-support-point special case (e==zj exactly),
    which has probability zero for a continuous quadrature grid and is
    skipped here since JAX control flow can't branch on a traced value
    cheaply anyway."""
    zj, wf, w = packed["zj"], packed["wf"], packed["w"]
    c = 1.0 / (e - zj)             # padding entries have w=wf=0, contribute 0
    num = jnp.sum(wf * c, axis=1)
    den = jnp.sum(w * c, axis=1)
    vals = num / den
    out = jnp.zeros((packed["dim"], packed["dim"]), dtype=jnp.complex128)
    return out.at[packed["ii"], packed["jj"]].set(vals)


def _rgf_chain(Es, taus, SigLess):
    """JAX translation of keldyshtk.current._rgf_chain_jit -- exact
    O(N) recursive Green's function sweep for a 1D block-tridiagonal
    chain, faithful to the numba kernel's algorithm (validated against
    it to ~1e-14/1e-15 on synthetic chains up to N=65). The sequential
    forward/backward sweeps use lax.scan (a genuine loop-carried
    dependency, not vmappable); the final combine step is fully
    vectorized over the chain-site axis via zero-padded shifts instead
    of the numba kernel's explicit Python loop, since here N is a static
    (traced-shape) dimension jnp.linalg.inv/matmul already batch over."""
    N = Es.shape[0]
    dim = Es.shape[1]
    zero = jnp.zeros((dim, dim), dtype=Es.dtype)

    def fwd_step(carry, inputs):
        gL_prev, gLessL_prev = carry
        Ei, ti_prev, SigLess_i = inputs
        td = jnp.conj(ti_prev).T
        sigl_r = ti_prev @ gL_prev @ td
        sigl_less = ti_prev @ gLessL_prev @ td
        gLi = jnp.linalg.inv(Ei - sigl_r)
        gLessLi = gLi @ (SigLess_i + sigl_less) @ jnp.conj(gLi).T
        return (gLi, gLessLi), (gLi, gLessLi)

    gL0 = jnp.linalg.inv(Es[0])
    gLessL0 = gL0 @ SigLess[0] @ jnp.conj(gL0).T
    (_, _), (gL_rest, gLessL_rest) = lax.scan(
        fwd_step, (gL0, gLessL0), (Es[1:], taus, SigLess[1:]))
    gL = jnp.concatenate([gL0[None], gL_rest], axis=0)
    gLessL = jnp.concatenate([gLessL0[None], gLessL_rest], axis=0)

    def bwd_step(carry, inputs):
        gR_next, gRless_next = carry
        Ei, ti, SigLess_i = inputs
        td = jnp.conj(ti).T
        sigr_r = td @ gR_next @ ti
        sigr_less = td @ gRless_next @ ti
        gRi = jnp.linalg.inv(Ei - sigr_r)
        gRlessi = gRi @ (SigLess_i + sigr_less) @ jnp.conj(gRi).T
        return (gRi, gRlessi), (gRi, gRlessi)

    gRNm1 = jnp.linalg.inv(Es[N-1])
    gRlessNm1 = gRNm1 @ SigLess[N-1] @ jnp.conj(gRNm1).T
    (_, _), (gR_rest_rev, gRless_rest_rev) = lax.scan(
        bwd_step, (gRNm1, gRlessNm1),
        (Es[:-1][::-1], taus[::-1], SigLess[:-1][::-1]))
    gR = jnp.concatenate([gR_rest_rev[::-1], gRNm1[None]], axis=0)
    gRless = jnp.concatenate([gRless_rest_rev[::-1], gRlessNm1[None]], axis=0)

    taus_pl = jnp.concatenate([zero[None], taus], axis=0)            # taus_pl[i]=taus[i-1], i>=1
    gL_sh = jnp.concatenate([zero[None], gL[:-1]], axis=0)           # gL_sh[i]=gL[i-1], i>=1
    gLessL_sh = jnp.concatenate([zero[None], gLessL[:-1]], axis=0)
    taus_pr = jnp.concatenate([taus, zero[None]], axis=0)            # taus_pr[i]=taus[i], i<=N-2
    gR_sh = jnp.concatenate([gR[1:], zero[None]], axis=0)            # gR_sh[i]=gR[i+1], i<=N-2
    gRless_sh = jnp.concatenate([gRless[1:], zero[None]], axis=0)

    taus_pl_d = jnp.conj(jnp.swapaxes(taus_pl, -1, -2))
    taus_pr_d = jnp.conj(jnp.swapaxes(taus_pr, -1, -2))
    left_r = taus_pl @ gL_sh @ taus_pl_d
    left_less = taus_pl @ gLessL_sh @ taus_pl_d
    right_r = taus_pr_d @ gR_sh @ taus_pr
    right_less = taus_pr_d @ gRless_sh @ taus_pr

    Eeff = Es - left_r - right_r
    SLtot = SigLess + left_less + right_less
    G = jnp.linalg.inv(Eeff)
    Gless = G @ SLtot @ jnp.conj(jnp.swapaxes(G, -1, -2))
    return G, Gless


def _chain_sites(nmax):
    """Same decomposition as keldyshtk.current._chain_sites: the
    (block,sideband) Floquet lattice splits into two independent 1D
    chains of ns=2*nmax+1 sites each."""
    ns = 2*nmax+1
    chainA = [(0 if k % 2 == 0 else 1, -nmax+k) for k in range(ns)]
    chainB = [(1 if k % 2 == 0 else 0, -nmax+k) for k in range(ns)]
    return chainA, chainB


def _build_static_system(ht, nmax):
    """One-time (per nmax) extraction of everything the per-quasienergy
    trace needs that does NOT depend on quasienergy or voltage: onsite
    blocks, the AC-carrying bond's electron/hole-projected pieces, the
    Nambu grading operator, and the two chains' (block,sideband) site
    lists. `ht` must already be _prepare_bias_target-ed/_check_supported-ed
    (JaxKeldyshCurrent does this before calling here)."""
    from .current import _prepare_system, _is_localprobe
    hlist, proje, projh, dim = _prepare_system(ht)
    if len(hlist) != 2:
        # This reformulation is built on the two-chain decomposition of the
        # (block,sideband) lattice, which only holds for a junction with a
        # single spatial bond. keldyshtk.current handles a junction with an
        # explicit central region by inverting the full Floquet matrix
        # instead (_dense_floquet_integrand); there is no jax mirror of that
        # path, so refuse rather than silently solving the wrong chain.
        raise NotImplementedError(
            "JaxKeldyshCurrent only supports junctions with no explicit "
            "central region (a two-block chain); use "
            "Heterostructure.get_dc_current for this junction")
    v0 = algebra.todense(hlist[1][0])
    ve = proje @ v0
    vhd = algebra.dagger(projh @ v0)
    hii = [algebra.todense(hlist[0][0]), algebra.todense(hlist[1][1])]
    lead0 = ht.lead if _is_localprobe(ht) else ht.Hl
    tauz = algebra.todense(lead0.get_operator("tauz").get_matrix())
    return dict(hii=[jnp.array(h, dtype=jnp.complex128) for h in hii],
                ve=jnp.array(ve, dtype=jnp.complex128),
                vhd=jnp.array(vhd, dtype=jnp.complex128),
                tauz=jnp.array(tauz, dtype=jnp.complex128),
                dim=dim, nmax=nmax, chains=_chain_sites(nmax))


def _current_integrand(static, selfenergy_packed, voltage, quasienergy, delta):
    """Zero-temperature current_integrand, JAX version -- faithful
    translation of keldyshtk.current.current_integrand/
    _floquet_green_functions/_integrand_trace_sum_jit, validated against
    them to ~1e-11..1e-15 (nmax up to 16, several quasienergies including
    a sharp resonance). Returns a real JAX scalar; `quasienergy` and
    `voltage` may be traced (this is vmapped over quasienergy and
    differentiated w.r.t. voltage by the callers below)."""
    dim = static["dim"]
    iden = jnp.eye(dim, dtype=jnp.complex128)
    nmax = static["nmax"]
    ns = 2*nmax+1

    Gr00 = [None]*ns
    Gless00 = [None]*ns
    sigL_less = [None]*ns
    sigL_a = [None]*ns

    for chain in static["chains"]:
        N = len(chain)
        Es_list, SigLess_list, taus_list = [], [], []
        for k, (b, n) in enumerate(chain):
            e = quasienergy + n*voltage
            sig_r = _eval_selfenergy(selfenergy_packed[b], e)
            sig_r_dag = jnp.conj(sig_r).T
            Es_list.append((e+1j*delta)*iden - static["hii"][b] - sig_r)
            f = jnp.where(e < 0, 1.0, jnp.where(e > 0, 0.0, 0.5))  # T=0 Fermi step
            sl = -f*(sig_r - sig_r_dag)
            SigLess_list.append(sl)
            if b == 0:
                sigL_less[n+nmax] = sl
                sigL_a[n+nmax] = sig_r_dag
            if k < N-1:
                taus_list.append(static["ve"] if b == 0 else static["vhd"])
        Es = jnp.stack(Es_list)
        SigLess = jnp.stack(SigLess_list)
        taus = jnp.stack(taus_list)
        G, Gless = _rgf_chain(Es, taus, SigLess)
        for k, (b, n) in enumerate(chain):
            if b == 0:
                Gr00[n+nmax] = G[k]
                Gless00[n+nmax] = Gless[k]

    Gr00 = jnp.stack(Gr00); Gless00 = jnp.stack(Gless00)
    sigL_less = jnp.stack(sigL_less); sigL_a = jnp.stack(sigL_a)
    M = Gr00 @ sigL_less + Gless00 @ sigL_a
    tr = jnp.einsum('nij,ji->n', M, static["tauz"])
    return jnp.sum(tr).real


def _boundary_eps(voltage, delta):
    """Offset used to nudge the Leibniz boundary term off quasienergy=
    |voltage| exactly -- shared by _make_I_and_didv (which uses it) and
    JaxKeldyshCurrent.__init__ (which must verify finiteness at the SAME
    point that will actually be evaluated later, not the unshifted one,
    or the check tests the wrong thing entirely -- exactly the bug this
    factoring-out fixes)."""
    return max(delta, abs(voltage)*1e-4)


def _make_I_and_didv(static, packed, delta, vmax, gl_order):
    """Build the jitted (I(voltage), dI/dV(voltage)) function for a FIXED
    quasienergy grid: interior Gauss-Legendre nodes on (0,vmax) (never
    touching e=0, a documented numerical singularity of this formalism
    for some lead geometries -- see tests/keldysh/
    test_andreev_linear_response.py), masked to e<=|voltage| rather than
    rescaled to [0,|voltage|] -- see this module's docstring for why the
    rescaled-grid version was numerically unstable to differentiate.

    I(V) = int_0^|V| g(V,e)de is a Leibniz-rule integral with a
    voltage-dependent limit: d/dV I = sign(V)*g(V,|V|) + int_0^|V|
    d_V[g(V,e)]de. jax.grad of the masked-sum form above ONLY gives the
    second (interior) term: jnp.where(cond,a,b)'s two branches are both
    voltage-independent constants (the array of already-computed
    integrand values), so autodiff sees no voltage-dependence through the
    mask itself and the moving-boundary term is silently dropped, not
    approximated -- a real, structural gap, not a resolution issue no
    amount of extra gl_order fixes. Confirmed by a plain normal-normal
    junction (turn_nambu, zero pairing; dc_current is exactly validated
    there against a non-Floquet static-bias reference in tests/keldysh/
    test_normal_junction_gauge_invariance.py): omitting this boundary
    term gave dI/dV off by 2 orders of magnitude and the wrong sign
    (-0.018 vs a reference of 1.53) -- current(V) itself, which needs no
    boundary term, matched to machine precision throughout, which is why
    this was missed until dI/dV was checked on a system simple enough to
    make a 2-orders-of-magnitude error obvious. The boundary term is
    added back explicitly below, evaluated directly (not through the
    quadrature sum) since it is just one more call to the same
    (validated) integrand.

    `voltage` may be negative (I(-V)=-I(V), the same odd-symmetry
    contract dc_current satisfies -- see test_normal_junction_gauge_
    invariance.py); |voltage| must not exceed vmax.

    The boundary term is evaluated at quasienergy=|voltage|-eps rather
    than exactly |voltage| (eps = max(delta, |voltage|*1e-4), a negligible
    O(eps) approximation to the exact Leibniz term): landing exactly on
    quasienergy=|voltage| puts some sideband's energy at exactly
    +-(nmax+1)*voltage (its own domain-edge coincidence, distinct from the
    one below) AND, whenever |voltage| itself is an exact multiple of the
    sideband spacing (always true here, since quasienergy=|voltage| and
    voltage are the same value), some OTHER sideband lands at exactly
    e=0 -- confirmed to be a genuine self-energy singularity for some
    systems (not an artifact of this module: the same "avoid evaluating
    exactly at E=0" coincidence tests/keldysh/
    test_andreev_linear_response.py already documents for the direct
    method), giving NaN regardless of how wide the AAA fit's window is
    made. scipy.integrate.quad never hits this in the direct method only
    because Gauss-Kronrod's nodes are open/interior and never land exactly
    on a rational multiple of the integration range; nudging this
    boundary-term evaluation off the exact edge is the same fix applied
    to a point that has no quadrature rule protecting it."""
    nodes, weights = np.polynomial.legendre.leggauss(gl_order)  # interior points on [-1,1]
    e_grid = jnp.array((nodes+1.0)*vmax/2.0)                    # interior points on (0,vmax)
    w = jnp.array(weights*vmax/2.0)
    batched_integrand = jax.vmap(_current_integrand, in_axes=(None, None, None, 0, None))

    def I(voltage):
        vals = batched_integrand(static, packed, voltage, e_grid, delta)
        mask = jnp.where(e_grid <= jnp.abs(voltage), 1.0, 0.0)
        return jnp.sum(w*mask*vals)

    I_and_interior_grad = jax.value_and_grad(I)

    def I_and_didv(voltage):
        val, interior = I_and_interior_grad(voltage)
        eps = jnp.maximum(delta, jnp.abs(voltage)*1e-4)  # matches _boundary_eps
        boundary = jnp.sign(voltage) * _current_integrand(
            static, packed, voltage, jnp.abs(voltage)-eps, delta)
        return val, interior + boundary

    return jax.jit(I_and_didv)


class JaxKeldyshCurrent:
    """Compiled, JAX-differentiable Floquet-Keldysh DC current I(V) (and
    dI/dV via jax.grad, not a finite difference) for a FIXED Floquet
    sideband count `nmax` and a FIXED quasienergy window [0,vmax] -- built
    ONCE (paying JIT compilation, and an adaptive search for a quasienergy
    quadrature order accurate enough for both I and dI/dV, a single time),
    then cheap to evaluate/differentiate at any |voltage|<=vmax many times
    over. Mirrors aaatk.selfenergy_aaa.SelfenergyAAA's "build once,
    evaluate many" contract, and is meant to be used the same way: build
    one instance per (ht, nmax, vmax) combination a sweep needs, not one
    per voltage. See this module's docstring for measured performance:
    for the hardest system this module has been tested against, current()
    and didv() are both roughly break-even to ~2x slower than the direct
    numba path once correctly implemented (a correctly-included Leibniz
    boundary term and a false-convergence-resistant quadrature-order
    search both cost real accuracy corners an earlier, incomplete version
    of this module had cut) -- this is not a reliable speedup on its own
    for that workload; see the module docstring's "WHAT THIS MEANS FOR A
    REAL WORKLOAD" section before reaching for this over the direct path.
    Also see why nmax is fixed rather than adaptively grown here.

    `ht` must be current.py's own dc_current-supported shape: a two-lead
    Heterostructure with no explicit central region, or a LocalProbe (see
    keldyshtk.current._check_supported) -- unlike keldyshtk.current.
    build_shared_selfenergy (a Tier-1 helper that only bothers building an
    interpolant when both leads are genuinely superconducting, since
    otherwise didv() has a cheap smatrix fallback to use instead), this
    class has no such fallback and builds an interpolant via
    build_selfenergy_aaa unconditionally -- it works just as well on a
    trivial-pairing (turn_nambu, zero gap) normal-normal junction, where
    dc_current is validated to exactly reduce to ordinary Landauer
    transport (tests/keldysh/test_normal_junction_gauge_invariance.py),
    as on a genuinely superconducting one. Zero temperature only.

    `selfenergy_qtci`, if given, must be a {0:SelfenergyAAA,1:SelfenergyAAA}
    dict (e.g. from keldyshtk.current.build_selfenergy_aaa/
    build_shared_selfenergy) already covering [-vmax,vmax] at this `delta`
    -- reuse one across several JaxKeldyshCurrent instances (e.g. several
    nmax values on the same junction) to skip rebuilding it. If omitted,
    one is built automatically via build_selfenergy_aaa.

    `gl_order`, if given, fixes the quasienergy quadrature order and skips
    the adaptive search entirely (use this once you already know a value
    that converges both I and dI/dV for your regime, e.g. from a previous
    run's `.gl_order`, to avoid paying the search's extra compiles again).
    Otherwise the order is doubled from `gl_order0` (default
    max(200,50*nmax), an empirical starting point -- see module docstring)
    until dI/dV agrees, to within `tol`, between two consecutive orders
    for `min_consecutive` doublings in a row AND at both +vmax and -vmax
    (the widest, generally hardest points the intended [-vmax,vmax] range
    reaches -- probing only +vmax was found to leave -vmax's dI/dV
    converged to a visibly worse tolerance, since this system's resonance
    structure need not land symmetrically in quasienergy for +V vs -V),
    capped at `gl_order_max`. Requiring more than one consecutive
    agreeing step (rather than the last pair alone) guards against a
    lucky-but-not-really-converged doubling -- confirmed to happen here:
    a single-pair check accepted an order whose dI/dV was still ~35% away
    from where two more doublings settle, the same false-convergence risk
    keldyshtk.current.dc_current's own adaptive nmax loop documents and
    guards against with its own min_consecutive. A warning is issued
    (never a wrong answer silently) if gl_order_max is hit first."""

    def __init__(self, ht, nmax, vmax, delta=None, selfenergy_qtci=None,
                 gl_order=None, gl_order0=None, gl_order_max=6400, tol=2e-2,
                 min_consecutive=2):
        from .current import _prepare_bias_target, _check_supported, build_selfenergy_aaa
        ht = _prepare_bias_target(ht)
        _check_supported(ht)
        if delta is None: delta = ht.delta
        if vmax <= 0: raise ValueError("vmax must be > 0")
        self.ht, self.nmax, self.vmax, self.delta = ht, nmax, vmax, delta

        self._static = _build_static_system(ht, nmax)

        if selfenergy_qtci is None:
            # build_selfenergy_aaa(ht,v,nmax,...) fits [-(nmax+1)*v,(nmax+1)*v];
            # the boundary term's outermost sideband, evaluated at
            # quasienergy=|voltage|-eps (see _make_I_and_didv), reaches
            # +-((nmax+1)*vmax - eps), just inside that window -- a small
            # fixed margin on top is cheap, harmless insurance against
            # landing exactly on the edge (a real, separate failure mode
            # from the eps offset above: confirmed to give a non-finite
            # self-energy exactly at a fitted domain's boundary, distinct
            # from the e=0 issue the eps offset targets).
            selfenergy_qtci = build_selfenergy_aaa(ht, vmax*1.02, nmax, delta=delta)
        if not all(s.converged for s in selfenergy_qtci.values()):
            raise ValueError(
                "JaxKeldyshCurrent: the lead self-energy AAA fit did not converge "
                "within its default build budget for this vmax/nmax/delta. Build one "
                "explicitly with keldyshtk.current.build_selfenergy_aaa (a larger "
                "ncand_max/mmax_max budget) and pass it as selfenergy_qtci.")
        self._packed = {lead: _pack_selfenergy_aaa(s) for lead, s in selfenergy_qtci.items()}
        # Verify the boundary term is actually finite at both edges of the
        # range this instance claims to cover -- guards the caller against
        # a silent NaN/wrong answer rather than a fixed margin being
        # trusted blindly (both the e=0-type and domain-edge-type
        # failures above were found exactly this way).
        eps = _boundary_eps(vmax, delta)
        for sign in (1.0, -1.0):
            probe = _current_integrand(self._static, self._packed, sign*vmax, vmax-eps, delta)
            if not np.isfinite(np.array(probe)):
                raise ValueError(
                    f"JaxKeldyshCurrent: the Leibniz boundary term is non-finite at "
                    f"voltage={sign*vmax} for nmax={nmax}, delta={delta} -- likely a "
                    f"self-energy singularity coinciding with this exact (nmax,vmax) "
                    f"combination (see _make_I_and_didv's docstring). Try a slightly "
                    f"different vmax, or pass selfenergy_qtci built over a wider window.")

        if gl_order is not None:
            self.gl_order = gl_order
            self._I_and_didv = _make_I_and_didv(self._static, self._packed, delta, vmax, gl_order)
            return
        if gl_order0 is None: gl_order0 = max(200, 50*nmax)

        def probe(fn):
            """dI/dV at both +vmax and -vmax -- the two hardest points in
            the range this instance is meant to cover."""
            _, gp = fn(vmax)
            _, gm = fn(-vmax)
            return float(gp), float(gm)

        order = gl_order0
        fn = _make_I_and_didv(self._static, self._packed, delta, vmax, order)
        prev = probe(fn)
        streak = 0
        converged = False
        while order < gl_order_max:
            order = min(2*order, gl_order_max)
            fn = _make_I_and_didv(self._static, self._packed, delta, vmax, order)
            cur = probe(fn)
            agree = all(abs(c-p)/max(abs(c), abs(p), 1e-12) < tol
                        for c, p in zip(cur, prev))
            streak = streak+1 if agree else 0
            prev = cur
            if streak >= min_consecutive:
                converged = True
                break
        if not converged:
            warnings.warn(
                f"JaxKeldyshCurrent: quasienergy quadrature order did not converge "
                f"dI/dV to tol={tol} by gl_order_max={gl_order_max} at vmax=+-{vmax}, "
                f"nmax={nmax}; result may be inaccurate, try a larger gl_order_max")
        self.gl_order = order
        self._I_and_didv = fn

    def current_and_didv(self, voltage):
        """(I(voltage), dI/dV(voltage)) in one pass. voltage must satisfy
        |voltage|<=self.vmax (the fitted self-energy/quadrature window);
        voltage==0. returns (0.,0.) directly (I(0)=0 exactly; dI/dV(0),
        the zero-bias/linear-response conductance, is generally nonzero
        but this fixed-grid formulation's domain shrinks to empty exactly
        at V=0 and cannot resolve it -- evaluate at a small nonzero
        voltage instead, exactly as keldysh_didv's own central difference
        never evaluates dc_current exactly at V=0 either)."""
        if voltage == 0.:
            return 0.0, 0.0
        if abs(voltage) > self.vmax:
            raise ValueError(f"|voltage|={abs(voltage)} exceeds this instance's "
                              f"fitted window vmax={self.vmax}")
        val, grad = self._I_and_didv(voltage)
        return float(val), float(grad)

    def current(self, voltage):
        return self.current_and_didv(voltage)[0]

    def didv(self, voltage):
        return self.current_and_didv(voltage)[1]


def keldysh_didv_jax(ht, voltage, nmax, delta=None, **kwargs):
    """Convenience one-off call: build a JaxKeldyshCurrent sized just for
    this `voltage` and return its dI/dV. For more than one voltage on the
    same junction/nmax, build a JaxKeldyshCurrent once (sized to cover the
    whole sweep) and call .didv repeatedly instead -- this function pays
    the full build (JIT compile + quadrature-order search) cost every
    single call, which is the expensive part; see the class docstring."""
    jkc = JaxKeldyshCurrent(ht, nmax, abs(voltage), delta=delta, **kwargs)
    return jkc.didv(voltage)
