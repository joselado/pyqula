# AAA-based cache of a lead's retarded self-energy, matrix(energy), built
# from far fewer true solves than qtcitk/selfenergy_qtci.py's quantics
# approach needs for the same target (see that module's docstring for the
# quantics benchmark this followed up on, and keldyshtk/current.py for how
# the two compare on the actual Keldysh dI/dV workload). Where quantics
# has to bisect a resonance's width bit-by-bit, AAA represents it directly
# as a pole -- the correct ansatz for a retarded self-energy's actual
# analytic structure -- so a handful of support points per feature
# suffices instead of thousands of dyadic grid points.
#
# That solve-count reduction (hundreds of true solves per lead versus the
# tens of thousands keldyshtk.current.dc_current's Floquet sideband sweep
# needs directly) does NOT translate one-for-one into wall-clock speedup,
# though, and measuring the difference mattered: for a cheap-to-solve
# target (a small-dim 1D Sancho-Rubio self-energy, evaluated through a
# compiled numba kernel), the per-call cost of *evaluating* this
# interpolant -- __call__ loops over every (i,j) entry, called once per
# (lead,energy) pair dc_current's sideband sweep visits, tens of
# thousands of times per call -- was initially the same order of
# magnitude as the numba solve it replaces, largely cancelling the win.
# Three real bugs/inefficiencies were found and fixed while measuring
# this: (1) the SVD inside aaa.aaa() defaulted to full_matrices=True,
# computing an unused (M,M) left singular-vector matrix instead of the
# (M,m) economy one -- O(M^3) instead of O(M*m^2), which dominated
# everything else once the candidate grid `M` reached a few thousand;
# (2) _BarycentricRational.__call__'s scalar path rebuilt small numpy
# arrays on every evaluation instead of using plain Python complex
# arithmetic, ~3.5x slower per call for the modest (tens of points)
# support-point counts this module actually produces; (3) that plain-
# Python scalar arithmetic was itself replaced with a numba-jitted core
# (aaa._eval_scalar_jit), another ~7.5x on top of fix (2) (0.5us vs
# 3.9us/call for ~30 support points, plus a one-time ~0.3s compile cost
# cached to disk via cache=True); (4) SelfenergyAAA.__call__ itself packed
# every (i,j) entry into one padded array and evaluates the whole matrix
# with a single compiled call (aaatk.selfenergy_aaa._eval_matrix_jit)
# instead of one Python dict-iteration + numba-dispatch pair per nonzero
# entry, another ~3.4x on the evaluation call (5.28us -> 1.54us for 8
# active entries). After all four fixes, measured net effect through the
# real dc_current pipeline (not an idealized batched-evaluation
# microbenchmark) is a consistent win for cheap-per-solve 1D targets
# (roughly break-even to ~40% faster, depending on the sideband window),
# and should be substantially larger for expensive-per-solve targets
# (e.g. a 2D sample's green_kchain/adaptive-quadrature self-energy, where
# a single direct solve alone costs far more than this interpolant's
# entire per-energy evaluation) -- there the reduced *solve* count, not
# evaluation overhead, dominates. What's left of the gap to the direct
# method beyond self-energy cost is shared _floquet_green_functions/
# current_integrand machinery (RGF-chain array building, the sideband
# trace sum) -- see keldyshtk/current.py's _rgf_chain_jit and
# _integrand_trace_sum_jit docstrings for that separate (non-AAA-specific)
# optimization round, which sped up the direct path by a similar amount.
#
# Two independent knobs control convergence, and confusing them causes a
# real, measured pathology: `ncand` (candidate/sample density) fixes
# under-*resolution* (the grid missing a feature entirely), while `mmax`
# (the per-entry cap on AAA support points, i.e. poles) fixes under-
# *capacity* (not enough poles allowed to represent every feature that
# WAS found). A wide window with many sidebands (e.g. a large nmax_max in
# keldyshtk/current.py's Floquet sweep, which packs many Andreev/MAR
# resonances into one window) can genuinely need more support points than
# a narrow one, with no amount of extra candidate density fixing it --
# doubling `ncand` alone in that regime just re-pays an ever-larger SVD
# cost every round for a fit that was never going to converge, which is
# exactly the runaway this module's __init__ now avoids by escalating
# `mmax` first whenever a fit saturates its cap. This was a real, measured
# multi-minute-plus hang (from bug (1) above compounding with unbounded
# ncand doubling) before both the SVD fix and this escalation split.
import warnings

import numpy as np
from numba import jit

from .. import algebra
from .aaa import aaa


@jit(nopython=True, cache=True)
def _eval_matrix_jit(e, zj_pad, wf_pad, w_pad, ii, jj, dim):
    """Evaluate every active matrix entry's barycentric rational fit at
    energy `e` in one compiled call (see SelfenergyAAA._pack_entries):
    `zj_pad`/`wf_pad`/`w_pad` are (nentries, maxlen), zero-padded past
    each entry's own support-point count -- a padding term has w=0 (and
    hence wf=0), contributing nothing to either the numerator or
    denominator, so every entry can share one loop bound (`maxlen`)."""
    out = np.zeros((dim, dim), dtype=np.complex128)
    nentries = ii.shape[0]
    maxlen = zj_pad.shape[1]
    for idx in range(nentries):
        num = 0j
        den = 0j
        for k in range(maxlen):
            wk = w_pad[idx, k]
            if wk == 0: continue
            zjk = zj_pad[idx, k]
            if e == zjk:
                num = wf_pad[idx, k]
                den = wk
                break
            c = 1.0 / (e - zjk)
            num += wf_pad[idx, k] * c
            den += wk * c
        out[ii[idx], jj[idx]] = num / den
    return out


@jit(nopython=True, cache=True)
def _eval_matrix_batch_jit(es, zj_pad, wf_pad, w_pad, ii, jj, dim):
    """Batched counterpart of _eval_matrix_jit: evaluate every active
    matrix entry's barycentric rational fit at every energy in `es` in one
    compiled call instead of one Python-level (and hence one numba-
    dispatch) call per energy. Motivated by profiling keldyshtk.current.
    dc_current on a deep-subgap junction (large nmax): once self-energies
    are this cheap to evaluate, the surrounding Python-level per-site
    dispatch of the Floquet sideband sweep (a dict-cache lookup + round()
    key + one numba call per (lead,sideband) site) -- not the barycentric
    arithmetic itself -- became the dominant cost (~58% of total wall time
    in one measured case, versus a few percent for the actual evaluations).
    keldyshtk.current._batch_selfenergy calls this once per lead per
    quasienergy point (covering every sideband energy that call needs) in
    place of that per-site loop."""
    n = es.shape[0]
    out = np.zeros((n, dim, dim), dtype=np.complex128)
    nentries = ii.shape[0]
    maxlen = zj_pad.shape[1]
    for m in range(n):
        e = es[m]
        for idx in range(nentries):
            num = 0j
            den = 0j
            for k in range(maxlen):
                wk = w_pad[idx, k]
                if wk == 0: continue
                zjk = zj_pad[idx, k]
                if e == zjk:
                    num = wf_pad[idx, k]
                    den = wk
                    break
                c = 1.0 / (e - zjk)
                num += wf_pad[idx, k] * c
                den += wk * c
            out[m, ii[idx], jj[idx]] = num / den
    return out


def default_ncand(erange, delta, scale=24, minimum=64):
    """Starting candidate-grid size for a window of width `erange` and
    features of width `delta`: scales with log2(erange/delta) (the number
    of independent resonance-width scales spanning the window), not with
    erange/delta itself -- AAA support points, unlike a quantics grid,
    don't need to *bisect* a resonance's width, just sample across it a
    handful of times, so this is deliberately far more modest than
    qtcitk.selfenergy_qtci.bits_from_delta's exponentiated (2**bits) grid
    size. This is only the *starting* size; SelfenergyAAA refines it
    adaptively (reusing every already-solved point, see `_refine_grid`)
    until a held-out validation check passes, so an unusually hard target
    still converges, just after revisiting this estimate rather than
    needing it to be exact upfront."""
    if delta <= 0: raise ValueError("delta must be positive")
    if erange <= 0: raise ValueError("erange must be positive")
    ratio = max(erange/delta, 2.0)
    return max(minimum, int(scale*np.ceil(np.log2(ratio))))


def _refine_grid(Z, Fmats, cap, growth=0.5):
    """Adaptive (curvature-driven) grid refinement: bisect the *roughest*
    intervals first (largest jump in the already-sampled self-energy
    matrix between adjacent candidates), up to `growth` fraction of the
    current grid size or the remaining budget to `cap`, whichever is
    smaller -- instead of bisecting every interval uniformly regardless of
    how well- or under-resolved it already is.

    This matters because `cap` (`ncand_max`) is a hard, fixed budget:
    uniform doubling spends that budget everywhere, including on regions
    already resolved, so it reaches `cap` after only a couple of rounds
    (`ncand` roughly doubles each round) with no way to keep refining a
    single narrow feature further. Measured directly on the case that
    motivated this (a superconducting lead's gap-edge coherence-peak
    singularity, physical width set by the broadening `delta`, e.g.
    `delta=1e-4`, sitting inside a candidate grid with a much coarser
    starting spacing): uniform doubling hit `ncand_max=2500` within ~3
    rounds, shrinking the local spacing there by only ~6x -- nowhere near
    enough to resolve a feature that started ~170x under-resolved -- and
    then permanently stopped growing, *regardless of how many further
    rounds were allowed*, because the escalation branch requires
    `ncand < ncand_max`. Repeatedly picking the roughest intervals instead
    keeps spending each round's budget where the fit is actually worst
    (the two new sub-intervals straddling a real singularity are
    themselves the roughest next round, so the zoom continues on its own),
    which is the only way to reach the needed local resolution within a
    bounded total-point budget. See
    documentation/keldysh_aaa_selfenergy_accuracy_plan.md for the
    measurement this is based on."""
    diffs = np.max(np.abs(Fmats[1:] - Fmats[:-1]), axis=(1, 2))
    n = diffs.shape[0]
    remaining = max(0, cap - len(Z))
    if remaining == 0:
        return Z
    k = min(max(1, int(np.ceil(growth*n))), n, remaining)
    worst = np.argsort(diffs)[::-1][:k]
    new_pts = 0.5*(Z[worst] + Z[worst+1])
    return np.sort(np.concatenate([Z, new_pts]))


class SelfenergyAAA:
    """Interpolated cache of a lead's retarded self-energy, matrix(energy),
    over a fixed window [emin,emax], built by fitting an independent AAA
    barycentric rational interpolant to each matrix entry -- all entries
    share the same underlying `get_selfenergy` samples (one true solve
    returns the whole matrix), so fitting all dim*dim entries costs no
    extra true solves beyond building the shared candidate grid.

    `get_selfenergy(e)` must return the (dim,dim) self-energy matrix at
    energy `e` (typically `lambda e: ht.get_selfenergy(e,lead=...,
    delta=delta,pristine=True,numba=True)`); `delta` is the broadening
    used there. Call the resulting object like a function, `sqtci(e)`, to
    get the interpolated matrix at energy e.

    The candidate grid starts at `default_ncand(emax-emin,delta)` points
    and the per-entry support-point budget starts at `mmax0`; each round
    fits every entry, then checks a held-out validation grid -- `nvalidate`
    points drawn uniformly at random across the *whole* `[emin,emax]`
    window (independent of where the candidates sit, so genuinely
    off-sample and not confined to the immediate neighborhood of an
    existing candidate) -- against `tolerance` (relative to the largest
    sampled |Sigma|). If any entry's fit saturated its support-point
    budget (used every one of `mmax`, the signature of "not enough poles
    allowed", not "not enough samples"), `mmax` is escalated first;
    otherwise the candidate grid is refined via `_refine_grid`, which
    concentrates new points on the currently-roughest intervals rather
    than bisecting everywhere uniformly (see that function's docstring for
    why uniform doubling cannot resolve a narrow feature within a bounded
    `ncand_max`, regardless of how many rounds are allowed). Every
    previously-solved point stays cached across rounds, so refinement
    never re-pays for work already done. Gives up (uses the best fit
    found, without raising) after `maxrounds` rounds or once both budgets
    hit their caps -- like keldyshtk.current.dc_current's own adaptive
    sideband loop, this guarantees termination rather than looping
    indefinitely on a target that resists this ansatz; check `.converged`
    if that matters to a caller. A caller relying on speed rather than
    guaranteed accuracy should treat `converged=False` as "fall back to a
    direct solve", not "use the best-effort fit anyway". The default
    `tolerance=1e-3` is tuned to match dc_current's own current-convergence
    target -- passing a much tighter `tolerance` (e.g. the old 1e-6
    default) on a genuine near-singularity target can legitimately take
    several minutes to build (measured: ~4-9 minutes for one lead on a
    deep-subgap superconducting case, `ncand` climbing toward `ncand_max`)
    rather than hang -- `maxrounds`/`ncand_max` bound the effort, not the
    wall-clock time of a single round once `ncand` has grown large."""

    def __init__(self, get_selfenergy, dim, emin, emax, delta,
                 tolerance=1e-3, ncand0=None, ncand_max=20000,
                 nvalidate=32, mmax0=100, mmax_max=400, aaa_tolerance=None,
                 maxrounds=20, refine_growth=0.5, get_selfenergy_batch=None,
                 **kwargs):
        if emax <= emin: raise ValueError("emax must be > emin")
        self.dim = dim
        self.emin, self.emax = emin, emax
        if ncand0 is None: ncand0 = default_ncand(emax-emin, delta)
        if aaa_tolerance is None: aaa_tolerance = 0.1*tolerance
        solved = {}
        def full_matrix_many(es):
            """Solve every energy in `es` not already in `solved`, in ONE
            batched call when `get_selfenergy_batch` is available (the
            numba prange-parallel Sancho-Rubio iteration, transporttk.
            selfenergy.get_selfenergy_batch / greentk.rg.
            green_renormalization_jit_batch -- see keldyshtk/current.py's
            _batch_selfenergy for the same batching pattern used elsewhere
            on this codebase's Keldysh path), falling back to a per-energy
            Python loop over `get_selfenergy` otherwise (e.g. a LocalProbe,
            which has no batched solve). Every solved energy is cached
            into `solved` either way, so later rounds/validation calls
            reuse it exactly as before -- this is a pure batching of the
            SAME true solves the unbatched path made, not a change to what
            gets solved or how many rounds/candidates are needed, so it
            cannot change the resulting fit."""
            # One vectorized np.round over the whole array instead of a Python
            # `round(e,12)` per energy, three times per call: `es` here is the
            # candidate grid, which grows to ncand_max (20000) over up to
            # `maxrounds` rounds, so the per-item version was a measurable
            # share of build time. Same pattern (and same reason) as
            # keldyshtk/current.py:_batch_selfenergy. Keys are bit-identical
            # to the old per-item round(), so cache hits are unchanged.
            keys = np.round(es, 12).tolist()
            missing = [(k, e) for k, e in zip(keys, es) if k not in solved]
            if missing:
                if get_selfenergy_batch is not None:
                    mats = get_selfenergy_batch(np.asarray([e for _, e in missing]))
                    for (k, _), m in zip(missing, mats):
                        solved[k] = algebra.todense(m)
                else:
                    for k, e in missing:
                        solved[k] = algebra.todense(get_selfenergy(e))
            return np.array([solved[k] for k in keys])
        self._solved = solved  # exposed for diagnostics/benchmarking

        rng = np.random.default_rng(0)  # deterministic validation draws
        Z = np.linspace(emin, emax, ncand0)
        mmax = min(mmax0, mmax_max)
        converged = False
        for _round in range(maxrounds):
            Fmats = full_matrix_many(Z)
            entries = {}
            saturated = False
            for i in range(dim):
                for j in range(dim):
                    Fij = Fmats[:, i, j]
                    if np.max(np.abs(Fij)) == 0.:
                        entries[(i, j)] = None
                        continue
                    r, zj, *_ = aaa(Fij, Z, tol=aaa_tolerance, mmax=mmax)
                    if len(zj) >= mmax: saturated = True
                    entries[(i, j)] = r

            ncand = len(Z)
            # Validation points come from two sources: (a) a broad,
            # domain-uniform random sample (independent of the candidate
            # grid, so not confined to a candidate's immediate
            # neighborhood), for generic/bulk coverage; (b) points drawn
            # from *inside* the currently roughest intervals (the same
            # curvature signal _refine_grid uses), at fractions of the
            # interval distinct from any future bisection midpoint, so a
            # narrow feature the fit hasn't resolved yet is actually
            # tested, not just a large empty region between features. Pure
            # domain-uniform sampling alone was tried and measured to
            # under-detect exactly this: a handful of random points over a
            # wide window has low power to land near a feature much
            # narrower than the window, or even near a moderately
            # under-resolved but still fairly localized region -- see
            # documentation/keldysh_aaa_selfenergy_accuracy_plan.md.
            diffs = np.max(np.abs(Fmats[1:] - Fmats[:-1]), axis=(1, 2))
            n_feat = min(16, diffs.shape[0])
            worst = np.argsort(diffs)[::-1][:n_feat]
            feat_val = np.concatenate([Z[worst] + f*(Z[worst+1]-Z[worst])
                                        for f in (0.3, 0.7)])
            bulk_val = rng.uniform(emin, emax, nvalidate)
            Zval = np.concatenate([feat_val, bulk_val])
            Trues = full_matrix_many(Zval)
            denom = max(np.max(np.abs(Trues)), 1e-12)
            maxerr = 0.
            for e, true in zip(Zval, Trues):
                approx = np.zeros((dim, dim), dtype=np.complex128)
                for (i, j), r in entries.items():
                    if r is not None: approx[i, j] = r(np.complex128(e))
                maxerr = max(maxerr, np.max(np.abs(approx-true))/denom)

            if maxerr <= tolerance:
                converged = True
                break
            if saturated and mmax < mmax_max:
                mmax = min(mmax_max, 2*mmax)
            elif ncand < ncand_max:
                Z = _refine_grid(Z, Fmats, ncand_max, growth=refine_growth)
            else:
                break

        self.entries = entries
        self.ncand = len(Z)
        self.mmax = mmax
        self.validation_error = maxerr
        self.converged = converged
        self._domain_warned = False  # __call__/call_batch warn at most once
        self._pack_entries()

    def _check_domain(self, emin_e, emax_e):
        """Warn (once per instance) if [emin_e,emax_e] pokes outside the
        fitted window [self.emin,self.emax]. This interpolant performs no
        domain enforcement -- __call__/call_batch will happily return a
        barycentric-formula value for an out-of-window energy, which is
        extrapolation, not interpolation, and carries no accuracy guarantee
        at all (unlike the in-window error `validation_error` actually
        bounds). A caller hitting this should widen the window the
        interpolant was built with (e.g. build_selfenergy_aaa's `erange`,
        build_shared_selfenergy's `vmax`/`dv`), not treat the returned
        value as trustworthy."""
        if emin_e >= self.emin and emax_e <= self.emax: return
        if self._domain_warned: return
        self._domain_warned = True
        warnings.warn(
            "SelfenergyAAA: evaluated outside its fitted window "
            f"[{self.emin:.6g},{self.emax:.6g}] (energy reached "
            f"[{emin_e:.6g},{emax_e:.6g}]); the interpolant performs no "
            "domain check and is silently extrapolating there -- treat the "
            "result as untrustworthy. This warning is shown once per "
            "interpolant.", stacklevel=3)

    def _pack_entries(self):
        """Precompute a padded (nentries, maxlen) array layout of every
        active entry's support points/weights, so __call__ below can
        evaluate the *whole* matrix with a single compiled call instead of
        one Python-level dict iteration plus one numba dispatch per
        nonzero (i,j) entry -- each cross into/out of a jitted function
        has its own small fixed cost, which for dim*dim small evaluations
        (a handful of support points each) is a bigger fraction of the
        total than the arithmetic itself. Padding entries are w=0 (hence
        wf=0 too, since wf=w*fj), which contribute exactly 0 to both the
        numerator and denominator of the barycentric formula, so shorter
        entries can share the same (nentries, maxlen) array as the
        longest one with no special-casing inside the jitted loop."""
        active = [(i, j, r) for (i, j), r in self.entries.items() if r is not None]
        self._ii = np.array([i for i, j, r in active], dtype=np.int64)
        self._jj = np.array([j for i, j, r in active], dtype=np.int64)
        n = len(active)
        maxlen = max((len(r.zj) for i, j, r in active), default=0)
        zj_pad = np.zeros((n, maxlen), dtype=np.complex128)
        wf_pad = np.zeros((n, maxlen), dtype=np.complex128)
        w_pad = np.zeros((n, maxlen), dtype=np.complex128)
        for idx, (i, j, r) in enumerate(active):
            m = len(r.zj)
            zj_pad[idx, :m] = r.zj
            wf_pad[idx, :m] = r.wf
            w_pad[idx, :m] = r.w
        self._zj_pad, self._wf_pad, self._w_pad = zj_pad, wf_pad, w_pad

    def __call__(self, e):
        """Return the interpolated self-energy matrix at energy e."""
        er = e.real if isinstance(e, complex) else e
        self._check_domain(er, er)
        return _eval_matrix_jit(complex(e), self._zj_pad, self._wf_pad,
                                 self._w_pad, self._ii, self._jj, self.dim)

    def call_batch(self, es):
        """Return the interpolated self-energy matrix at every energy in
        `es` (1D array-like), as one (len(es),dim,dim) array, in a single
        compiled call -- see _eval_matrix_batch_jit for why this beats
        calling __call__ once per energy in a Python loop."""
        es = np.asarray(es, dtype=np.complex128)
        if es.size:
            er = es.real
            self._check_domain(er.min(), er.max())
        return _eval_matrix_batch_jit(es, self._zj_pad, self._wf_pad,
                                       self._w_pad, self._ii, self._jj, self.dim)

    def nsolved(self):
        """Number of true (uncompressed) self-energy solves used to build
        every entry's interpolant -- the actual cost paid, versus however
        many energies are evaluated afterward via __call__."""
        return len(self._solved)
