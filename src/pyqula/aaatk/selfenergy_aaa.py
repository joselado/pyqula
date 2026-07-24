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


def default_ncand(erange, delta, scale=24, minimum=64):
    """Starting candidate-grid size for a window of width `erange` and
    features of width `delta`: scales with log2(erange/delta) (the number
    of independent resonance-width scales spanning the window), not with
    erange/delta itself -- AAA support points, unlike a quantics grid,
    don't need to *bisect* a resonance's width, just sample across it a
    handful of times, so this is deliberately far more modest than
    qtcitk.selfenergy_qtci.bits_from_delta's exponentiated (2**bits) grid
    size. This is only the *starting* size; SelfenergyAAA doubles it
    (reusing every already-solved point) until a held-out validation check
    passes, so an unusually hard target still converges, just after
    revisiting this estimate rather than needing it to be exact upfront."""
    if delta <= 0: raise ValueError("delta must be positive")
    if erange <= 0: raise ValueError("erange must be positive")
    ratio = max(erange/delta, 2.0)
    return max(minimum, int(scale*np.ceil(np.log2(ratio))))


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
    fits every entry, then checks a held-out validation grid (points
    offset from the candidates, so genuinely off-sample) against
    `tolerance` (relative, entrywise). If any entry's fit saturated its
    support-point budget (used every one of `mmax`, the signature of "not
    enough poles allowed", not "not enough samples"), `mmax` is escalated
    first; otherwise `ncand` is doubled. Every previously-solved point
    stays cached across rounds, so refinement never re-pays for work
    already done. Gives up (uses the best fit found, without raising)
    after `maxrounds` rounds or once both budgets hit their caps -- like
    keldyshtk.current.dc_current's own adaptive sideband loop, this
    guarantees termination rather than looping indefinitely on a target
    that resists this ansatz; check `.converged` if that matters to a
    caller."""

    def __init__(self, get_selfenergy, dim, emin, emax, delta,
                 tolerance=1e-6, ncand0=None, ncand_max=2500,
                 nvalidate=32, mmax0=100, mmax_max=400, aaa_tolerance=None,
                 maxrounds=8, **kwargs):
        if emax <= emin: raise ValueError("emax must be > emin")
        self.dim = dim
        self.emin, self.emax = emin, emax
        if ncand0 is None: ncand0 = default_ncand(emax-emin, delta)
        if aaa_tolerance is None: aaa_tolerance = 0.1*tolerance
        solved = {}
        def full_matrix(e):
            key = round(e, 12)
            if key not in solved:
                solved[key] = algebra.todense(get_selfenergy(e))
            return solved[key]
        self._solved = solved  # exposed for diagnostics/benchmarking

        rng = np.random.default_rng(0)  # deterministic validation offsets
        # A dyadically-refinable grid (each round inserts the midpoint of
        # every adjacent pair) rather than a fresh np.linspace(...,ncand)
        # each round: previous rounds' points are then an exact subset of
        # the new grid, so every doubling's true solves are 100% additive
        # -- nothing already in `solved` is ever re-requested at a
        # different (unlucky, non-matching) grid coordinate and silently
        # recomputed.
        Z = np.linspace(emin, emax, ncand0)
        mmax = min(mmax0, mmax_max)
        converged = False
        for _round in range(maxrounds):
            Fmats = np.array([full_matrix(e) for e in Z])
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
            step = (emax-emin)/(ncand-1)
            nval = min(nvalidate, ncand-1)
            offsets = rng.uniform(0.1*step, 0.9*step, nval)
            base = rng.choice(ncand-1, nval, replace=False)
            Zval = Z[base] + offsets
            maxerr, denom = 0., 0.
            for e in Zval:
                true = full_matrix(e)
                denom = max(denom, np.max(np.abs(true)))
            denom = max(denom, 1e-12)
            for e in Zval:
                true = full_matrix(e)
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
                Z = np.sort(np.concatenate([Z, 0.5*(Z[:-1]+Z[1:])]))
            else:
                break

        self.entries = entries
        self.ncand = len(Z)
        self.mmax = mmax
        self.validation_error = maxerr
        self.converged = converged
        self._pack_entries()

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
        return _eval_matrix_jit(complex(e), self._zj_pad, self._wf_pad,
                                 self._w_pad, self._ii, self._jj, self.dim)

    def nsolved(self):
        """Number of true (uncompressed) self-energy solves used to build
        every entry's interpolant -- the actual cost paid, versus however
        many energies are evaluated afterward via __call__."""
        return len(self._solved)
