# Quantics tensor-cross-interpolation (qtci) cache for a lead's retarded
# self-energy as a function of energy, replacing many independent
# Sancho-Rubio/bloch_selfenergy solves with one compressed interpolant built
# from far fewer "training" evaluations, then evaluated cheaply everywhere
# else via tensor-train contraction (see qutecipytk.tensortrain.base.
# AbstractTensorTrain.evaluate: a chain of small matrix products, O(bits),
# not a fresh iterative solve).
#
# Motivated by keldyshtk.current.dc_current's Floquet-Keldysh sideband
# sweep: a single LocalProbe Keldysh dI/dV evaluation was measured to solve
# ~28500 distinct (lead,energy) self-energies from scratch, with
# essentially no reuse even between the two nearby voltages needed for the
# finite-difference derivative (each dc_current call gets its own fresh
# cache dict) -- self-energy computation alone was ~78% of total wall time
# in that profile. Building one interpolant per lead up front and sharing
# it across the whole sideband sweep (and across both finite-difference
# evaluations) targets that redundancy directly.
#
# Self-energy is a small dim x dim matrix, not a scalar, so this follows
# qtcitk/densitymatrix_qtci.py's precedent: one independent 1D qtci per
# matrix entry, all sharing one cache of the true (expensive) self-energy
# solves so a given energy is only ever solved once regardless of how many
# entries' pivot searches revisit it.
#
# MEASURED RESULT: for a LocalProbe's Sancho-Rubio lead self-energy over
# the energy range multiple Andreev reflection needs, this does NOT
# currently win, and it isn't wired in by default (keldyshtk.current.
# dc_current/build_selfenergy_qtci, transporttk.didv.keldysh_didv's
# use_qtci flag are all opt-in). Benchmarked on examples/transport/
# decay_constant_keldysh's parameters (SC probe + SC sample): building the
# two per-lead interpolants needed ~42700 true solves combined, *more*
# than the ~28500 the direct per-energy approach needs, so it was slower
# end to end (24.9s vs 23.9s) despite matching the direct result to 1.5e-5
# relative accuracy. Narrowing the covered energy window helps some (a
# probe-lead interpolant over erange=4.6 needed 21832 solves -- 8.3% of
# its grid -- versus 5251 over erange=0.2, i.e. 64.1% of its now-smaller
# grid) but the *fraction* of the grid needing a true solve goes up as the
# window shrinks, the signature of a function that genuinely isn't very
# compressible over the range that matters here (real structure --
# superconducting gap edges, van Hove features -- spread across it, not
# concentrated at a few points), rather than a "wrong window size"
# problem a smarter windowing scheme would fix. A per-quasienergy-point
# windowing redesign was considered and not pursued: the reduction it
# would plausibly buy (roughly halving to quartering the solve count, per
# the erange scan above) looked unlikely to outweigh the bookkeeping
# overhead of managing many narrower, potentially-overlapping
# interpolants across the quadrature. Left as documented, tested,
# off-by-default infrastructure -- e.g. for a self-energy from a
# different, smoother lead where compression may fare better.
import numpy as np

from .. import algebra


def bits_from_delta(erange, delta, margin=4):
    """Number of quantics bits needed to resolve features of width `delta`
    (e.g. the Lorentzian broadening in a retarded self-energy/Green's
    function) over an energy window of width `erange`. Each extra bit
    halves the quantics grid spacing, so matching a target resolution
    delta over a window erange only takes bits growing logarithmically
    with erange/delta, not linearly -- the same principle as
    gkintegrate.gkorder_from_nk's nk-to-Gauss-Kronrod-order mapping.
    `margin` extra bits are added on top of the bare log2 estimate since
    crossinterpolate itself still needs some resolution *within* one
    delta-wide feature to represent its shape, not just to distinguish it
    from its neighbors."""
    if delta <= 0: raise ValueError("delta must be positive")
    if erange <= 0: raise ValueError("erange must be positive")
    ratio = max(erange/delta, 2.0)
    return max(1, int(np.ceil(np.log2(ratio))) + margin)


class SelfenergyQTCI:
    """Interpolated cache of a lead's retarded self-energy, matrix(energy),
    over a fixed window [emin,emax], built once via qutecipy's tensor
    cross interpolation and then evaluated at any energy in the window
    through cheap tensor-train contractions instead of a fresh solve.

    `get_selfenergy(e)` must return the (dim,dim) self-energy matrix at
    energy `e` (typically `lambda e: ht.get_selfenergy(e,lead=...,
    delta=delta,pristine=True,numba=True)`); `delta` is the broadening
    used there, which sets the number of quantics bits via
    bits_from_delta. Call the resulting object like a function,
    `sqtci(e)`, to get the interpolated matrix at energy e."""

    def __init__(self, get_selfenergy, dim, emin, emax, delta,
                 margin=4, tolerance=1e-6, **kwargs):
        from ..qutecipytk import crossinterpolate2
        from ..qutecipytk.tensortrain.core import tensortrain
        from ..qutecipytk.tensortrain.cachedfunction import CachedFunction
        from ..qutecipytk.quantics.discretized import DiscretizedGrid
        if emax<=emin: raise ValueError("emax must be > emin")
        bits = bits_from_delta(emax-emin, delta, margin=margin)
        qgrid = DiscretizedGrid.from_resolutions(["e"], [bits],
                lower_bound=[emin], upper_bound=[emax],
                includeendpoint=True)
        localdims = qgrid.localdimensions()
        self.qgrid = qgrid
        self.dim = dim
        self.bits = bits
        # Shared across every (i,j) entry's independent pivot search: a
        # given quantics energy is only ever solved once overall, no
        # matter how many matrix entries' interpolations end up visiting
        # it (mirrors densitymatrix_qtci.py's shared k-point cache).
        solved = {}
        def full_matrix(e):
            key = round(e, 12)
            if key not in solved:
                solved[key] = algebra.todense(get_selfenergy(e))
            return solved[key]
        self._solved = solved # exposed for diagnostics/benchmarking
        # Candidate initial pivots to seed each entry's cross interpolation
        # with: TensorCI2 refuses to start from a default (all-zero-index)
        # pivot that evaluates to exactly zero ("maxsamplevalue is zero!"),
        # which happens routinely here for off-diagonal self-energy
        # entries related by a symmetry that makes them vanish at that
        # particular grid point (same issue gkintegrate.integrate_robust
        # handles for Berry curvature/BZ integrals). Try a handful of grid
        # points spread across the window instead of just the edge.
        ngrid = 2**bits
        candidate_grididx = sorted({0, ngrid//4, ngrid//2, (3*ngrid)//4,
                                     ngrid-1})
        candidate_quantics = [qgrid.grididx_to_quantics([g])
                               for g in candidate_grididx]
        self.entries = {}
        for i in range(dim):
            for j in range(dim):
                def qf(quantics, i=i, j=j): # default args freeze the loop vars
                    e = qgrid.quantics_to_origcoord(quantics)[0]
                    return full_matrix(e)[i, j]
                qf = CachedFunction(np.complex128, qf, localdims)
                pivot = None
                for cand in candidate_quantics:
                    if qf(cand) != 0:
                        pivot = cand
                        break
                if pivot is None:
                    # identically zero at every candidate point (almost
                    # certainly a symmetry-protected zero, e.g. an
                    # electron-hole cross term absent for this lead) --
                    # skip qtci for this entry, __call__ leaves it at 0
                    self.entries[(i, j)] = None
                    continue
                tci, ranks, errors = crossinterpolate2(np.complex128, qf,
                        localdims, initialpivots=[pivot],
                        tolerance=tolerance, **kwargs)
                self.entries[(i, j)] = tensortrain(tci)

    def __call__(self, e):
        """Return the interpolated self-energy matrix at energy e."""
        grididx = self.qgrid.origcoord_to_grididx([e])
        quantics = self.qgrid.grididx_to_quantics(grididx)
        out = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for (i, j), tt in self.entries.items():
            if tt is not None: out[i, j] = tt.evaluate(quantics)
        return out

    def nsolved(self):
        """Number of true (uncompressed) self-energy solves used to build
        every entry's interpolant -- the actual cost paid, versus however
        many energies are evaluated afterward via __call__."""
        return len(self._solved)
