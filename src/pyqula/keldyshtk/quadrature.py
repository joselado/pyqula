"""Globally adaptive Gauss-Kronrod quadrature that evaluates its integrand
in BATCHES rather than one node at a time.

`scipy.integrate.quad` (QUADPACK) is an excellent adaptive rule but calls
its integrand through a scalar Python callback, once per node. That is
exactly the wrong shape for `keldyshtk.current`'s quasienergy integral,
whose integrand (`current_integrand`) is a numba-compiled chain solve: each
node costs a separate Python->numba dispatch, and none of the node-level
parallelism `current_integrand_batch`/`_rgf_chain_batch_jit` already
implement (one `prange` over an arbitrary set of independent quasienergies)
can be used at all.

`adaptive_quad_batch` keeps QUADPACK's node economy -- the same 21-point
Gauss-Kronrod rule with the same embedded error estimator, so an easy
integrand still converges on its first 21 nodes -- but restructures the
adaptive loop so that every panel awaiting evaluation in a given refinement
round is evaluated in ONE batched call. Instead of "bisect the single worst
panel, re-evaluate it" repeated serially (QUADPACK's QAG loop), it bisects
every panel that needs bisecting to reach the requested tolerance and
evaluates all the resulting sub-panels together.

That changes the node set relative to `scipy.integrate.quad` (the same
integral, a different adaptive path to it), so it is not a bit-identical
drop-in; it is accurate to the same requested tolerance, not to the same
floating-point value. See `keldyshtk/current.py`'s `dc_current` docstring
for the measured accuracy/speed comparison on this repo's own Keldysh
benchmark cases.

Note `scipy.integrate.quad_vec` does NOT solve this problem: it is
vector-valued in the integrand's OUTPUT (integrating an array-valued f),
still evaluating one x at a time -- the axis that needs batching here is
the integration variable itself.
"""
import numpy as np

# Gauss-Kronrod 21-point rule (QUADPACK's DQK21 data): `_XGK` are the
# non-negative abscissae in decreasing order, with the odd-indexed entries
# (0-based: 1,3,5,7,9) being the 10-point Gauss-Legendre abscissae and the
# even-indexed ones the added Kronrod abscissae; `_WGK` are the 21-point
# Kronrod weights and `_WG` the 10-point Gauss weights, both for the
# non-negative half (the rule is symmetric). Verified at import against
# numpy's own `leggauss(10)` and against exact integration of monomials --
# see `_build_rule`, which turns these into the flat symmetric arrays used
# below.
_XGK = (0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.000000000000000000000000000000000)
_WGK = (0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821)
_WG = (0.066671344308688137593568809893332,
       0.149451349150580593145776339657697,
       0.219086362515982043995534934228163,
       0.269266719309996355091226921569469,
       0.295524224714752870173892994651338)


def _build_rule():
    """Expand the half-rule tables above into the full symmetric 21-node
    rule: abscissae `x` (21,), Kronrod weights `wgk` (21,) and Gauss
    weights `wg` (21,, zero at the 11 Kronrod-only nodes) so both sums are
    plain dot products over the same node array."""
    x = np.array([-v for v in _XGK] + list(_XGK[-2::-1]))
    wgk = np.array([v for v in _WGK] + list(_WGK[-2::-1]))
    wg = np.zeros_like(wgk)
    # Gauss nodes sit at the odd indices of the decreasing half-table, i.e.
    # at 1,3,5,7,9 and their mirrors.
    for i, j in enumerate(range(1, 10, 2)):
        wg[j] = _WG[i]
        wg[20-j] = _WG[i]
    return x, wgk, wg


_X, _WGK_FULL, _WG_FULL = _build_rule()


def _validate_rule():
    """Self-check of the hardcoded tables: the Gauss subset must reproduce
    numpy's own `leggauss(10)`, and both rules must integrate the monomials
    they are exact for. Cheap (a few dot products), run once at import, and
    it turns a mistyped constant into an immediate ImportError instead of a
    silently slightly-wrong quadrature."""
    gx, gw = np.polynomial.legendre.leggauss(10)
    mask = _WG_FULL != 0.
    assert np.allclose(np.sort(_X[mask]), np.sort(gx), atol=1e-14)
    assert np.allclose(np.sort(_WG_FULL[mask]), np.sort(gw), atol=1e-14)
    for p in range(0, 20):  # Gauss-10 is exact to degree 19
        exact = 0. if p % 2 else 2./(p+1)
        assert abs(np.dot(_WG_FULL, _X**p) - exact) < 1e-13
    for p in range(0, 32):  # Kronrod-21 is exact to degree 31
        exact = 0. if p % 2 else 2./(p+1)
        assert abs(np.dot(_WGK_FULL, _X**p) - exact) < 1e-13


_validate_rule()


def _panel_nodes(edges):
    """Nodes of the 21-point rule on every panel in `edges` -- an
    (npanels,2) array of [left,right] endpoints -- returned flat,
    (npanels*21,), in panel-major order (so a reshape to (npanels,21)
    recovers the per-panel grouping)."""
    half = 0.5*(edges[:, 1]-edges[:, 0])
    mid = 0.5*(edges[:, 1]+edges[:, 0])
    return (mid[:, None] + half[:, None]*_X[None, :]).ravel()


def _panel_rule(edges, fv):
    """Per-panel integral estimate and QUADPACK error estimate, given
    `fv` (npanels,21), the integrand at `_panel_nodes(edges)`.

    This is DQK21's own estimator, kept verbatim rather than simplified to
    |Kronrod-Gauss|: for a smooth integrand that raw difference is a wildly
    pessimistic error bound, and using it directly would make an easy panel
    look unconverged and subdivide it needlessly -- exactly the node-economy
    loss that makes a fixed grid slower than adaptive quadrature on this
    repo's normal-junction Keldysh case."""
    half = 0.5*(edges[:, 1]-edges[:, 0])
    resk = fv@_WGK_FULL
    resg = fv@_WG_FULL
    resabs = np.abs(fv)@_WGK_FULL
    resasc = np.abs(fv - 0.5*resk[:, None])@_WGK_FULL
    ahalf = np.abs(half)
    result = resk*half
    resabs = resabs*ahalf
    resasc = resasc*ahalf
    err = np.abs((resk-resg)*half)
    # DQK21's rescaling: for a smooth panel (resasc, the rule's own measure
    # of how much f varies about its mean, comparable to |resk-resg|) the
    # 1.5 power drives the estimate far below the raw difference; for a
    # rough one it saturates at resasc.
    # min(1,x)**1.5 rather than min(1,x**1.5): identical for x>=0, but does
    # not overflow when a nearly-smooth panel makes resasc tiny.
    scale = np.where(resasc > 0., 200.*err/np.where(resasc > 0., resasc, 1.), 0.)
    rescaled = resasc*np.minimum(1., scale)**1.5
    err = np.where((resasc > 0.) & (err > 0.), rescaled, err)
    eps = np.finfo(float).eps
    return result, np.maximum(50.*eps*resabs, err)


def adaptive_quad_batch(fbatch, a, b, epsrel=1e-3, epsabs=0., limit=50,
                        min_split=1, full_output=False):
    """Integrate `fbatch` over [a,b] with a globally adaptive composite
    Gauss-Kronrod 21 rule, evaluating one BATCH of nodes per refinement
    round (see the module docstring).

    `fbatch(x)` takes a 1D array of nodes and returns a 1D array of the
    integrand there -- the batched-integrand contract, not a scalar
    callback.

    The loop mirrors QUADPACK's accuracy contract (stop once the summed
    error estimate is within `max(epsabs, epsrel*|I|)`) but not its
    one-panel-at-a-time refinement: each round bisects the smallest set of
    worst panels whose removal brings the *unrefined* remainder safely
    under the target, and evaluates all of their children at once. `limit`
    caps the total panel count exactly as `scipy.integrate.quad`'s does.
    `min_split` forces at least that many panels to be bisected per round
    even when fewer would do -- purely a batch-size knob (more nodes per
    round, fewer rounds), left at 1 (pure economy) by default.

    Returns the integral, or `(integral, abserr, info)` with
    `info = {"nevals","nrounds","npanels","converged"}` if `full_output`.
    No warning is raised on non-convergence -- the caller owns that policy
    (see `keldyshtk.current.dc_current`)."""
    if a == b:
        return (0., 0., {"nevals": 0, "nrounds": 0, "npanels": 0,
                         "converged": True}) if full_output else 0.
    edges = np.array([[a, b]], dtype=float)
    fv = fbatch(_panel_nodes(edges)).reshape(1, len(_X))
    res, err = _panel_rule(edges, fv)
    nevals = fv.size
    nrounds = 1
    converged = False
    while True:
        total = res.sum()
        toterr = err.sum()
        target = max(epsabs, epsrel*abs(total))
        if toterr <= target or not np.isfinite(toterr):
            converged = bool(toterr <= target)
            break
        if len(edges) >= limit:
            break
        # Panels to bisect this round: worst first, taking the shortest
        # prefix that leaves the untouched remainder comfortably (half the
        # target) under tolerance -- so a single dominant singular panel
        # costs one bisection, while a uniformly-hard integrand bisects
        # broadly, both in one batched round either way.
        order = np.argsort(err)[::-1]
        tail = np.cumsum(err[order][::-1])[::-1]  # tail[k] = sum(err[order][k:])
        keep = np.nonzero(tail <= 0.5*target)[0]
        nsel = int(keep[0]) if len(keep) else len(edges)
        nsel = max(nsel, min_split, 1)
        nsel = min(nsel, len(edges), limit-len(edges))
        sel = order[:nsel]
        left = edges[sel, 0]
        right = edges[sel, 1]
        mid = 0.5*(left+right)
        new = np.empty((2*nsel, 2))
        new[0::2, 0] = left; new[0::2, 1] = mid
        new[1::2, 0] = mid; new[1::2, 1] = right
        fvn = fbatch(_panel_nodes(new)).reshape(2*nsel, len(_X))
        nevals += fvn.size
        nrounds += 1
        resn, errn = _panel_rule(new, fvn)
        mask = np.ones(len(edges), dtype=bool)
        mask[sel] = False
        edges = np.concatenate([edges[mask], new])
        res = np.concatenate([res[mask], resn])
        err = np.concatenate([err[mask], errn])
    out = float(res.sum())
    if full_output:
        return out, float(err.sum()), {"nevals": nevals, "nrounds": nrounds,
                                       "npanels": len(edges),
                                       "converged": converged}
    return out
