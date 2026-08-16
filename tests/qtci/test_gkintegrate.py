"""Robustness contract of qtcitk/gkintegrate.py, the primitive shared by
topology.chern(integration="qtci") and scftk's get_dm_qtci backend.

Both entry points are already validated end to end against external oracles
(analytic Chern on Haldane models; the full dense density matrix), so what is
tested here is the part those end-to-end tests cannot see: the nk-to-order
mapping, whether the requested tolerance is actually honoured, and the
zero-integrand pivot search -- the branch that decides where to seed the
tensor cross interpolation, and when to declare the integral zero without
building one at all.

The oracles are analytic integrals of elementary functions over [0,1]^2, not
values recorded from pyqula's own earlier output.
"""
import numpy as np
import pytest

from pyqula.qtcitk.gkintegrate import (gkorder_from_nk, integrate_robust,
        _pivot_candidates)
from pyqula.qutecipytk.gausskronrod import kronrod

TOL = 1e-8


def gk_nodes(GKorder):
    """The 1d Gauss-Kronrod nodes integrate_robust samples, mapped to [0,1]."""
    nodes1d,_,_ = kronrod(GKorder//2,-1,1)
    return (nodes1d+1)/2


def counted(f):
    """Wrap f so the number of evaluations can be asserted on."""
    calls = []
    def g(k):
        calls.append(k)
        return f(k)
    g.calls = calls
    return g


# --- gkorder_from_nk: the nk -> quadrature-order mapping ------------------

def test_gkorder_matches_its_documented_formula():
    """GKorder = 4*ceil(log2(nk))+1, with nk<=2 clamped to one bit."""
    for nk in [3,4,5,7,8,9,16,17,31,32,100,1000]:
        assert gkorder_from_nk(nk) == 4*int(np.ceil(np.log2(nk)))+1
    assert gkorder_from_nk(1) == 5 # log2 would give 0 (or -inf) bits
    assert gkorder_from_nk(2) == 5


def test_gkorder_is_odd_and_nondecreasing():
    """qutecipy's integrate() rejects an even order outright, and a denser
    requested mesh must never map to a coarser quadrature."""
    orders = [gkorder_from_nk(nk) for nk in range(1,2000)]
    assert all(o%2==1 for o in orders)
    assert all(b>=a for a,b in zip(orders,orders[1:]))
    # logarithmic, not linear: 1000x the mesh is well under 10x the order
    assert gkorder_from_nk(1000) < 10*gkorder_from_nk(1)


# --- accuracy and tolerance handling --------------------------------------

def test_smooth_integrand_matches_analytic_value():
    """A smooth, everywhere-nonzero integrand is reproduced at every order
    the nk mapping can produce, and raising the order converges towards the
    analytic value rather than drifting -- which is the premise
    gkorder_from_nk's logarithmic mapping rests on.
    int_0^1 int_0^1 cos(2 pi kx)^2 exp(ky) = (e-1)/2."""
    f = lambda k: np.cos(2*np.pi*k[0])**2*np.exp(k[1])
    exact = (np.e-1)/2
    errors = {o: abs(integrate_robust(np.float64,f,o,TOL)-exact)
              for o in [9,13,17,21]}
    assert errors[9] < 1e-4 # even the coarsest order is in the ballpark
    assert errors[13] < errors[9]/100. # spectral, not algebraic, convergence
    # by order 13 it is already at the floating-point floor, so the higher
    # orders can only be compared against that floor, not against each other
    assert all(errors[o] < 1e-10 for o in [13,17,21])


def test_tolerance_is_honoured_and_tightening_it_helps():
    """A loose tolerance must not be silently ignored, and a tight one must
    actually buy accuracy on an integrand the TT cannot represent exactly.
    int exp(-((kx-0.5)^2+(ky-0.5)^2)/0.05) = (pi*0.05)*erf(0.5/sqrt(0.05))^2."""
    from scipy.special import erf
    s = 0.05
    f = lambda k: np.exp(-((k[0]-0.5)**2+(k[1]-0.5)**2)/s)
    exact = np.pi*s*erf(0.5/np.sqrt(s))**2
    loose = abs(integrate_robust(np.float64,f,21,1e-2)-exact)
    tight = abs(integrate_robust(np.float64,f,21,1e-12)-exact)
    assert tight <= loose
    assert tight < 1e-9


def test_complex_dtype_keeps_both_parts():
    """get_dm_qtci integrates complex density-matrix entries, so a complex
    dtype must carry the imaginary part through, not silently drop it.
    int exp(2 i pi kx) ky = 0 + 0i; int (kx + i ky) = 0.5 + 0.5i."""
    got = integrate_robust(np.complex128,lambda k: k[0]+1j*k[1],15,TOL)
    assert abs(got-(0.5+0.5j)) < 1e-9
    got = integrate_robust(np.complex128,
            lambda k: np.exp(2j*np.pi*k[0])*k[1],15,TOL)
    assert abs(got) < 1e-9


# --- the pivot search -----------------------------------------------------

def test_pivot_candidates_cover_the_grid_diagonal_first():
    """The search order must start on the diagonal (so integrands that were
    already working keep their old seed) and then reach every node pair
    exactly once (so exhausting it proves f vanishes on the whole grid)."""
    n = 6
    cand = list(_pivot_candidates(n))
    assert cand[:n] == [(i,i) for i in range(n)]
    assert len(cand) == n*n
    assert len(set(cand)) == n*n


def test_zero_at_the_default_pivot_still_integrates():
    """The reason this wrapper exists: TensorCI2 refuses to start when its
    default seed point (the first node along each axis) is exactly zero.
    f = (kx-k0)(ky-k0) vanishes there by construction; the integral is
    (1/2-k0)^2 and must come out anyway."""
    k0 = gk_nodes(15)[0]
    f = lambda k: (k[0]-k0)*(k[1]-k0)
    assert f([k0,k0]) == 0. # the pathology this test is about
    got = integrate_robust(np.float64,f,15,TOL)
    assert abs(got-(0.5-k0)**2) < 1e-12


def test_zero_on_the_whole_diagonal_still_integrates():
    """A mirror exchanging kx and ky forces f(kx,ky) = -f(ky,kx), so any
    such f vanishes on the entire line kx=ky -- and a product of two of
    them, f=(kx-ky)^2, vanishes there while integrating to 1/6. A pivot
    search that only scanned the diagonal returned exactly 0 here."""
    f = lambda k: (k[0]-k[1])**2
    nodes = gk_nodes(15)
    assert all(f([t,t])==0. for t in nodes) # zero at every diagonal node
    got = integrate_robust(np.float64,f,15,TOL)
    assert abs(got-1/6) < 1e-9


def test_identically_zero_returns_zero_after_scanning_the_full_grid():
    """A true symmetry-protected zero (e.g. a spin-off-diagonal density-
    matrix element of a spin-conserving Hamiltonian) integrates to exactly
    0, and only after f has been checked at every node -- at which point 0
    is the exact value of the quadrature rule, not an extrapolation."""
    GKorder = 15
    nnodes = len(gk_nodes(GKorder))
    f = counted(lambda k: 0.)
    got = integrate_robust(np.complex128,f,GKorder,TOL)
    assert got == 0.
    assert isinstance(got,np.complex128)
    assert len(f.calls) == nnodes*nnodes # the whole grid, no tensor train
    # and every sampled point is a real quadrature node, not a nearby guess
    nodes = set(gk_nodes(GKorder))
    assert all(k[0] in nodes and k[1] in nodes for k in f.calls)


def test_nonzero_integrand_exits_the_scan_immediately():
    """The full-grid scan must be reserved for the zero case: an integrand
    that is nonzero at the first node pays exactly one extra evaluation."""
    GKorder = 15
    f = counted(lambda k: 1.+k[0])
    got = integrate_robust(np.float64,f,GKorder,TOL)
    assert abs(got-1.5) < 1e-12
    k0 = gk_nodes(GKorder)[0]
    assert f.calls[0] == [k0,k0] # first candidate is the first diagonal node
