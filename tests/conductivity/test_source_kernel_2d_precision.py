"""Precision of the source kernel of the simplex optical conductivity in 2D.

The source kernel builds the sum over triangles once for all frequencies,
as vertex sources, Taylor series of close edges and triangles, and energy
bins of the far triangles, then sums it per frequency. The vertex form of
a triangle cancels terms at a frequency far from it, so the kernel loses
some digits there. The invariant checked here is that this loss stays at
the 1e-12 level of the largest entry, on the same pair fields, against an
independent evaluation of the same sum: a numpy longdouble sum (eps about
1e-19) that decides its branch per triangle and per frequency, the exact
divided difference or a Taylor series of 24 terms when the spread is below
0.05 of the distance to the frequency, which is exact to about 1e-16 of
every triangle.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import kubosimplex

LD = np.longdouble
CLD = np.clongdouble

pytestmark = pytest.mark.skipif(np.finfo(np.longdouble).eps > 1e-18,
        reason="numpy longdouble is a plain double on this platform")


def _psi2(e, z):
    x = e-z
    return -(x*np.log(x)-x)


def _edge(a, b, z):
    """psi_2[a,b], the first divided difference of psi_2, for a <= b"""
    h = b-a
    m = (a+b)/LD(2)
    xm = m-z
    out = np.zeros(len(a), dtype=CLD)
    tay = h < LD(0.05)*np.abs(xm)
    if np.any(tay): # -ln x + sum over even n of (h/2x)^n/(n(n+1))
        q2 = (h[tay]/(LD(2)*xm[tay]))**2
        acc = np.zeros(len(q2), dtype=CLD)
        p = np.ones(len(q2), dtype=CLD)
        for n in range(2, 25, 2):
            p = p*q2
            acc = acc+p/LD(n*(n+1))
        out[tay] = -np.log(xm[tay])+acc
    ex = ~tay
    out[ex] = (_psi2(b[ex], z)-_psi2(a[ex], z))/h[ex]
    return out


def _triangle(e0, e1, e2, z):
    """2! psi_2[e0,e1,e2], the average of 1/(z-E) over triangles with the
    sorted vertex energies e0 <= e1 <= e2"""
    s = e2-e0
    c = (e0+e1+e2)/LD(3)
    xc = c-z
    out = np.zeros(len(e0), dtype=CLD)
    tay = s < LD(0.05)*np.abs(xc)
    if np.any(tay): # sum_n 2 h_n(e-c) u^(n+1)/((n+1)(n+2)), u = -1/x
        u = -1/xc[tay]
        w0, w1, w2 = [(e[tay]-c[tay])*u for e in (e0, e1, e2)]
        H0 = np.ones(len(u), dtype=CLD); H1 = H0.copy(); H2 = H0.copy()
        acc = np.ones(len(u), dtype=CLD)
        for n in range(1, 25):
            H0 = w0*H0
            H1 = H0+w1*H1
            H2 = H1+w2*H2
            acc = acc+H2*(LD(2)/LD((n+1)*(n+2)))
        out[tay] = u*acc
    ex = ~tay
    out[ex] = 2*(_edge(e1[ex], e2[ex], z)-_edge(e0[ex], e1[ex], z))/s[ex]
    return out


def _reference(sim, dE, W, trans, diag, ws, delta):
    """the unnormalized simplex sum of the kernels, in longdouble"""
    terms = []
    for p in range(dE.shape[1]):
        for sg in ([1.] if diag[p] else [1., -1.]):
            e = np.sort(sg*dE[sim, p], axis=1).astype(LD)
            Wp = W[:, p, :] if sg > 0 else W[:, p, trans]
            wbar = Wp[sim].astype(CLD).mean(axis=1)
            terms.append((e[:, 0], e[:, 1], e[:, 2], wbar))
    out = np.zeros((len(ws), W.shape[2]), dtype=np.complex128)
    for iw, w in enumerate(ws):
        z = CLD(LD(w)+1j*LD(delta))
        acc = np.zeros(W.shape[2], dtype=CLD)
        for e0, e1, e2, wbar in terms:
            acc = acc+(wbar*_triangle(e0, e1, e2, z)[:, None]).sum(axis=0)
        out[iw] = acc
    return out


def _fields(h, nk, T=0.01):
    ks = kubo._kmesh(h, nk)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, T,
            1e-6*scale, True, True)
    pairs, dE, W = kubosimplex._prune(pairs, dE, W)
    diag = np.array([p[0] == p[1] for p in pairs], dtype=np.bool_)
    return np.ascontiguousarray(dE), np.ascontiguousarray(W), trans, diag


def _spinful_haldane():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_haldane(0.2)
    h.shift_fermi(0.3)
    return h


def _gapped_honeycomb():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    return h


def test_source_kernel_reaches_the_longdouble_sum_in_2d():
    """The spinful Haldane model, every band a degenerate pair, and the
    gapped honeycomb at nk = 128, on frequencies up to the top of the
    spectrum, where the largest error sits, far from most triangles. On
    141 frequencies on [0, 7] t at delta = 1e-3 the error is 7.0e-13,
    6.6e-13 and 7.5e-13 of the largest entry at nk = 128, 256 and 512 on
    the spinful Haldane model, and 6.0e-13 on the gapped honeycomb at
    nk = 512. At delta = 1e-5 the triangles kept whole, those whose vertex
    form would lose more than 3e-11 of their peak, are what holds it: the
    spinful Haldane model at nk = 128 is at 2.8e-12 with them and at 2.4e-10
    without them."""
    nk = 128
    ws = np.linspace(0., 7., 141)[::5]
    sim, frac = kubosimplex.simplices(2, nk)
    haldane = _fields(_spinful_haldane(), nk)
    gapped = _fields(_gapped_honeycomb(), nk)
    for fields, delta, tol in [(haldane, 1e-3, 3e-12), (gapped, 1e-3, 3e-12),
            (haldane, 1e-5, 1e-11)]:
        dE, W, trans, diag = fields
        ref = _reference(sim, dE, W, trans, diag, ws, delta)
        s = kubosimplex._kernels["sources"](sim, dE, W, trans, diag, ws,
                delta)
        assert np.max(np.abs(s-ref)) < tol*np.max(np.abs(ref))
