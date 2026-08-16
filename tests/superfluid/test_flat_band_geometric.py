"""Flat-band limit: a superfluid weight that is purely quantum geometric.

The sawtooth chain (base sites A, apex sites B, A-B hopping t, A-A hopping
t/sqrt(2)) has an exactly flat lower band, isolated from the dispersive one
by a gap sqrt(2) t.  Putting the chemical potential on the flat band makes
the conventional contribution vanish (a flat band has no group velocity),
so the entire superfluid weight is geometric -- the point of Peotta &
Toermae, Nat. Commun. 6, 8944 (2015).

In that isolated-flat-band limit the geometric part must reduce to the
quantum metric of the flat band, Liang et al. Eq. (23),

    D_geom = (2 |Delta|^2/(V N_k)) sum_k [tanh(beta E/2)/E] g(k) ,

with g from pyqula's own, independently implemented quantum geometric
tensor (topologytk/qgt.py).  qgt.py works in pyqula's lattice (cell) gauge
and in reduced coordinates, so the comparison is made with the superfluid
weight in the same gauge (gauge="lattice") and with the metric converted to
Cartesian coordinates -- a metric and a stiffness are not the same object
unless both conventions are matched, and mismatching them is exactly the
kind of silent factor this test exists to catch.  The identity is exact
only for Delta small compared with the band gap, so the test also checks
that the ratio approaches one as Delta shrinks."""
import numpy as np

from pyqula import geometry
from pyqula.kpointstk.kmesh import kmesh
from pyqula.multihopping import MultiHopping
from pyqula.sctk import superfluidweight as sw

FLAT = -np.sqrt(2.)   # energy of the flat band for t=1


def sawtooth_chain(t=1., mu=-FLAT):
    """Sawtooth chain with an exactly flat lower band, shifted by mu so that
    the flat band sits at the Fermi level by default."""
    t2 = t/np.sqrt(2.)
    g = geometry.chain(2)          # two sites per cell, |a1| = 2
    h = g.get_hamiltonian()
    i2 = np.identity(2)            # spin
    intra = np.array([[mu, t], [t, mu]], dtype=np.complex128)
    tp = np.array([[t2, 0.], [t, 0.]], dtype=np.complex128)
    d = {(0, 0, 0): np.kron(intra, i2), (1, 0, 0): np.kron(tp, i2),
         (-1, 0, 0): np.kron(tp, i2).conj().T}
    h.set_multihopping(MultiHopping(d))
    return h


def _metric_cartesian(h0, k, occ_idxs):
    """h0.get_quantum_metric in Cartesian coordinates.  qgt.py differentiates
    with respect to reduced k, so g^red_ij = sum_ab (G_i)_a (G_j)_b g^cart_ab
    with G_i the reciprocal lattice vectors, G_i.a_j = 2 pi delta_ij."""
    gred = np.atleast_2d(h0.get_quantum_metric(k=k, occ_idxs=occ_idxs))
    a1 = h0.geometry.a1
    gg = 2.*np.pi/np.sqrt(a1.dot(a1))       # |G_1| for a chain along a1
    return gred/gg**2


def test_sawtooth_lower_band_is_flat_and_isolated():
    from pyqula import algebra
    h = sawtooth_chain()
    f = h.get_hk_gen()
    es = np.array([np.sort(algebra.eigvalsh(f([k, 0., 0.])))
                   for k in np.linspace(0., 1., 21)])
    assert np.max(np.abs(es[:, 0:2])) < 1e-10        # flat, at E=0
    assert np.min(es[:, 2]) > np.sqrt(2.)-1e-8       # gapped from the rest


def test_flat_band_weight_is_geometric_and_matches_the_quantum_metric():
    nk = 24
    h0 = sawtooth_chain()
    volume = 2.0                                     # |a1| of chain(2)
    ks = kmesh(1, nk=nk)
    # quantum metric of the flat band, traced over its two spin copies
    gs = np.array([_metric_cartesian(h0, k, [0, 1]) for k in ks]).flatten()
    ratios = []
    for delta in [0.2, 0.05, 0.01]:
        h = h0.copy()
        h.add_swave(delta)
        out = sw.superfluid_weight_decomposition(h, nk=nk, T=0.,
                                                 gauge="lattice")
        # a flat band carries no group velocity: no conventional weight
        assert abs(out["conventional"][0, 0]) < 0.03*out["total"][0, 0]
        # E = |Delta| on the flat band, tanh(beta E/2) = 1 at T=0
        pred = 2.*delta**2/(volume*nk*delta)*np.sum(gs)
        ratios.append(out["geometric"][0, 0]/pred)
    assert abs(ratios[-1]-1.) < 0.01, ratios
    # the residual is linear in Delta/gap, so it must shrink monotonically
    assert abs(ratios[0]-1.) > abs(ratios[1]-1.) > abs(ratios[2]-1.)


def test_flat_band_weight_is_geometric_in_the_atomic_gauge_too():
    """The physical (default) gauge gives a different number -- that is the
    orbital-embedding dependence of the flat-band superfluid weight -- but
    the physics is unchanged: still purely geometric, still finite."""
    h = sawtooth_chain()
    h.add_swave(0.05)
    out = sw.superfluid_weight_decomposition(h, nk=24, T=0.)
    assert out["total"][0, 0] > 0.
    assert abs(out["conventional"][0, 0]) < 0.03*out["total"][0, 0]


def test_flat_band_weight_is_linear_in_the_gap():
    """D_s ~ 2 |Delta| int g in a flat band: linear in |Delta|, unlike the
    conventional weight of a dispersive band, which saturates."""
    h0 = sawtooth_chain()
    ds = []
    for delta in [0.01, 0.02, 0.04]:
        h = h0.copy()
        h.add_swave(delta)
        ds.append(sw.superfluid_weight(h, nk=24, T=0.)[0, 0])
    assert abs(ds[1]/ds[0]-2.) < 0.02, ds
    assert abs(ds[2]/ds[1]-2.) < 0.03, ds


def test_flat_band_analytic_matches_finite_difference():
    """The flat-band case is where the geometric physics lives, so the
    twist finite difference is checked there too, in both gauges."""
    for gauge in ["atomic", "lattice"]:
        h = sawtooth_chain()
        h.add_swave(0.1)
        a = sw.superfluid_weight(h, nk=8, T=0., gauge=gauge)
        f = sw.superfluid_weight_finite_difference(h, nk=8, T=0., dQ=3e-4,
                                                   gauge=gauge)
        assert np.max(np.abs(a-f))/np.max(np.abs(a)) < 1e-3, (gauge, a, f)
