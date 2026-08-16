"""Absolute normalisation of the superfluid weight, against closed forms.

The Kubo-vs-finite-difference test pins the internal consistency of the
implementation but not its overall scale: both routes would move together
under a wrong factor of 2 from the doubled Nambu basis, or a wrong overall
normalisation of the bond vectors.  These tests pin the scale against
results derived independently of the code:

  * for a one-orbital BdG model with a uniform gap the exact T=0 weight is
        D^{ab} = (1/(V N_k)) sum_k |Delta|^2 v_a v_b / E^3 ,
    E = sqrt(eps^2+|Delta|^2), with eps and the Cartesian velocities v_a
    built here straight from the stored hoppings (that is the T=0 limit of
    Liang et al. Eq. (21), and equals the diamagnetic form
    -(1/(V N_k)) sum_k (eps/E) d_a d_b eps after an integration by parts);
  * its large-|Delta| limit, 2 t^2/|Delta| for the square lattice;
  * D_s -> 0 when the pairing is switched off at finite temperature.

Using a triangular lattice as well as a square one exercises
non-orthogonal lattice vectors, not just the orthonormal a=1 special case;
a chain exercises dimensionality 1.  All of these have one orbital per
cell, so they say nothing about the bond vectors of the twist -- that is
what tests/superfluid/test_gauge_and_bond_vectors.py is for.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpointstk.kmesh import kmesh
from pyqula.sctk import superfluidweight as sw


def _cartesian_dispersion(h0, ks):
    """(eps(k), v(k)) in Cartesian coordinates for a one-orbital spinful
    normal-state Hamiltonian, straight from its real-space hoppings:
    eps(K) = sum_R t_R exp(i K.r_R) with r_R = sum_i R_i a_i, and
    v_a = d eps/d K_a = sum_R t_R (i r_R,a) exp(i K.r_R)."""
    hm = h0.get_multicell()
    g = h0.geometry
    dim = g.dimensionality
    avecs = [g.a1, g.a2, g.a3][0:dim]
    ts = [np.asarray(hm.intra)[0, 0]] + [np.asarray(t.m)[0, 0]
                                         for t in hm.hopping]
    rs = [np.zeros(3)] + [sum(d*a for (d, a) in zip(t.dir[0:dim], avecs))
                          for t in hm.hopping]
    eps = np.zeros(len(ks))
    v = np.zeros((len(ks), dim))
    for (ik, k) in enumerate(ks):
        # Cartesian momentum K = sum_i k_i G_i with G_i.a_j = 2 pi delta_ij
        A = np.array([a[0:dim] for a in avecs])
        G = 2.*np.pi*np.linalg.inv(A).T
        K = np.zeros(3)
        K[0:dim] = np.array(k[0:dim])@G
        for (t, r) in zip(ts, rs):
            ph = np.exp(1j*K.dot(r))
            eps[ik] += (t*ph).real
            v[ik] += (t*1j*r[0:dim]*ph).real
    return eps, v


def _closed_form_weight(h0, delta, nk):
    """(1/(V N_k)) sum_k |Delta|^2 v_a v_b / E^3 on the same mesh"""
    dim = h0.geometry.dimensionality
    ks = kmesh(dim, nk=nk)
    eps, v = _cartesian_dispersion(h0, ks)
    E = np.sqrt(eps**2+delta**2)
    w = delta**2/E**3
    D = np.einsum("k,ka,kb->ab", w, v, v)/len(ks)
    return D/sw._cell_volume(h0.geometry, dim)


@pytest.mark.parametrize("lattice,mu,delta,nk", [
    (geometry.square_lattice, -0.7, 0.5, 40),
    (geometry.square_lattice, 1.3, 0.8, 30),
    (geometry.triangular_lattice, -1.4, 0.6, 60),
    (geometry.chain, -0.5, 0.4, 60),
    ])
def test_one_orbital_weight_matches_the_closed_form(lattice, mu, delta, nk):
    """Pins the absolute normalisation and the Cartesian scale of the twist
    derivatives in one shot."""
    h0 = lattice().get_hamiltonian()
    h0.add_onsite(mu)
    h = h0.copy()
    h.add_swave(delta)
    d = sw.superfluid_weight(h, nk=nk, T=0.)
    ref = _closed_form_weight(h0, delta, nk)
    assert np.max(np.abs(d-ref))/np.max(np.abs(ref)) < 2e-3, (d, ref)


def test_large_gap_limit_of_the_square_lattice():
    """For |Delta| >> bandwidth, D_xx -> (1/V) <(d eps/d K_x)^2>/|Delta| =
    2 t^2/|Delta| on the square lattice."""
    prev = None
    for delta in [4., 8., 16., 32.]:
        h = geometry.square_lattice().get_hamiltonian()
        h.add_swave(delta)
        d = sw.superfluid_weight(h, nk=16, T=0.)[0, 0]
        ratio = d*delta/2.   # -> 1
        assert ratio < 1.
        if prev is not None: assert ratio > prev   # monotone approach
        prev = ratio
    assert prev > 0.99


def test_no_pairing_gives_no_superfluid_weight():
    """A normal metal at finite temperature has zero superfluid weight: the
    paramagnetic and diamagnetic terms cancel exactly.  (At T=0 they do not
    -- there the same expression is the Drude weight, which is why this
    check needs a temperature that the k-mesh resolves.)"""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.0)
    d = sw.superfluid_weight(h, nk=24, T=0.4)
    assert np.max(np.abs(d)) < 1e-4, d


def test_finite_pairing_gives_a_positive_semidefinite_symmetric_tensor():
    """D_s is a second derivative of a minimised grand potential: symmetric,
    and positive semi-definite in a stable superconducting state."""
    for h in [_rashba_honeycomb(), _zeeman_triangular()]:
        for T in [0., 0.1]:
            d = sw.superfluid_weight(h, nk=12, T=T)
            assert np.max(np.abs(d-d.T)) < 1e-10*max(np.max(np.abs(d)), 1.)
            ev = np.linalg.eigvalsh(d)
            assert np.min(ev) > -1e-9, ev
            assert np.max(ev) > 1e-3, ev


def test_spinless_nambu_gives_half_of_the_spinful_weight():
    """pyqula's spinless_nambu mode carries a single spin species, so its
    grand potential -- and with it the weight -- is exactly half of the
    spinful one for the same model."""
    g = geometry.square_lattice()
    hs = g.get_hamiltonian(has_spin=True)
    hs.add_onsite(-0.7) ; hs.add_swave(0.4)
    hl = g.get_hamiltonian(has_spin=False)
    hl.add_onsite(-0.7) ; hl.add_swave(0.4)
    a = sw.superfluid_weight(hs, nk=16, T=0.)
    b = sw.superfluid_weight(hl, nk=16, T=0.)
    assert np.allclose(a, 2.*b)


def _rashba_honeycomb():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.4)
    h.add_rashba(0.3)
    h.add_swave(0.35)
    return h


def _zeeman_triangular():
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_onsite(-1.0)
    h.add_zeeman([0., 0., 0.2])
    h.add_swave(0.6)
    return h
