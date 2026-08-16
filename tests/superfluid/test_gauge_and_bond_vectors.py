"""The twist must use the *full* bond vector, not just the lattice vector.

pyqula builds Bloch matrices in the lattice (cell) gauge, whose phase
carries only R -- so a twist written as exp(2 pi i q.R), and equally the
shared dH/dk wrapper current.hk_derivative, drops every intracell bond.
On any lattice with more than one orbital per cell that is not the physical
Peierls substitution, and these tests show it in two ways that need no
external reference:

  * on the honeycomb lattice, where one of the three nearest-neighbour
    bonds is intracell, the lattice-gauge twist produces D_xx != D_yy,
    which C3 symmetry forbids;
  * the lattice-gauge weight changes when the very same crystal is
    described with a supercell, since redrawing the unit cell turns
    intercell bonds into intracell ones.

The default ("atomic") twist, built from d_ij = R + r_j - r_i, passes both.
The lattice gauge is kept as an option because it is the convention of the
Peotta/Toermae superfluid-weight literature and of pyqula's own
h.get_quantum_metric(); the difference between them is exactly the
orbital-embedding dependence of the fixed-|Delta| superfluid weight
discussed in Huhtinen, Herzog-Arbeitman, Chew, Bernevig & Toermae,
PRB 106, 014518 (2022)."""
import numpy as np
import pytest

from pyqula import current
from pyqula import geometry
from pyqula.sctk import superfluidweight as sw


def _honeycomb(delta=0.4, mu=0.3):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(mu)
    h.add_swave(delta)
    return h


def test_honeycomb_weight_is_isotropic_in_the_atomic_gauge():
    """C3 symmetry forces D_xx = D_yy on the honeycomb lattice."""
    d = sw.superfluid_weight(_honeycomb(), nk=12, T=0.)
    assert abs(d[0, 0]-d[1, 1]) < 1e-8*d[0, 0], d
    assert abs(d[0, 1]) < 1e-8*d[0, 0], d


def test_lattice_gauge_breaks_the_honeycomb_isotropy():
    """Guards the reason the default is what it is: dropping the intracell
    bond is not a small correction, it breaks a symmetry of the crystal."""
    d = sw.superfluid_weight(_honeycomb(), nk=12, T=0., gauge="lattice")
    assert abs(d[0, 0]-d[1, 1]) > 0.2*d[1, 1], d


def test_weight_is_invariant_under_redescribing_the_crystal_by_a_supercell():
    """D_s is an intensive property of the crystal, so describing the same
    honeycomb lattice with a 2x2 supercell (same physical model, different
    unit cell, half the k-mesh for the same sampling) must give the same
    tensor.  This is the sharpest available statement that the bond vectors
    are right, since a supercell turns intercell bonds into intracell
    ones."""
    h1 = _honeycomb()
    g2 = geometry.honeycomb_lattice().get_supercell(2)
    h2 = g2.get_hamiltonian()
    h2.add_onsite(0.3)
    h2.add_swave(0.4)
    d1 = sw.superfluid_weight(h1, nk=12, T=0.)
    d2 = sw.superfluid_weight(h2, nk=6, T=0.)
    assert np.max(np.abs(d1-d2))/np.max(np.abs(d1)) < 1e-8, (d1, d2)


def test_lattice_gauge_is_not_supercell_invariant():
    h1 = _honeycomb()
    g2 = geometry.honeycomb_lattice().get_supercell(2)
    h2 = g2.get_hamiltonian()
    h2.add_onsite(0.3)
    h2.add_swave(0.4)
    d1 = sw.superfluid_weight(h1, nk=12, T=0., gauge="lattice")
    d2 = sw.superfluid_weight(h2, nk=6, T=0., gauge="lattice")
    assert np.max(np.abs(d1-d2))/np.max(np.abs(d1)) > 0.2, (d1, d2)


def test_single_site_cell_is_gauge_independent():
    """With one orbital per cell there is no intracell bond, so the two
    gauges must coincide exactly."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.4)
    a = sw.superfluid_weight(h, nk=8, T=0.)
    b = sw.superfluid_weight(h, nk=8, T=0., gauge="lattice")
    assert np.allclose(a, b)


def test_lattice_gauge_derivative_matches_current_hk_derivative():
    """In the lattice gauge the twist derivative dH/dQ_a is exactly the
    shared, correctly normalised current.hk_derivative of the tau_z-masked
    Hamiltonian, divided by 2 pi for the square lattice (a=1) where the
    Cartesian and reduced directions coincide.  This keeps the explicit
    construction pinned to the repository's own dH/dk operator, even though
    the atomic-gauge twist cannot go through it."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.4)
    ops = sw.TwistOperators(h, gauge="lattice")
    tau, diag = sw.twist_masks(ops.h)
    hm = ops.h.copy()                      # tau_z-masked copy of h
    hm.intra = ops.h.intra*tau
    for (to, t) in zip(hm.hopping, ops.h.hopping): to.m = t.m*tau
    hd = ops.h.copy()                      # and the mask for the second one
    hd.intra = ops.h.intra*diag
    for (to, t) in zip(hd.hopping, ops.h.hopping): to.m = t.m*diag
    for k in [[0.13, 0.27, 0.], [0.4, 0.1, 0.]]:
        a = ops.A(k)
        for (i, order) in enumerate([[1, 0], [0, 1]]):
            ref = current.hk_derivative(hm, k, order=order)/(2.*np.pi)
            assert np.max(np.abs(a[i]-ref)) < 1e-10
        b = ops.B(k)
        for (key, order) in [((0, 0), [2, 0]), ((0, 1), [1, 1]),
                             ((1, 1), [0, 2])]:
            ref = current.hk_derivative(hd, k, order=order)/(2.*np.pi)**2
            assert np.max(np.abs(b[key]-ref)) < 1e-10


def test_unknown_gauge_raises():
    with pytest.raises(ValueError, match="gauge"):
        sw.superfluid_weight(_honeycomb(), nk=4, gauge="nonsense")
