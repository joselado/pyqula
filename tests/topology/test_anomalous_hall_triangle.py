"""h.get_anomalous_hall_conductivity, the Fermi-sea integral of the Kubo
Berry curvature, with the triangle and the uniform integration.

The invariants: in a gap the result is the Chern number and the two
integrations are the same number (a fully occupied triangle gives the
uniform weights); in a metal the triangle on a coarse mesh agrees with a
converged dense uniform mesh and converges as 1/nk^2; the result is odd
under time reversal; and it is the antisymmetric part of the Kubo
conductivity of conductivity.py at zero frequency, in that module's units.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula.topologytk import anomaloushall


def _haldane(t2=0.1, imbalance=0.2):
    """Spinless Haldane model with a staggered onsite, C=1 at half filling,
    gap from -0.25 to 0.25, a metal at fermi=0.5"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(imbalance)
    return h


def _rashba_exchange():
    """Rashba plus out-of-plane exchange on the square lattice"""
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0., 0., 0.6])
    return h


def test_insulator_gives_the_chern_number():
    """In the gap both integrations give the Chern number, and they are
    the same number, since every triangle is fully occupied or empty"""
    h = _haldane()
    tri = h.get_anomalous_hall_conductivity(nk=48)
    grid = h.get_anomalous_hall_conductivity(nk=48, integration="grid")
    chern = h.get_chern(nk=40, write=False)
    assert abs(chern-1.) < 1e-9
    assert abs(tri-chern) < 1e-4, (tri, chern)
    assert abs(tri-grid) < 1e-12, (tri, grid)


def test_spin_degenerate_insulator_counts_both_spins():
    """Exact spin degeneracy goes through the multiplet averaging"""
    h = _haldane()
    h.turn_spinful()
    tri = h.get_anomalous_hall_conductivity(nk=48)
    assert abs(tri-h.get_chern(nk=40, write=False)) < 1e-4, tri


def test_time_reversal_reverses_the_metal():
    """sigma_xy is odd under time reversal, which here is t2 -> -t2"""
    up = _haldane(t2=0.1).get_anomalous_hall_conductivity(nk=32, fermi=0.5)
    down = _haldane(t2=-0.1).get_anomalous_hall_conductivity(nk=32, fermi=0.5)
    assert abs(up) > 0.5
    assert abs(up+down) < 1e-9, (up, down)


@pytest.mark.parametrize("build,fermi", [(_haldane, 0.5),
                                         (_rashba_exchange, -0.5)])
def test_triangle_metal_converges_to_the_dense_grid(build, fermi):
    """At T=0.02 the uniform mesh is converged at nk=200 (checked against
    nk=150 here); the triangle on the coarse nk=48 already agrees within
    2e-3, and doubling the mesh brings it at least 2.5 times closer, its
    1/nk^2 convergence"""
    h = build()
    T = 0.02
    ref = h.get_anomalous_hall_conductivity(nk=200, fermi=fermi, T=T,
                                            integration="grid")
    ref2 = h.get_anomalous_hall_conductivity(nk=150, fermi=fermi, T=T,
                                             integration="grid")
    assert abs(ref-ref2) < 2e-5, (ref, ref2) # well below e96
    e48 = abs(h.get_anomalous_hall_conductivity(nk=48, fermi=fermi, T=T)-ref)
    e96 = abs(h.get_anomalous_hall_conductivity(nk=96, fermi=fermi, T=T)-ref)
    assert e48 < 2e-3, e48
    assert e96 < e48/2.5, (e48, e96)


def test_fermi_array_matches_single_calls():
    h = _rashba_exchange()
    fs = np.array([-2., -0.5])
    many = h.get_anomalous_hall_conductivity(nk=24, fermi=fs, T=0.01)
    one = [h.get_anomalous_hall_conductivity(nk=24, fermi=f, T=0.01)
           for f in fs]
    assert np.allclose(many, one, atol=1e-13, rtol=0.)


def test_metal_is_the_zero_frequency_kubo_hall_conductivity():
    """The grid sum is the antisymmetric part of the Kubo-Greenwood sigma_xy
    of conductivity.py at omega=0, in units of e^2/h instead of e^2/hbar
    and with the opposite sign (sigma_xy 2 pi = -C there)"""
    h = _haldane()
    T = 0.05
    ahc = h.get_anomalous_hall_conductivity(nk=40, fermi=0.5, T=T,
                                            integration="grid")
    h2 = h.copy()
    h2.shift_fermi(-0.5)
    (_, s) = conductivity.optical_conductivity(h2, energies=np.array([0.]),
                                               nk=40, T=T, delta=1e-7)
    sxy = (s[0, 0, 1]-s[0, 1, 0]).real/2.
    assert abs(ahc+2.*np.pi*sxy) < 1e-6, (ahc, -2.*np.pi*sxy)


def test_unknown_integration_lists_the_accepted_ones():
    h = _haldane()
    with pytest.raises(ValueError) as err:
        h.get_anomalous_hall_conductivity(nk=8, integration="trapezoid")
    for name in anomaloushall.get_integration_names():
        assert name in str(err.value)
    assert "triangle" in anomaloushall.get_integration_names()
    assert "grid" in anomaloushall.get_integration_names()


def test_refuses_a_one_dimensional_hamiltonian():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="2D"):
        h.get_anomalous_hall_conductivity(nk=8)


def test_refuses_a_nambu_hamiltonian():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.setup_nambu_spinor()
    with pytest.raises(NotImplementedError, match="Nambu"):
        h.get_anomalous_hall_conductivity(nk=8)
