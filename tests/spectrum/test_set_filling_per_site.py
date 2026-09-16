"""h.set_filling() with one filling per site.

A per-site array used to go down the lattice-averaged route, which adds
the array to a float and raised TypeError, and even with average=False the
range guard converted the whole array with float() and raised as well. The
per-site solver it should reach already existed. Each filling keeps the
scalar convention, the fraction of the states of that site that are
occupied, so the occupancy per site is filling*2 when spinful."""
import numpy as np
import pytest
from pyqula import geometry


def _island(has_spin):
    g = geometry.chain().supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=has_spin)
    if has_spin: h.add_exchange([0.3, 0.5, 0.2])
    return h


@pytest.mark.parametrize("has_spin", [True, False])
def test_island_reaches_every_site_filling(has_spin):
    h = _island(has_spin)
    f = np.array([0.3, 0.6, 0.45, 0.7])
    h.set_filling(f)
    nper = 2 if has_spin else 1
    # the solver works with the same broadening, so compare with it
    occ = h.get_vev(delta=1e-2)/nper
    assert np.max(np.abs(occ - f)) < 1e-4


def test_periodic_sublattice_filling():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_sublattice_imbalance(0.3)  # generic, the two sites differ
    f = np.array([0.4, 0.6])
    h.set_filling(f)
    occ = h.get_vev(nk=20)/2
    assert np.max(np.abs(occ - f)) < 1e-3
    assert abs(np.mean(occ) - 0.5) < 1e-3  # total filling is the mean


def test_uniform_array_matches_the_individual_scalar_route():
    ha = _island(True)
    hs = _island(True)
    ha.set_filling(np.full(4, 0.4))
    hs.set_filling(0.4, average=False)
    assert np.allclose(ha.get_vev(delta=1e-2), hs.get_vev(delta=1e-2),
                       atol=1e-6)


def test_wrong_length_and_out_of_range_raise():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="one value per site"):
        h.set_filling([0.5])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        h.set_filling([0.5, 1.2])
