import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpointstk import labels


def test_every_advertised_label_resolves_to_a_kpoint():
    """known_labels used to be a second list kept in sync by hand with the
    if/elif chain of label2k, so a label could be advertised without being
    dispatched. It is now derived from the registry; this pins that every
    advertised label really returns a three-component reduced kpoint."""
    g = geometry.honeycomb_lattice()
    assert len(labels.known_labels) > 0
    for kl in labels.known_labels:
        k = np.array(labels.label2k(g, kl), dtype=float)
        assert k.shape == (3,), (kl, k.shape)
        assert np.all(np.isfinite(k)), kl


def test_known_labels_is_derived_from_the_dispatch():
    assert list(labels.known_labels) == labels.get_label_names()


def test_the_two_hexagonal_corners_are_opposite():
    """K' is defined as -K, and that relation is what makes a G-K-M-K'-G
    path close; it is the one label built by recursion"""
    g = geometry.honeycomb_lattice()
    k = np.array(labels.label2k(g, "K"))
    kp = np.array(labels.label2k(g, "K'"))
    assert np.max(np.abs(k + kp)) < 1e-12
    assert np.max(np.abs(k)) > 1e-3 # not the trivial k=0 solution


@pytest.mark.parametrize("kl", ["Q", "Gamma", "k"])
def test_an_unknown_label_lists_the_accepted_ones(kl):
    g = geometry.honeycomb_lattice()
    with pytest.raises(ValueError) as e:
        labels.label2k(g, kl)
    msg = str(e.value)
    assert kl in msg
    for name in labels.known_labels:
        assert name in msg, (name, msg)


@pytest.mark.parametrize("kl", ["K", "K'"])
def test_hexagonal_corner_on_a_square_lattice_says_why(kl):
    """K used to fail on a square lattice with "no integer combination of
    the two reciprocal vectors has moduli 1.73...", which does not say
    that K is only defined for a hexagonal Brillouin zone"""
    with pytest.raises(ValueError) as e:
        labels.label2k(geometry.square_lattice(), kl)
    assert "hexagonal" in str(e.value)


@pytest.mark.parametrize("build", [geometry.honeycomb_lattice,
    geometry.triangular_lattice, geometry.kagome_lattice,
    lambda: geometry.honeycomb_lattice().get_supercell(3)])
def test_hexagonal_corner_is_still_found_on_hexagonal_lattices(build):
    """the check in front of K must not refuse a lattice it used to accept:
    K sits at a third of the reciprocal vectors on every one of these"""
    k = np.array(labels.label2k(build(), "K"))
    assert np.max(np.abs(np.abs(k[:2]) - 1./3.)) < 1e-9
