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
