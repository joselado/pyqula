"""`add_sublattice_imbalance` and `add_antiferromagnetism` used to do nothing
at all, silently, on a geometry whose sublattice they could not use -- so a
caller building "a gapped semiconductor" on a triangular lattice got a
gapless metal and no warning."""
import numpy as np
import pytest

from pyqula import geometry


def test_a_geometry_with_no_sublattice_is_refused():
    for name in ["chain", "triangular_lattice", "square_lattice"]:
        g = getattr(geometry, name)()
        assert not g.has_sublattice
        h = g.get_hamiltonian()
        with pytest.raises(ValueError, match="no sublattice"):
            h.add_sublattice_imbalance(0.3)
        with pytest.raises(ValueError, match="no sublattice"):
            h.add_antiferromagnetism(0.3)


def test_more_than_two_sublattices_is_refused_for_the_imbalance():
    """kagome's sublattice index runs 0,1,2 rather than +-1, so a single mass
    has no staggering to apply. The antiferromagnet has a frustrated
    implementation for the same geometry, and keeps working."""
    g = geometry.kagome_lattice()
    assert g.sublattice_number == 3
    h = g.get_hamiltonian()
    with pytest.raises(ValueError, match="two sublattices"):
        h.add_sublattice_imbalance(0.3)
    h.add_antiferromagnetism(0.3)  # this one is supported
    m = h.get_magnetization()  # the frustrated pattern: 120 degrees, in-plane
    n = np.linalg.norm(m, axis=1)
    assert np.max(n) > 1e-3 and np.allclose(n, n[0], atol=1e-8), m
    assert np.allclose(np.sum(m, axis=0), 0., atol=1e-6), m


def test_the_refusal_names_a_workflow_that_actually_works():
    """The message points at get_supercell + get_sublattice; check that the
    two of them really do produce a gap on a bipartite lattice."""
    g = geometry.chain().get_supercell(2)
    g.get_sublattice()
    assert g.has_sublattice
    h = g.get_hamiltonian()
    h.add_sublattice_imbalance(0.5)
    (k, e) = h.get_bands()
    assert np.min(np.abs(e)) > 0.4  # a gap, where the chain had none
    h2 = g.get_hamiltonian()
    h2.add_antiferromagnetism(0.5)
    mz = h2.get_vev("sz")
    assert np.max(np.abs(mz)) > 1e-3
    assert abs(np.sum(mz)) < 1e-6  # compensated, i.e. Neel


def test_the_two_sublattice_geometries_still_work():
    for name in ["honeycomb_lattice", "bichain", "lieb_lattice"]:
        g = getattr(geometry, name)()
        h = g.get_hamiltonian()
        h.add_sublattice_imbalance(0.3)
        h2 = g.get_hamiltonian()
        h2.add_antiferromagnetism(0.3)
        assert np.max(np.abs(h2.get_vev("sz"))) > 1e-3
