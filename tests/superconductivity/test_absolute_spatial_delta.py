import numpy as np

from pyqula import geometry


def _site_dependent_swave():
    """Honeycomb lattice with a different on-site gap on each sublattice, so
    the two sites discriminate a normalisation error from a real spatial
    profile."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_swave(lambda r: 0.3 if r[0] < 0 else 0.1)  # 0.3 on A, 0.1 on B
    return h


def test_absolute_spatial_delta_is_the_on_site_gap():
    """h.extract("absolute_spatial_delta") must return |Delta_i| per site,
    not a constant multiple of it. The oracle is a second code path in the
    same file: h.extract("swave") reads the on-site anomalous matrix element
    directly. It used to come out sqrt(2) too large because the resummation
    over the four Nambu components per site was divided by 2 instead of by
    the number of components per site."""
    h = _site_dependent_swave()
    ref = np.abs(h.extract("swave"))
    got = h.extract("absolute_spatial_delta", nk=8)
    assert np.allclose(got, ref, atol=1e-8), (got, ref)


def test_absolute_spatial_delta_rms_equals_absolute_delta():
    """The sibling routine h.extract("absolute_delta") is the k- and
    site-averaged sqrt(<|Delta|^2>), so the root mean square of the spatial
    profile must reproduce it exactly. This pins the relative normalisation
    of the two routines against each other, with no reference number."""
    h = _site_dependent_swave()
    spatial = h.extract("absolute_spatial_delta", nk=8)
    total = np.real(h.extract("absolute_delta", nk=8))
    assert abs(np.sqrt(np.mean(spatial**2)) - total) < 1e-8, (spatial, total)
