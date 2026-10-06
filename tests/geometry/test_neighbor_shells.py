import numpy as np
import pytest
from scipy.spatial.distance import pdist

from pyqula import geometry


def _shells_by_brute_force(g, n):
    """Oracle: the n smallest distinct distances among every pair of sites
    of the same supercell neighbor_distances uses"""
    nsuper = max([n//len(g.r) + 3, 3])
    r = np.array(g.supercell(nsuper).r)
    d = np.unique(np.round(pdist(r), 6))
    return d[d > 0.][0:n]


def _geometries():
    g = geometry.square_lattice().get_supercell(4)
    g.dimensionality = 0
    return [g, geometry.chain(), geometry.honeycomb_lattice(),
            geometry.triangular_lattice(), geometry.kagome_lattice(),
            geometry.cubic_lattice(), geometry.diamond_lattice()]


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_neighbor_shells_match_every_pair(n):
    """neighbor_distances finds the shells with a KD-tree inside a growing
    radius, linear in the number of sites, where it used to fill the
    distance of every pair (28 s at 10,000 sites, an 80 GB array at
    100,000); the shells are the same as those of every pair, and there
    are as many as were asked for"""
    for g in _geometries():
        out = g.neighbor_distances(n=n)
        assert np.array_equal(out, _shells_by_brute_force(g, n))
        assert len(out) == n


def test_neighbor_shells_of_a_tiny_island():
    """An island with fewer distinct distances than requested returns the
    ones it has, and one site has none"""
    g = geometry.square_lattice().get_supercell(2)
    g.dimensionality = 0
    assert np.allclose(g.neighbor_distances(n=6), [1., np.sqrt(2.)])
    g = geometry.square_lattice()
    g.dimensionality = 0
    assert len(g.neighbor_distances()) == 0
