"""Normalization of the local Chern marker of topology.real_space_chern.

The marker of a site is divided by the area the site occupies, which makes
it the Chern number deep inside a crystal. The area used to be estimated
from the number of sites inside a disk of radius r_max/sqrt(3) around the
origin; that disk does not fit inside a triangular island (its radius is
larger than the inradius), so the estimate took in the empty corners and
the marker read 3% low there (0.9688 instead of 1 at the center of the
triangular Haldane island below) and 1.1% high on a square island (1.0110).
Each site now takes the area of its Voronoi cell (topologytk.realspace.
site_areas), which for these islands is exactly the area per site of the
honeycomb lattice, so the interior reads the Chern number to within the
exponentially small finite-size error of the island.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import islands
from pyqula import topology
from pyqula.topologytk.realspace import site_areas

AREA_PER_SITE = 3*np.sqrt(3)/4 # honeycomb lattice, first-neighbor distance 1


def _interior_marker(g, spinful, t2, workdir):
    """Mean marker of the six sites closest to the center of the island"""
    import os
    old = os.getcwd()
    os.chdir(workdir) # real_space_chern writes REAL_SPACE_CHERN.OUT
    try:
        h = g.get_hamiltonian(has_spin=spinful)
        h.add_haldane(t2)
        (r, c) = topology.real_space_chern(h)
    finally:
        os.chdir(old)
    r = np.array(r)
    d = np.linalg.norm(r - r.mean(axis=0), axis=1)
    return np.mean(np.array(c)[np.argsort(d)[:6]])


def _triangle():
    return islands.get_geometry(name="honeycomb", n=8, nedges=3)


def _square():
    return islands.get_geometry(name="honeycomb", n=10, nedges=4, rot=0.0,
                                clean=False)


@pytest.mark.slow
@pytest.mark.parametrize("island", [_triangle, _square])
def test_interior_marker_of_a_spinless_haldane_island_is_one(island,
                                                             tmp_path):
    """The old normalization gave 0.9688 (triangle) and 1.0110 (square)"""
    marker = _interior_marker(island(), False, 0.2, tmp_path)
    assert np.isclose(marker, 1.0, atol=5e-3), marker


@pytest.mark.slow
def test_interior_marker_of_a_spinful_haldane_island_is_two(tmp_path):
    """Two spin copies of the Haldane model, so the Chern number is 2; the
    old normalization gave 1.938"""
    marker = _interior_marker(_triangle(), True, 0.2, tmp_path)
    assert np.isclose(marker, 2.0, atol=1e-2), marker


@pytest.mark.parametrize("island", [_triangle, _square])
def test_every_site_of_a_honeycomb_island_gets_the_area_per_site(island):
    """Interior cells are exactly the lattice's area per site, and the sites
    at the edge, whose cells are open, take the mean of the others"""
    areas = site_areas(island())
    assert np.allclose(areas, AREA_PER_SITE, rtol=1e-8), (areas.min(),
                                                          areas.max())


def test_stacked_sites_share_their_area():
    """Two identical layers on top of each other: each site takes half the
    area of its cell, so the areas still add up to the area of the sample"""
    g = _square()
    g2 = g.copy()
    g2.r = np.concatenate([g.r, g.r + np.array([0., 0., 3.])])
    g2.r2xyz()
    assert np.allclose(site_areas(g2), AREA_PER_SITE/2, rtol=1e-8)


def test_sites_on_a_line_have_no_area():
    """A chain has no area per site, and says so rather than failing inside
    the Voronoi construction"""
    g = geometry.chain().get_supercell(10)
    g.dimensionality = 0
    with pytest.raises(ValueError, match="lie on a line"):
        site_areas(g)
