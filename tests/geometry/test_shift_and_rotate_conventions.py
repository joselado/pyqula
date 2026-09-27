import numpy as np

from pyqula import geometry


def _one_site(r):
    g = geometry.square_lattice()
    g.dimensionality = 0
    g.r = np.array([r], dtype=float)
    g.r2xyz()
    return g


def test_shift_moves_the_origin_to_r0():
    """the docstring used to say the positions were shifted by r0, while
    the code subtracts it; the documented convention is r -> r - r0"""
    g = _one_site([1.,2.,3.])
    g.shift([0.5,-1.,2.])
    assert np.allclose(g.r[0], [0.5,3.,1.])


def test_rotate_is_clockwise_in_degrees_and_leaves_the_input_alone():
    g = _one_site([1.,0.,0.])
    go = g.rotate(90.)
    assert np.allclose(go.r[0], [0.,-1.,0.])
    assert np.allclose(g.r[0], [1.,0.,0.])
