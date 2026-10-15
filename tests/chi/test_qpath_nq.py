import numpy as np
import pytest

from pyqula import geometry
from pyqula.klist import get_qpath


def test_vertices_with_nq_are_filled():
    """Explicit vectors with nq are the vertices of a path of nq points,
    so [[0,0,0],[0.5,0,0]] with nq=26 is the segment from 0 to 1/2 in steps
    of 0.02, and not the two points q=0 and q=1/2"""
    g = geometry.chain()
    qs = get_qpath(g, [[0., 0., 0.], [0.5, 0., 0.]], 26)
    assert qs.shape == (26, 3)
    assert np.allclose(qs[:, 0], np.linspace(0., 0.5, 26))


def test_every_vertex_is_on_the_path():
    g = geometry.square_lattice()
    vs = [[0., 0., 0.], [0.5, 0., 0.], [0.5, 0.5, 0.]]
    qs = get_qpath(g, vs, 11)
    assert len(qs) == 11
    for v in vs:
        assert np.min(np.linalg.norm(qs - np.array(v), axis=1)) < 1e-12


def test_literal_vectors_without_nq_or_with_their_number():
    """Without nq, or with nq equal to the number of vectors, the vectors
    are the q-points, as before"""
    g = geometry.chain()
    vs = [[0.1, 0., 0.], [0.13, 0., 0.], [0.4, 0., 0.]] # not equally spaced
    assert np.allclose(get_qpath(g, vs, None), vs)
    assert np.allclose(get_qpath(g, vs, 3), vs)


def test_too_few_points_raises():
    g = geometry.chain()
    with pytest.raises(ValueError, match="nq"):
        get_qpath(g, [[0., 0., 0.], [0.2, 0., 0.], [0.5, 0., 0.]], 2)


def test_plasmon_bands_on_a_filled_segment():
    """The plasmon of the chain with V1=0.6, scanned on a segment given by
    its two ends and nq, is the same as on the explicit list of points"""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    kw = dict(V1=0.6, energies=np.linspace(0.01, 4., 200), delta=2e-2, nk=500)
    (q1, w1, g1) = h.get_plasmon_bands(qpath=[[0.05, 0., 0.], [0.15, 0., 0.]],
                                       nq=3, **kw)
    (q2, w2, g2) = h.get_plasmon_bands(qpath=[[0.05, 0., 0.], [0.1, 0., 0.],
                                              [0.15, 0., 0.]], **kw)
    assert len(w1) > 0 # the plasmon is there
    assert np.allclose(q1, q2) and np.allclose(w1, w2)
