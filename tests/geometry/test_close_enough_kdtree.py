import numpy as np

from pyqula import geometry
from pyqula.multicell import close_enough, close_enough_bruteforce


def test_close_enough_kdtree_matches_bruteforce():
    """close_enough's KD-tree implementation must agree with the O(N^2)
    brute-force reference for both nearby and far-apart shifted-cell
    position sets (the case that used to force the reference's full,
    non-early-exiting scan), and for the empty-input edge case."""
    g = geometry.honeycomb_lattice().get_supercell(6)
    r1 = g.r
    for i in range(-2, 3):
        for j in range(-2, 3):
            rtmp = np.array([ir + i * g.a1 + j * g.a2 for ir in g.r])
            a = close_enough(r1, rtmp, rcut=2.1)
            b = close_enough_bruteforce(r1, rtmp, rcut=2.1)
            assert a == b, (i, j)

    empty = np.zeros((0, 3))
    assert close_enough(empty, r1, rcut=2.1) == False
    assert close_enough(r1, empty, rcut=2.1) == False
