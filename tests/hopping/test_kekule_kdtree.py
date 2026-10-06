import numpy as np

from pyqula import geometry
from pyqula import kekule


def test_hexagon_centers_match_every_pair_in_order():
    """hexagon_centers takes the pairs at distance 2 from a KD-tree instead
    of the n x n array of every pair (80 GB at 100,000 sites); the centers
    are the same and come in the same order, which matters since the
    Kekule registry is grown from the first of them"""
    g = geometry.honeycomb_lattice().get_supercell(5)
    r = np.array(g.r)
    d2 = np.sum((r[:, None, :] - r[None, :, :])**2, axis=-1)
    ii, jj = np.where((d2 > 3.9) & (d2 < 4.1))
    expected = (r[ii] + r[jj])/2.
    assert np.array_equal(kekule.hexagon_centers(r), expected)


def test_sparse_and_dense_kekule_islands_agree():
    """The Kekule bonds of an island are evaluated on the first-neighbor
    pairs only, and a sparse island keeps a sparse matrix"""
    g = geometry.honeycomb_lattice().get_supercell(4)
    g.dimensionality = 0
    out = []
    for sparse in [False, True]:
        h = g.get_hamiltonian(has_spin=False, is_sparse=sparse)
        h.add_kekule(0.1)
        out.append(h.intra)
    assert hasattr(out[1], "toarray") # still sparse
    assert np.max(np.abs(np.asarray(out[0]) - out[1].toarray())) < 1e-14
    assert np.max(np.abs(np.asarray(out[0]) - np.asarray(out[0]).T.conj())) < 1e-14
