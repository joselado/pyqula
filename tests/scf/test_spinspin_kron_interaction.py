"""The spin-orbital interaction of VJinteraction puts the coupling of every
pair of sites on the four spin entries of the pair, with the weights
[[1,-1],[-1,1]]/4 of Sz_i Sz_j for an exchange channel and [[1,1],[1,1]]
for a density-density one, plus U/2 on the two up-down entries of each
site. It is built with np.kron instead of a Python double loop over the
pairs, and the entries are the same, exactly, since the weights are powers
of two."""
import numpy as np

from pyqula import geometry, specialhopping
from pyqula.scftk import spinspin


def _reference(h, couplings, weights, U=None):
    """The entries of the pairs, one by one"""
    v = specialhopping.shell_interaction(h.geometry, couplings,
            h.geometry.neighbor_distances())
    out = dict()
    for d, m in v.items():
        n = m.shape[0]
        m1 = np.zeros((2*n, 2*n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                for a in range(2):
                    for b in range(2):
                        m1[2*i+a, 2*j+b] = m[i, j]*weights[a][b]
        out[d] = m1
    if U is not None:
        for i in range(len(h.geometry.r)):
            out[(0, 0, 0)][2*i, 2*i+1] += U/2.
            out[(0, 0, 0)][2*i+1, 2*i] += U/2.
    return out


def test_the_kron_builders_give_the_entries_of_every_pair():
    h = geometry.honeycomb_lattice().get_supercell(2).get_hamiltonian(
            has_spin=True)
    vz = spinspin._build_v(h, J1=1.0, J2=0.3, J3=0.1)
    ref = _reference(h, [1.0/2., 0.3/2., 0.1/2.], [[0.25, -0.25], [-0.25, 0.25]])
    vd = spinspin._build_density_v(h, V1=0.6, V2=0.2, U=2.0)
    refd = _reference(h, [0.6/2., 0.2/2., 0./2.], [[1., 1.], [1., 1.]], U=2.0)
    for v, r in [(vz, ref), (vd, refd)]:
        assert set(v) == set(r)
        for d in v:
            assert isinstance(v[d], np.ndarray) and v[d].dtype == np.complex128
            assert np.array_equal(v[d], r[d]), d
