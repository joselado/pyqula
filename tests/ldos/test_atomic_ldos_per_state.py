import numpy as np

from pyqula import geometry

# The "atomic" projection writes each eigenstate in real space as
# Psi(r) = sum_n Psi(n) phi(r-r_n) with an exponential orbital phi. The
# orbitals overlap, so |Psi(r)|^2 carries the cross terms between
# neighboring sites, and without a normalization per state a bonding state
# (same sign on neighbors) came out about 50 times brighter than the
# antibonding one at the same density of states. Each state is now
# normalized on the grid so that its integral is the weight it carries in
# the tight-binding map. On a bipartite lattice the spectrum is symmetric,
# so the integrated map at w and -w must be equal, and both must be the
# sum of the tight-binding map at that energy.

DELTA = 0.2
NK = 20
# a coarse grid and a short orbital tail: the normalization is exact on
# any grid, so resolving the profile is not the point here
GRID = dict(dr=0.4, ratomic=1.0)


def _cell_area(x, y):
    xs, ys = np.unique(np.round(x, 10)), np.unique(np.round(y, 10))
    return (xs[1]-xs[0])*(ys[1]-ys[0])


def test_atomic_map_integrates_to_the_tb_map_on_a_ribbon():
    h = geometry.honeycomb_zigzag_ribbon(3).get_hamiltonian(has_spin=False)
    energies = np.array([-2.0, -1.5, -0.5, 0.5, 1.5, 2.0])
    (x, y, es, atomic) = h.get_multildos(projection="atomic",
            energies=energies, delta=DELTA, nk=NK, write=False, **GRID)
    (_, _, _, tb) = h.get_multildos(projection="TB", energies=energies,
            delta=DELTA, nk=NK, write=False)
    integral = np.sum(atomic, axis=1)*_cell_area(x, y)
    reference = np.sum(tb, axis=1)
    assert np.max(np.abs(integral-reference)) < 1e-6*np.max(reference)
    # the spectrum of the bipartite ribbon is symmetric about zero
    assert np.max(np.abs(integral-integral[::-1])) < 1e-6*np.max(integral)
