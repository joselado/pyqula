import numpy as np

from pyqula import geometry
from pyqula.neighbor import find_first_neighbor, find_first_neighbor_bruteforce


def _as_set(pairs):
    return set((int(i), int(j)) for i, j in pairs)


def test_kdtree_neighbor_search_matches_bruteforce():
    """find_first_neighbor's KD-tree search must find exactly the same
    (i,j) first-neighbor pairs as the O(N^2) brute-force reference, for
    both r1==r2 and r1!=r2 (as used for the tx/ty/inter shifted-cell
    hoppings), across several lattices."""
    cases = []
    g = geometry.honeycomb_lattice().get_supercell(6)
    cases.append(("honeycomb_super", g.r, g.r))
    g = geometry.kagome_lattice().get_supercell(4)
    cases.append(("kagome_super", g.r, g.r))
    g = geometry.triangular_lattice().get_supercell(5)
    cases.append(("triangular_super", g.r, g.r))
    g = geometry.honeycomb_lattice().get_supercell(5)
    r2 = [ir + g.a1 for ir in g.r]
    cases.append(("honeycomb_shifted", g.r, r2))
    g = geometry.chain()
    cases.append(("single_atom_no_neighbors", g.r, g.r))

    for name, r1, r2 in cases:
        a = find_first_neighbor(r1, r2)
        b = find_first_neighbor_bruteforce(r1, r2)
        assert _as_set(a) == _as_set(b), name
