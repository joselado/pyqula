"""Scaling laws of the entanglement entropy, which pin the ABSOLUTE
normalization of the correlation-matrix formula (unlike the symmetry
checks in test_entanglement_basic.py, which any rescaled version of the
formula would also pass):

* a critical 1D chain must follow the Calabrese-Cardy logarithmic law with
  central charge c=1 (Calabrese & Cardy, J. Stat. Mech. P06002 (2004);
  the free-fermion/correlation-matrix side is reviewed in Peschel &
  Eisler, J. Phys. A 42, 504003 (2009));
* a gapped 2D insulator must obey the area (here boundary) law: the
  entropy per unit length of the cut saturates as the region grows.
"""

import numpy as np

from pyqula import geometry


def test_critical_chain_follows_the_c1_logarithmic_law():
    """Half-filled nearest-neighbor chain on a ring of L sites, cut into
    two arcs of L/2. Conformal field theory predicts

        S(l,L) = (c/3) ln[ (L/pi) sin(pi l/L) ] + const

    with c=1 for a single free-fermion channel. Only rings with L = 2 mod
    4 are used: for L divisible by 4 the k-mesh hits the Fermi points, the
    ground state is degenerate and the entropy is not defined (that case
    is asserted to raise in test_entanglement_basic.py)."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    lengths = [10, 14, 18, 22, 26, 30, 38, 50, 62, 78, 98]
    x, y = [], []
    for L in lengths:
        l = L // 2  # half of the ring
        y.append(h.get_entanglement_entropy(nsuper=L, region=0.5))
        x.append(np.log(L / np.pi * np.sin(np.pi * l / L)))
    slope = np.polyfit(x, y, 1)[0]
    central_charge = 3 * slope
    assert abs(central_charge - 1.0) < 0.05


def test_logarithmic_law_holds_at_fixed_size_versus_region_length():
    """Same law seen the other way round: at fixed ring size, sweeping the
    length l of the region must trace out the chord function
    ln[(L/pi) sin(pi l/L)] with the same slope c/3."""
    L = 62
    h = geometry.chain().get_hamiltonian(has_spin=False)
    ls = [8, 12, 16, 20, 24, 28, 31]
    y = [h.get_entanglement_entropy(nsuper=L, region=list(range(l)))
         for l in ls]
    x = [np.log(L / np.pi * np.sin(np.pi * l / L)) for l in ls]
    slope = np.polyfit(x, y, 1)[0]
    assert abs(3 * slope - 1.0) < 0.05


def test_gapped_two_dimensional_insulator_obeys_the_area_law():
    """Entropy per parallel unit cell of a gapped honeycomb insulator,
    for regions of increasing depth. A volume law would grow linearly with
    the number of cells in the region; the boundary law makes it saturate
    once the region is deeper than the correlation length."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.6)  # open a trivial gap
    entropies = [h.get_entanglement_entropy(nsuper=n, nk=12)
                 for n in [4, 8, 12, 16]]
    # saturation: the last increments are tiny compared with the entropy
    assert entropies[0] > 0.1
    assert abs(entropies[3] - entropies[2]) < 1e-3
    assert abs(entropies[3] - entropies[2]) < 0.05 * abs(
        entropies[1] - entropies[0])
    # a volume law would have doubled the entropy from 8 to 16 cells
    assert abs(entropies[3] - entropies[1]) < 0.05 * entropies[1]
