from pyqula import geometry
from pyqula.superconductivity import hopping2deltaud


def test_hopping2deltaud_matches_add_pairing_swave():
    """Building an extended s-wave pairing via hopping2deltaud must give the
    same Hamiltonian as adding it directly with add_pairing."""
    g = geometry.honeycomb_lattice()
    g = g.get_supercell((2, 2))
    h = g.get_hamiltonian()
    h0 = h.copy()

    h1 = h0.copy()
    h1.add_pairing(mode="swave", nn=1, delta=0.1)
    h2 = hopping2deltaud(h0, h0 * 0.1)

    assert (h1 - h2).is_zero(), "hopping2deltaud disagrees with add_pairing(mode='swave')"
