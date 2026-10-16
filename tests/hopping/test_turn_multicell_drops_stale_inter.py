import numpy as np

from pyqula import geometry, dos


# Hamiltonian.turn_multicell, in place, used to keep h.inter next to the new
# hopping list. It went stale at the first change of the hoppings (add_swave
# on a chain left a 2x2 inter beside a 4x4 intra), and dos_surface, which
# decimates with h.inter, then used the old hopping without any warning.


def test_in_place_turn_multicell_drops_the_superseded_attributes():
    h = geometry.chain().get_hamiltonian()
    h.turn_multicell()
    assert h.inter is None
    h2 = geometry.honeycomb_lattice().get_hamiltonian()
    h2.turn_multicell()
    assert not any(hasattr(h2, name) for name in ("tx", "ty", "txy", "txmy"))


def test_the_surface_dos_reads_the_current_hoppings():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h.turn_multicell()
    h.set_multihopping(2.*h.get_multihopping())  # hopping 1 -> 2
    e = np.array([1.5])
    _, s, b = dos.dos_surface(h, energies=e, delta=1e-3, write=False)
    _, s0, b0 = dos.dos_surface(h.get_no_multicell(), energies=e,
            delta=1e-3, write=False)
    assert np.allclose(b, b0) and np.allclose(s, s0)
    # the bulk DOS of a chain of hopping t, 1/(pi sqrt(4t^2-E^2)), at t=2
    assert abs(b[0]/np.pi - 1./(np.pi*np.sqrt(16. - 1.5**2))) < 1e-3
