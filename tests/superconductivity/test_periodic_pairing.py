import numpy as np
import pytest

from pyqula import geometry

# add_pairing evaluates the bond from site i of the cell to site j of the
# cell at R, and the same bond seen from its other end (site j of the cell
# to site i of the cell at -R), at two positions a lattice vector apart. A
# d-vector or an amplitude given as a function of position that is not
# periodic with the lattice gave them two different values, and the BdG
# matrix it built broke Fermi antisymmetry across cells without saying so:
# the part that breaks it only adds a constant to the many-body
# Hamiltonian, and yet it shows up in the spectrum and the non-unitarity.


def test_non_periodic_d_is_refused_before_h_is_touched():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="periodic with the lattice"):
        h.add_pairing(delta=0.3, mode="pwave", d=lambda r: [1., 1j*r[0], 0.])
    assert not h.has_eh # refused before turn_nambu


def test_non_periodic_amplitude_is_refused():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="delta was given as a function"):
        h.add_pairing(delta=lambda r: 0.2+0.05*r[0], mode="extended_swave")


def test_ribbon_d_along_the_periodic_direction_is_refused():
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    with pytest.raises(ValueError):
        h.add_pairing(delta=0.3, mode="pwave", d=lambda r: [1., 1j*r[0], 0.])


def test_periodic_functions_are_accepted():
    # periodic with a supercell commensurate with the modulation
    g = geometry.chain().supercell(3)
    h = g.get_hamiltonian()
    h.add_pairing(delta=0.3, mode="pwave",
                  d=lambda r: [1., 1j*np.cos(2.*np.pi*r[0]/3.), 0.])
    assert h.has_eh
    # a ribbon modulated across its width only
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    h.add_pairing(delta=lambda r: 0.2+0.05*r[1], mode="pwave",
                  d=lambda r: [1., 1j*r[1], 0.])
    assert h.has_eh


def test_zero_dimensional_geometry_takes_any_function():
    g = geometry.chain().supercell(6)
    g.set_finite()
    h = g.get_hamiltonian()
    h.add_pairing(delta=0.3, mode="pwave", d=lambda r: [1., 1j*r[0], 0.])
    assert h.has_eh
