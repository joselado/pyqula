import numpy as np
import pytest

from pyqula import geometry, meanfield
from pyqula import superconductivity


def _spinless_bdg():
    g = geometry.chain()
    h0 = g.get_hamiltonian(has_spin=False)
    h0.setup_nambu_spinor()
    h = g.get_hamiltonian(has_spin=False)
    h.add_swave(0.3)
    return (h0, h)


def test_identify_superconductivity_classifies_a_spinful_bdg():
    """Positive control: the spinful BdG route is unaffected and still
    reports an s-wave spin singlet."""
    g = geometry.chain()
    h0 = g.get_hamiltonian()
    h0.setup_nambu_spinor()
    h = g.get_hamiltonian()
    h.add_swave(0.3)
    out = meanfield.identify_symmetry_breaking(h0, h)
    assert "up-down pairing" in out
    assert "Spin-singlet superconductivity" in out


def test_a_bdg_built_from_a_spinless_chain_is_classified_as_spinful():
    """A Nambu Hamiltonian is always spinful, so add_swave on a spinless
    chain gives the same spin singlet as on a spinful one, and the
    classifier reads it in the spin x electron-hole basis. It used to build
    a spinless Nambu Hamiltonian that the classifier could only refuse."""
    (h0, h) = _spinless_bdg()
    assert h.check_mode("spinful_nambu")
    out = meanfield.identify_symmetry_breaking(h0, h)
    assert "up-down pairing" in out
    assert "Spin-singlet superconductivity" in out


def test_an_empty_bdg_hamiltonian_is_still_empty():
    """Nothing to report is not an error. This is the shape
    identify_symmetry_breaking passes in: the difference between two
    Hamiltonians, which is zero when nothing broke."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    h.setup_nambu_spinor()
    assert superconductivity.identify_superconductivity(h*0.) == []


def test_identify_superconductivity_is_empty_without_nambu():
    """A Hamiltonian with no electron-hole degree of freedom has no pairing
    to report, and that is not an error."""
    h = geometry.chain().get_hamiltonian()
    assert superconductivity.identify_superconductivity(h) == []
