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


def test_identify_superconductivity_names_itself_on_a_spinless_bdg():
    """Every route below identify_superconductivity reads the pairing out of
    a 4x4 spin x electron-hole block per site, so a spinless BdG is not
    supported. It used to reach the d-vector extraction and surface that
    routine's message, which named the d-vector and not the classifier the
    user actually called; the guard must name its own routine and the
    Hilbert space it needs."""
    (h0, h) = _spinless_bdg()
    with pytest.raises(NotImplementedError) as e:
        superconductivity.identify_superconductivity(h)
    msg = str(e.value)
    assert "identify_superconductivity" in msg
    assert "spinless" in msg
    with pytest.raises(NotImplementedError):
        meanfield.identify_symmetry_breaking(h0, h)


def test_an_empty_spinless_nambu_hamiltonian_is_still_empty():
    """The guard must not turn "there is nothing here" into an error. This is
    the shape identify_symmetry_breaking passes in: the difference between
    two Hamiltonians, which is zero when nothing broke."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    h.setup_nambu_spinor()
    assert superconductivity.identify_superconductivity(h*0.) == []


def test_identify_superconductivity_is_empty_without_nambu():
    """A Hamiltonian with no electron-hole degree of freedom has no pairing
    to report, and that is not an error."""
    h = geometry.chain().get_hamiltonian()
    assert superconductivity.identify_superconductivity(h) == []
