import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.slow
def test_silicene_field_ribbon_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a buckled-honeycomb (silicene) ribbon with a
    perpendicular field and Kane-Mele SOC, at a small width (n=4 instead of
    20): the sz-resolved band energies must match the values recorded from
    a known-good run. Marked slow: runtime here is dominated by fixed
    overhead (e.g. first-use JIT compilation), not the ribbon width --
    shrinking further (down to n=2) did not reliably reduce it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.buckled_honeycomb_lattice()
    g = geometry.bulk2ribbon(g, n=4)
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: 0.2 * np.sign(r[2]))
    h.add_kane_mele(0.1)
    (k, e, c) = h.get_bands(operator="sz")
    assert np.isclose(np.sum(e), 3.1317171078626416e-12, atol=1e-6)
    assert np.isclose(np.sum(c), 6.432632204678157e-13, atol=1e-6)


def test_ribbon_velocity_operator_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a Kane-Mele zigzag ribbon's band structure
    resolved by the velocity operator, at a small width (n=4 instead of
    10): the band energies and velocity expectation values must match the
    values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    h.add_kane_mele(0.2)
    (k, e, c) = h.get_bands(operator=h.get_operator("velocity"))
    assert np.isclose(np.sum(e), 1.893596390800667e-12, atol=1e-6)
    assert np.isclose(np.sum(c), -1.5182299861749016e-14, atol=1e-6)
