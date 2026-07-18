import numpy as np

from pyqula import geometry


def test_armchair_ribbon_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a plain honeycomb armchair ribbon at a small
    width (n=10 instead of 40): the band energy sum must match the value
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    g = geometry.honeycomb_armchair_ribbon(10)
    h = g.get_hamiltonian()
    (k, e) = h.get_bands()
    assert np.isclose(np.sum(e), 1.2079226507921703e-13, atol=1e-6)


def test_quantum_spin_hall_armchair_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a Kane-Mele armchair ribbon at a small width
    (n=8 instead of 30): the band energy sum must match the value recorded
    from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_armchair_ribbon(8)
    h = g.get_hamiltonian()
    h.add_kane_mele(.1)
    (k, e) = h.get_bands()
    assert np.isclose(np.sum(e), 1.021405182655144e-13, atol=1e-6)


def test_haldane_armchair_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a spinful Haldane armchair ribbon at a small
    width (n=8 instead of 30): the band energy sum must match the value
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_armchair_ribbon(8)
    h = g.get_hamiltonian(has_spin=True)
    h.add_haldane(.1)
    (k, e) = h.get_bands()
    assert np.isclose(np.sum(e), -9.769962616701378e-15, atol=1e-6)
