import numpy as np
import pytest

from pyqula import specialhamiltonian


@pytest.mark.slow
def test_kekule_bilayer_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a bilayer graphene Hamiltonian with independent
    Kekule distortions on each layer: the band energies and z-position
    expectation values must match the values recorded from a known-good
    run. Marked slow: the supercell(3) multiplier is required by the Kekule
    distortion's periodicity (cannot be shrunk further), and reducing the
    band-structure k-mesh (nk=20 vs. the 400-point default) did not remove
    the dominant cost."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    h = specialhamiltonian.multilayer_graphene(l=[0, 1], ti=0.2)
    h = h.get_supercell(3)
    h.add_kekule(lambda r: (r[2] > 0) * 0.1)
    h.add_kekule(lambda r: (r[2] < 0) * 0.2)
    (k, e, c) = h.get_bands(operator="zposition", nk=20)
    assert np.isclose(np.sum(e), 7.638334409421077e-14, atol=1e-6)
    assert np.isclose(np.sum(c), 6.994405055138486e-14, atol=1e-6)
