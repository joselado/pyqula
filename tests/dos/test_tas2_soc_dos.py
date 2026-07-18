import numpy as np

from pyqula.specialhamiltonian import TaS2_SOC


def test_tas2_soc_dos_matches_reference(tmp_path, monkeypatch):
    """Regression check for the DOS of the TaS2 SOC model, at a coarse
    mesh (nk=30 instead of 300): the total DOS must match the value
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    h = TaS2_SOC()
    (e, d) = h.get_dos(nk=30)
    assert np.isclose(np.sum(d), 79.1645365256044, atol=1e-4)
