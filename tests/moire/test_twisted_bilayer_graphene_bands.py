import numpy as np

from pyqula import specialhamiltonian


def test_twisted_bilayer_graphene_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for specialhamiltonian.twisted_bilayer_graphene at
    the smallest commensurate moire index (n=1): the band energies along
    G-K-M-K'-G must match the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    h = specialhamiltonian.twisted_bilayer_graphene(n=1, ti=0.4, has_spin=False)
    h.set_filling(0.5, nk=1)
    (k, e) = h.get_bands(num_bands=8, kpath=["G", "K", "M", "K'", "G"], nk=8)
    assert e.shape == (104,)
    assert np.isclose(np.sum(e), -13.735331753001446, atol=1e-6)
