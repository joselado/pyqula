import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.slow
def test_rpa_ferro_chain_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a ferromagnetic Hubbard mean-field calculation
    on a chain: the sz-resolved band energies must match the values
    recorded from a known-good run. Marked slow: the SCF convergence itself
    (not the k-mesh) drives the runtime -- an explicit nk=4 for the SCF and
    nk=20 for get_bands (vs. defaults of 8 and 400) barely changed it."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    g = geometry.chain()
    h = g.get_hamiltonian()
    h = h.get_mean_field_hamiltonian(U=10.0, filling=0.2, mf="ferro", nk=4)
    (k, e, c) = h.get_bands(operator="sz", nk=20)
    assert np.isclose(np.sum(e), 43.99999999999999, atol=1e-4)
    assert np.isclose(np.sum(c), 0.0, atol=1e-6)
