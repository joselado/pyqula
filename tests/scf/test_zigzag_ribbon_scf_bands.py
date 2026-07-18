import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.slow
def test_zigzag_ribbon_scf_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for a Hubbard mean-field calculation on a zigzag
    ribbon: the band energy sum must match the value recorded from a
    known-good run. Marked slow: the SCF convergence itself (not the
    ribbon width or k-mesh) drives the runtime -- shrinking the ribbon
    width (10->4) and nk (100->20) did not reduce it."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    g = geometry.honeycomb_zigzag_ribbon(4)
    h = g.get_hamiltonian()
    h = h.get_mean_field_hamiltonian(U=1.0, nk=20)
    (k, e) = h.get_bands(nk=20)
    assert np.isclose(np.sum(e), -2.5934809855243657e-13, atol=1e-6)
