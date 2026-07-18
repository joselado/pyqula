import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.slow
def test_gs_density_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_vev (ground-state expectation value of the
    site-resolved density operator) on a 2-cell chain supercell: the total
    density must match the value recorded from a known-good run. Marked
    slow: the runtime here is dominated by fixed one-time cost (e.g. numba
    JIT compilation the first time this code path runs), not by the k-mesh
    -- an explicit nk=5 (vs. the default nk=30) barely changed it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    g = g.get_supercell(2)
    h = g.get_hamiltonian()
    d = h.get_vev(nk=5)
    assert np.isclose(np.sum(d), 2.0, atol=1e-6)
