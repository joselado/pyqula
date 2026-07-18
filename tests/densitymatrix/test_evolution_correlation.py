import numpy as np
import pytest

from pyqula import geometry
from pyqula import densitymatrix


@pytest.mark.slow
def test_evolution_correlation_kpm_vs_full_matches_reference(tmp_path, monkeypatch):
    """Regression check comparing KPM vs. full density-matrix correlators
    on a bichain supercell with a sublattice imbalance, at a small size
    (supercell(20) instead of 100): the summed KPM and full correlators
    must match the values recorded from a known-good run. Marked slow:
    runtime is dominated by fixed overhead (e.g. first-use JIT
    compilation), not the supercell size."""
    monkeypatch.chdir(tmp_path)
    g = geometry.bichain()
    g = g.supercell(20)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(.03)
    pairs = [(0, i) for i in range(20)]
    y1 = densitymatrix.restricted_dm(h, mode="KPM", pairs=pairs)
    y2 = densitymatrix.restricted_dm(h, mode="full", pairs=pairs)
    assert np.isclose(np.sum(y1), 0.17676826690151742 + 0j, atol=1e-4)
    assert np.isclose(np.sum(y2), 0.17706233781240255 + 0j, atol=1e-4)
