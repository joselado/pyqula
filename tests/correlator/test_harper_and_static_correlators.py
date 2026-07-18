import numpy as np
import pytest

from pyqula import geometry
from pyqula import densitymatrix


@pytest.mark.slow
def test_harper_correlator_matches_reference(tmp_path, monkeypatch):
    """Regression check for a real-space density-matrix correlator on a
    Harper (incommensurate cosine potential) chain, at a smaller size
    (chain(60) instead of 400): the summed correlator must match the value
    recorded from a known-good run. Marked slow: runtime is dominated by
    fixed overhead (e.g. first-use JIT compilation), not the chain size."""
    monkeypatch.chdir(tmp_path)
    n = 60
    g = geometry.chain(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 2.3 * np.cos(np.sqrt(2) / 2. * r[0]))
    pairs = [(n // 2, n // 2 + i) for i in range(n // 3)]
    y = densitymatrix.restricted_dm(h, mode="full", pairs=pairs).real
    assert np.isclose(np.sum(y), 0.24735491057026135, atol=1e-6)


@pytest.mark.slow
def test_static_correlator_matches_reference(tmp_path, monkeypatch):
    """Regression check for site-to-site correlator expectation values on
    a chain supercell with a chemical potential, at a smaller size
    (supercell(4) instead of 10): the summed correlators must match the
    value recorded from a known-good run. Marked slow: runtime is
    dominated by fixed overhead, not the supercell size."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    g = g.get_supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian()
    h.add_onsite(-1.)
    Pijs = [h.get_operator("correlator", i=0, j=j) for j in range(len(g.r))]
    vevs = h.get_several_vev(Pijs)
    assert np.isclose(np.sum(vevs), 0.5527864045000417, atol=1e-6)
