import numpy as np
import pytest

from pyqula import geometry
from pyqula import potentials


@pytest.mark.slow
def test_berry_curvature_disentangle_strained_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for valley-operator-resolved bands on a honeycomb
    supercell(4) with a commensurate sublattice-imbalance potential: the
    band energies and valley expectation values must match the values
    recorded from a known-good run. Marked slow: this is already the
    smallest meaningful supercell for the commensurate potential and
    stays a few seconds."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(4)
    h = g.get_hamiltonian()
    h.remove_spin()
    f3 = potentials.commensurate_potential(g, minmax=[-0.6, 0.6])
    h.add_sublattice_imbalance(f3)
    op = h.get_operator("valley")
    (k, e, c) = h.get_bands(operator=op)
    assert np.isclose(np.sum(e), 1.0631495683810499e-11, atol=1e-6)
    assert np.isclose(np.sum(c), -1.7368051441479793e-14, atol=1e-6)
