import numpy as np

from pyqula import geometry
from pyqula import topology


def _z2(soc, mass, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # topology.z2_invariant writes *.OUT files to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_kane_mele(soc)
    if mass != 0.:
        h.add_sublattice_imbalance(mass)
    return topology.z2_invariant(h, nk=20, nt=20)


def test_z2_kane_mele_transition_at_analytic_critical_mass(tmp_path, monkeypatch):
    """The Kane-Mele model is Z2 topological for a sublattice mass below the
    known analytic critical value m_c = 3*sqrt(3)*lambda_SOC, and trivial
    above it."""
    soc = 0.05
    mc = 3 * np.sqrt(3) * soc
    z2_below = _z2(soc, 0.5 * mc, tmp_path, monkeypatch)
    z2_above = _z2(soc, 1.5 * mc, tmp_path, monkeypatch)
    assert z2_below != z2_above
