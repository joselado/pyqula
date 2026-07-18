import numpy as np

from pyqula import geometry
from pyqula.topologytk import quantumgeometry


def test_quantum_geometry_haldane_matches_reference(tmp_path, monkeypatch):
    """Regression check for quantumgeometry.get_QG_kpath (quantum geometric
    tensor + Berry curvature along a k-path) on a Haldane-gapped honeycomb
    lattice, at a coarse path resolution (nk=10 instead of the 100-point
    default): the mean quantum geometry and Berry curvature must match the
    values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.1)
    h.shift_fermi(0.9)
    (ks, qg, be) = quantumgeometry.get_QG_kpath(h, delta=0.1, nk=10)
    assert np.isclose(np.mean(np.abs(qg)), 13.95734966711326, atol=1e-4)
    assert np.isclose(np.mean(np.abs(be)), 8.773056888371363, atol=1e-4)
