import numpy as np
import pytest

from pyqula import geometry


@pytest.mark.slow
def test_triplet_scf_dvector_non_unitarity_matches_reference(tmp_path, monkeypatch):
    """Regression check for a non-collinear superconducting mean-field
    calculation (Nambu spinor, random init) on a ferromagnetic triangular
    lattice: the spin-triplet d-vector non-unitarity must match the value
    recorded from a known-good run. Marked slow: the SCF convergence itself
    (not the k-mesh) drives the runtime here -- an explicit nk=4 (vs. the
    default nk=8) barely changed it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_exchange([3., 3., 3.])
    h.setup_nambu_spinor()
    h = h.get_mean_field_hamiltonian(V1=-1.0, filling=0.3, mf="random", nk=4)
    d = h.get_dvector_non_unitarity()
    assert np.allclose(d, 0.02182528, atol=1e-4)
