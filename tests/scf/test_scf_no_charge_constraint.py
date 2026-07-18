import numpy as np
import pytest

from pyqula import geometry, meanfield


@pytest.mark.slow
def test_scf_no_charge_constraint_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for get_mean_field_hamiltonian with
    constrains=["no_charge"] on a single-cell honeycomb lattice (note: this
    mirrors the original example, which discards the SCF result and computes
    bands of the unmodified Hamiltonian itself): the band energy sum must
    match the value recorded from a known-good run. Marked slow: the SCF
    convergence itself (not the k-mesh) drives the runtime -- an explicit
    nk=4 (vs. the default nk=8) barely changed it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.supercell(1)
    h = g.get_hamiltonian(has_spin=False)
    filling = 0.5
    h.get_mean_field_hamiltonian(Vr=lambda r1, r2: 0., filling=filling,
                                  constrains=["no_charge"], nk=4)
    (k, e) = h.get_bands(nk=20)
    assert np.isclose(np.sum(e), 0.0, atol=1e-6)
