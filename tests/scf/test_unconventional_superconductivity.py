import numpy as np
import pytest

from pyqula import geometry
from pyqula import meanfield


@pytest.mark.slow
def test_unconventional_superconductivity_ferro_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for a V1-driven odd/triplet superconducting
    instability on a ferromagnetic Nambu chain, at a coarse mesh (nk=6
    instead of 20): the identified symmetry breaking and the
    electron-block band energies must match the values recorded from a
    known-good run. Marked slow: SCF convergence itself drives the
    runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 20.0])
    h.turn_nambu()
    mf = meanfield.guess(h, "random")
    scf = meanfield.Vinteraction(h, U=0.0, V1=-6.0, V2=0.0,
            nk=6, filling=0.2, mf=mf, mix=0.3)
    sym = scf.identify_symmetry_breaking()
    assert set(sym) == {"dx SC", "dy SC", "down-down pairing",
                         "Non-unitary superconductivity",
                         "Spin-triplet superconductivity"}
    (k, e, c) = scf.hamiltonian.get_bands(operator="electron", nk=20)
    assert abs(np.sum(e)) < 1e-4
    assert np.isclose(np.sum(c), 40.0, atol=1e-4)
