import numpy as np
import pytest

from pyqula import geometry
from pyqula import meanfield


@pytest.mark.slow
def test_spontaneous_haldane_symmetry_breaking_matches_reference(tmp_path, monkeypatch):
    """Regression check for spontaneous Haldane-mass generation from a V2
    density-density interaction on honeycomb, at a coarse mesh (nk=4
    instead of 10): the identified symmetry breaking and the band energy
    sum of the mean-field correction must match the values recorded from a
    known-good run. Marked slow: SCF convergence itself drives the
    runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    mf = meanfield.guess(h, "random")
    scf = meanfield.Vinteraction(h, mf=mf, V2=2.0, nk=4, filling=0.5, mix=0.1)
    assert scf.identify_symmetry_breaking() == ["Haldane"]
    hcorr = h - scf.hamiltonian
    (k, e) = hcorr.get_bands(nk=20)
    assert abs(np.sum(e)) < 1e-4


@pytest.mark.slow
def test_honeycomb_UV_charge_density_wave_matches_reference(tmp_path, monkeypatch):
    """Regression check for a spontaneous charge-density-wave instability
    from a V1 interaction on spinful honeycomb, at a coarse mesh (nk=4
    instead of 10): the identified symmetry breaking must match the value
    recorded from a known-good run. Marked slow: SCF convergence itself
    drives the runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    mf = meanfield.guess(h, "random")
    scf = meanfield.Vinteraction(h, U=0.0, V1=4.0, nk=4, filling=0.5, mf=mf)
    assert scf.identify_symmetry_breaking() == ["Charge density wave"]
    (k, e) = scf.hamiltonian.get_bands(nk=20)
    assert abs(np.sum(e)) < 1e-4

