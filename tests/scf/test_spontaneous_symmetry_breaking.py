import numpy as np
import pytest

from pyqula import geometry
from pyqula import films
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


@pytest.mark.slow
def test_spontaneous_topological_superconductivity_matches_reference(tmp_path, monkeypatch):
    """Regression check for a spontaneous Kane-Mele-like instability on an
    antiferromagnetic diamond-lattice film (nz=4), at a coarse mesh (nk=3
    instead of 6): the identified symmetry breaking must match the value
    recorded from a known-good run. Marked slow: SCF convergence itself
    drives the runtime, not the k-mesh. Note: e_sum is only reproducible to
    ~1e-6 (residual SCF convergence noise from the random-ish guess), not
    to machine precision."""
    monkeypatch.chdir(tmp_path)
    g = geometry.diamond_lattice()
    g = films.geometry_film(g, nz=4)
    h = g.get_hamiltonian()
    h.add_antiferromagnetism(lambda r: 0.8 * np.sign(r[2]))
    mf = meanfield.guess(h, "random")
    scf = meanfield.Vinteraction(h, V1=3.0, U=0.0, mf=mf, V2=3.0, V3=0.0,
            nk=3, filling=0.5, mix=0.3, compute_normal=True, compute_dd=False,
            compute_anomalous=False)
    assert scf.identify_symmetry_breaking() == ["kanemele"]
    (k, e, c) = scf.hamiltonian.get_bands(operator="sz", nk=20)
    assert abs(np.sum(e)) < 1e-4
    assert np.isclose(np.sum(c), -8.79296635503124e-14, atol=1e-4)


@pytest.mark.slow
def test_spontaneous_rashba_matches_reference(tmp_path, monkeypatch):
    """Regression check for a spontaneous Rashba-like instability from
    V1+V2 interactions on a Zeeman-polarized honeycomb lattice, at a coarse
    mesh (nk=4 instead of 10): the sz-resolved band energies must match
    the values recorded from a known-good run. Marked slow: SCF
    convergence itself drives the runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., 1.0])
    scf = meanfield.Vinteraction(h, V1=1.0, V2=1.0, nk=4, filling=0.5,
            mf=None, mix=0.2)
    (k, e, c) = scf.hamiltonian.get_bands(operator="sz", nk=20)
    assert abs(np.sum(e)) < 1e-4
    assert np.isclose(np.sum(c), 2.6645352591003757e-15, atol=1e-6)
