import numpy as np

from pyqula import geometry


def test_aah_model_hofstadter_and_pumping_matches_reference(tmp_path, monkeypatch):
    """Regression check for a spinful AAH bichain, sweeping the modulation
    frequency (Hofstadter butterfly) and the phason phi (Thouless pumping),
    at a small chain (bichain(30) instead of 100) and coarse sweeps (10
    points instead of 100 each): the summed spectra must match the values
    recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.bichain(30)
    g.dimensionality = 0

    def get_energies(omega=0.0, phi=0.0):
        h = g.get_hamiltonian(has_spin=True)
        def fm(r):
            return .5 + .5 * np.cos(2 * np.pi * (omega * r[0] + phi))
        h.add_antiferromagnetism(fm)
        inds, es = h.get_bands()
        return es

    hofstadter = []
    for omega in np.linspace(0., 1., 10):
        hofstadter += list(get_energies(omega=omega))

    pumping = []
    for phi in np.linspace(0., 1., 10):
        pumping += list(get_energies(omega=0.1, phi=phi))

    assert np.isclose(np.sum(hofstadter), -1.4210854715202004e-14, atol=1e-6)
    assert np.isclose(np.sum(pumping), 1.4210854715202004e-14, atol=1e-6)
