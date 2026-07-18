import numpy as np
import pytest

from pyqula import geometry, meanfield


@pytest.mark.slow
def test_scf_sc_critical_temperature_chain_matches_reference(tmp_path, monkeypatch):
    """Regression check for the superconducting gap vs. temperature on an
    attractive-Hubbard Nambu chain, at a coarse mesh (nk=20 instead of 100,
    3 temperatures instead of 10): the maximum gap and summed gap curve
    must match the values recorded from a known-good run. Marked slow: SCF
    convergence itself (run 4 times, once per temperature) drives the
    runtime. Note: only reproducible to ~1e-4 (residual SCF convergence
    noise), not machine precision."""
    monkeypatch.chdir(tmp_path)

    def get(T):
        g = geometry.chain()
        h = g.get_hamiltonian()
        h.turn_nambu()
        h = h.get_mean_field_hamiltonian(U=-.6, nk=20, T=T, mf="random", maxerror=1e-6)
        return h.get_gap() / 2.

    Tmax = get(0.)
    Ts = np.linspace(0., Tmax, 3)
    gs = np.array([get(T) for T in Ts])

    assert np.isclose(Tmax, 0.03903702048239533, atol=1e-4)
    assert np.isclose(np.sum(gs), 0.042088347683341035, atol=1e-3)
