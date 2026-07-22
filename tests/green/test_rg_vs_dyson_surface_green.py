import numpy as np
import pytest

from pyqula.greentk.rg import green_renormalization
from pyqula.green import dyson, gf_convergence


def random_lead(n, seed):
    """Random Hermitian intracell block and random intercell hopping."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    intra = h + h.conj().T
    inter = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return intra, inter


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("energy", [-1.5, 0.0, 1.2])
def test_rg_surface_green_matches_dyson_iteration(n, seed, energy):
    """The renormalization-algorithm surface Green's function
    (greentk.rg.green_renormalization) and the independent fixed-point
    Dyson-equation iteration (green.dyson) solve the same semi-infinite
    lead self-energy equation g = (e - intra - inter@g@inter^dagger)^-1
    through unrelated numerical schemes (quadratically-converging
    decimation vs. linearly-converging mixed fixed point). They must
    agree to numerical precision for the same random Hamiltonian."""
    intra, inter = random_lead(n, seed)
    delta = 0.08

    _, g_surf_rg = green_renormalization(intra, inter, energy=energy, delta=delta)

    conv = gf_convergence("lead")
    conv.eps = delta
    conv.max_error = 1e-10
    conv.mixing = 0.5
    g_surf_dyson = dyson(intra, inter, energy=energy, gf=conv)

    diff = np.max(np.abs(np.array(g_surf_rg) - np.array(g_surf_dyson)))
    assert diff < 1e-6
