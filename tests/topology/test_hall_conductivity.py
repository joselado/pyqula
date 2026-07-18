import numpy as np

from pyqula import geometry
from pyqula import topology


def test_hall_conductivity_vs_chemical_potential_matches_reference(tmp_path, monkeypatch):
    """Regression check for topology.hall_conductivity vs. chemical
    potential on a Zeeman+Rashba honeycomb lattice, at a coarse sweep (5
    mu values instead of 40, nk=8 instead of 20): the summed conductivity
    must match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    mus = np.linspace(-0.7, 0.7, 5)
    sigmas = []
    for mu in mus:
        g = geometry.honeycomb_lattice()
        h = g.get_hamiltonian(has_spin=True)
        h.add_zeeman([0., 0., 0.2])
        h.add_rashba(0.2)
        h.shift_fermi(mu)
        sigmas.append(topology.hall_conductivity(h, nk=8))
    assert np.isclose(np.sum(sigmas), 1.6320443848026955, atol=1e-4)
