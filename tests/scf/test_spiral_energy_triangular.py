import numpy as np

from pyqula import geometry


def test_spiral_energy_triangular_lattice_matches_reference(tmp_path, monkeypatch):
    """Regression check for the spin-spiral total energy as a function of
    q-vector on a ferromagnetic triangular lattice, for two hopping
    anisotropies (shrunk from 5 to 2) at a coarse q-path and k-mesh (nk=10
    for the path, nk=4 for the energy instead of 100/10): the summed
    energies must match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    ts = [[1.0, 0.0], [1.0, 0.4]]

    all_es = []
    for t in ts:
        g = geometry.triangular_lattice()
        h0 = g.get_hamiltonian(tij=t)
        qpath = h0.geometry.get_kpath(["G", "K", "M", "G"], nk=10)
        for q in qpath:
            h = h0.copy()
            h.generate_spin_spiral(vector=[0., 0., 1.], qspiral=q, fractional=True)
            h.add_zeeman([10., 0., 0.0])
            all_es.append(h.total_energy(nk=4, mode="mesh"))

    assert np.isclose(np.sum(all_es), -263.1717252451515, atol=1e-4)
