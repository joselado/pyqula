import numpy as np
import pytest

from pyqula import geometry
from pyqula import scftypes


@pytest.mark.slow
def test_spiral_energy_map_triangular_lattice_matches_reference(tmp_path, monkeypatch):
    """Regression check for a spin-spiral energy map over a 2D q-mesh on a
    ferromagnetic-guess SCF ground state (triangular lattice), at a coarse
    mesh (nk=6 instead of 20, 5x5 instead of 20x20 q-grid, nk=4 instead of
    10 for the total energy): the summed energies must match the value
    recorded from a known-good run. Marked slow: SCF convergence plus the
    q-grid sweep drives the runtime.

    The reference was re-recorded once the interaction actually reached the
    SCF: `g=10.0` is the old scftypes.selfconsistency spelling of the
    Hubbard U, and `nkp` of the k-mesh. Neither exists in the signature that
    name is now aliased to, so `g` was dropped and the old value (-30.38)
    came from a U=0 run."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    h0 = g.get_hamiltonian(has_spin=True)
    mf = scftypes.guess(h0, "ferro", fun=[1., 0., 0.])
    scf = scftypes.selfconsistency(h0, filling=0.5, nk=6, U=10.0,
               mf=mf, mix=0.8, maxerror=1e-6)
    hscf = scf.hamiltonian

    qs = np.linspace(-1, 1, 5)
    es = []
    for qx in qs:
        for qy in qs:
            h = hscf.copy()
            q = h.geometry.reciprocal2natural([qx, qy, 0.])
            h.generate_spin_spiral(vector=[0., 0., 1.], qspiral=q)
            es.append(h.total_energy(nk=4, mode="mesh"))

    assert np.isclose(np.sum(es), -97.54127887586074, atol=1e-3)
