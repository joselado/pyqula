import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology
from pyqula.strain import graphene_buckling


@pytest.mark.slow
def test_berry_map_buckling_matches_reference(tmp_path, monkeypatch):
    """Regression check for the spatially resolved Berry curvature of a
    buckled (non-uniform strain) honeycomb supercell, at a small size
    (supercell(4) instead of 16): the LDOS and Berry curvature sums must
    match the values recorded from a known-good run. Marked slow: this
    stays a few seconds regardless of nrep (which only affects file
    replication, not the returned values)."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(4)
    h = g.get_hamiltonian(has_spin=False)

    omega = np.pi * 2. / np.sqrt(g.a1.dot(g.a1))
    pot = graphene_buckling(omega=omega, dt=0.2, geometry=g)
    h.add_strain(pot, mode="non_uniform")
    h.shift_fermi(0.1)

    (x, y, d) = h.get_ldos(e=0., nk=2, nrep=1)
    b = topology.Omega_rmap(h, k=[0., 0., 0.0], nrep=3, nk=2,
            integral=False, eps=1e-4, delta=1e-2, operator="valley")

    assert np.isclose(np.sum(d), 0.013664886994499557, atol=1e-6)
    assert np.isclose(np.sum(b), 0.0016220952163156638, atol=1e-6)
