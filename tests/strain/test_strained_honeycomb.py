import numpy as np

from pyqula import geometry
from pyqula.strain import graphene_buckling


def test_strained_honeycomb_buckling_matches_reference(tmp_path, monkeypatch):
    """Regression check for a non-uniform (buckling) strain on a honeycomb
    supercell, at a small size (supercell(3) instead of 11) and coarse DOS
    mesh (60 energies, nk=8 instead of 500/30): the summed bands, LDOS, and
    DOS must match the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(3)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)

    omega = np.pi * 2. / np.sqrt(g.a1.dot(g.a1))
    pot = graphene_buckling(omega=omega, dt=0.2, geometry=g)
    h.add_strain(pot, mode="non_uniform")
    (kb, eb) = h.get_bands(num_bands=20)
    (x, y, ld) = h.get_ldos(e=0., nrep=2)
    h.turn_dense()
    (e, d) = h.get_dos(energies=np.linspace(-3.5, 3.5, 60), nk=8, delta=1e-2)

    assert np.isclose(np.sum(eb), 6.394884621840902e-14, atol=1e-6)
    assert np.isclose(np.sum(ld), 0.14831566651161937, atol=1e-4)
    assert np.isclose(np.sum(d), 157.08599238680168, atol=1e-2)
