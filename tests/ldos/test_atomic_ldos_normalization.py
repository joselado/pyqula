import numpy as np

from pyqula import geometry

# The "atomic" projection of the LDOS envelops each site with an atomic
# orbital on a real-space grid. Its DOS.OUT went through calculate_dos raw
# -- no 1/pi, no average over the Brillouin zone -- while the maps written
# beside it in the same directory had neither factor either, and the maps
# grew with the density of the k-mesh, which an intensive quantity must not
# do. The two invariants below are the same ones the tight-binding
# projection is held to in test_multildos_normalization.py.

DELTA = 0.1
# a coarse real-space grid and a short orbital tail: this test is about
# the normalization of the profile, not about resolving it
GRID = dict(dr=0.5, ratomic=1.0)


def test_the_written_dos_is_the_dos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    energies = np.linspace(-2.0, 2.0, 20)
    es2 = np.linspace(min(energies), max(energies), len(energies)*10)
    ref = h.get_dos(energies=es2, delta=DELTA, nk=6, write=False)[1]
    h.get_multildos(projection="atomic", energies=energies, delta=DELTA,
                    nk=6, **GRID)
    out = np.genfromtxt("MULTILDOS/DOS.OUT").T
    assert np.max(np.abs(out[0]-es2)) < 1e-12  # same energy grid
    assert np.max(np.abs(out[1]-ref)) < 1e-10*np.max(np.abs(ref))


def test_the_map_does_not_depend_on_the_k_mesh(tmp_path, monkeypatch):
    """Refining the Brillouin-zone mesh must leave a local density of
    states alone; without the 1/nk it grew with the number of k-points."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    args = dict(e=0.4, delta=0.3, projection="atomic", **GRID)
    coarse = h.get_ldos(nk=10, **args)[2]
    fine = h.get_ldos(nk=20, **args)[2]
    assert np.max(np.abs(coarse-fine)) < 0.02*np.max(np.abs(fine))
