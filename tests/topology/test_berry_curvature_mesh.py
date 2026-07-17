import numpy as np

from pyqula import geometry, topology
from pyqula.topologytk.berry import berry_curvature_mesh

DK = 0.01


def _random_ks(n, seed=0):
    rng = np.random.RandomState(seed)
    return np.array([[rng.random(), rng.random(), 0.0] for _ in range(n)])


def test_berry_curvature_mesh_matches_serial_reference_topological():
    """The batched, numba-parallel Wilson-loop calculation must agree
    with the existing per-kpoint topology.berry_curvature, kpoint by
    kpoint, on a Hamiltonian with a nontrivial (Haldane) gap."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    ks = _random_ks(20)
    ref = np.array([topology.berry_curvature(h, k, dk=DK) for k in ks])
    batch = berry_curvature_mesh(h, ks, dk=DK, batch_size=7) # not a divisor of 20
    assert np.max(np.abs(ref - batch)) < 1e-8


def test_berry_curvature_mesh_matches_serial_reference_trivial():
    """Same, for a topologically trivial Hamiltonian (no Haldane flux)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    ks = _random_ks(20, seed=1)
    ref = np.array([topology.berry_curvature(h, k, dk=DK) for k in ks])
    batch = berry_curvature_mesh(h, ks, dk=DK, batch_size=5)
    assert np.max(np.abs(ref - batch)) < 1e-8


def test_berry_curvature_mesh_independent_of_batch_size():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    ks = _random_ks(15, seed=2)
    outs = [berry_curvature_mesh(h, ks, dk=DK, batch_size=bs) for bs in (1, 4, 15, 100)]
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) < 1e-10


def test_get_berry_curvature_master_matches_serial_reference(tmp_path, monkeypatch):
    """h.get_berry_curvature() (get_berry_curvature_master) applies a
    change of basis (R@ki) before evaluating the Wilson loop; check the
    fast batched path reproduces the same values topology.berry_curvature
    gives directly at the transformed kpoints."""
    monkeypatch.chdir(tmp_path) # writes BERRY_MAP.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    nk = 6
    dk = 1./float(2*nk)
    kx, ky, bs = h.get_berry_curvature(nk=nk, dk=dk, write=False)

    R = np.array(g.get_k2K())
    ks = []
    for x in np.linspace(-1, 1, nk, endpoint=False):
        for y in np.linspace(-1, 1, nk, endpoint=False):
            ks.append([x, y, 0.])
    ref = np.array([topology.berry_curvature(h, R @ k, dk=dk) for k in ks])
    assert np.max(np.abs(np.array(bs) - ref)) < 1e-8
