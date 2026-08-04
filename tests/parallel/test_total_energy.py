import numpy as np

from pyqula import geometry, algebra
from pyqula.spectrum import total_energy


def test_total_energy_mesh_matches_serial_reference():
    """total_energy's default (mesh, dense, no KPM/num_bands) case is now
    batched through numba prange; check it still agrees with the plain
    per-kpoint loop it replaced."""
    from pyqula.klist import kmesh
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    nk = 6
    new = total_energy(h, nk=nk, mode="mesh")

    f = h.get_hk_gen()
    kp = kmesh(h.dimensionality, nk=nk)
    old = np.mean([np.sum(algebra.eigvalsh(f(k))[algebra.eigvalsh(f(k)) < 0.0]) for k in kp])
    assert abs(new - old) < 1e-8


def test_total_energy_random_matches_serial_reference(monkeypatch):
    """Same, for mode='random' -- kpoints are drawn from np.random, so use
    a fixed stand-in to guarantee both paths see identical kpoints
    (see tests/ldos/test_ldosmap.py for why this matters)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    nk = 8
    rng = np.random.RandomState(0)
    ks = [rng.random(3) for _ in range(nk)]

    it = iter(ks)
    monkeypatch.setattr(np.random, "random", lambda *a, **kw: next(it))
    new = total_energy(h, nk=nk, mode="random")

    f = h.get_hk_gen()
    old = np.mean([np.sum(algebra.eigvalsh(f(k))[algebra.eigvalsh(f(k)) < 0.0]) for k in ks])
    assert abs(new - old) < 1e-8


def test_total_energy_kpm_and_num_bands_still_use_pcall_path():
    """Smoke test: use_kpm and num_bands both fall back to the original
    pcall path and shouldn't be affected by the batched-mesh change --
    confirmed by running the exact same calls against the pre-change
    code (git stash) and getting the identical (pre-existing, NaN for
    use_kpm on this particular tiny test system) result, so this test
    only pins down that the routing/branching wasn't broken, not the
    numerical output of a path this change doesn't touch."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    e_kpm = total_energy(h, nk=4, mode="mesh", use_kpm=True) # noqa: F841 (just must not raise)
    e_nb = total_energy(h, nk=4, mode="mesh", nbands=2)
    assert np.isfinite(e_nb)
