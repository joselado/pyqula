import numpy as np

from pyqula import geometry
from pyqula.ldos import ldosmap
from pyqula.ldostk.ldoswaves import ldos_waves


def _fixed_random_random(monkeypatch, ks):
    """Replace np.random.random with a stand-in that returns the given
    kpoints in order, one per call -- used so ldosmap's internal call and
    a hand-written serial reference are guaranteed to see the identical
    kpoints, independent of whatever else in the process (e.g. a
    module's own import-time use of the global numpy RNG) may have
    touched np.random's state first."""
    it = iter(ks)
    monkeypatch.setattr(np.random, "random", lambda *a, **kw: next(it))


def test_ldosmap_matches_serial_reference(monkeypatch):
    """ldosmap's default (dense, Hermitian) case is now batched through
    numba prange; check it still agrees with the plain per-kpoint
    ldos_waves loop it replaced, given the same kpoints."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    energies = np.linspace(-1, 1, 10)
    delta = 0.1
    nk = 6
    rng = np.random.RandomState(0)
    ks = [rng.random(3) for _ in range(nk)]

    _fixed_random_random(monkeypatch, ks)
    _, new = ldosmap(h, energies=energies, delta=delta, nk=nk)

    hkgen = h.get_hk_gen()
    ds = [ldos_waves(hkgen(k), k=k, es=energies, delta=delta, operator=None) for k in ks]
    dstot = np.mean(ds, axis=0)
    from pyqula.ldos import spatial_dos
    old = np.array([spatial_dos(h, d) for d in dstot])

    assert np.allclose(new, old)
