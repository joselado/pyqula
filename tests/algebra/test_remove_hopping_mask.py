import numpy as np

from pyqula import geometry


def _reference_remove_hopping(m, r, has_spin, f):
    """Verbatim pre-refactor per-site double loop that
    htk.hamiltonianmodify.remove_hopping used to zero rows/cols with."""
    m = m.copy()
    for i in range(len(r)):
        if f(r[i]):
            if has_spin:
                for j in range(m.shape[0]):
                    m[2*i, j] = 0.
                    m[2*i+1, j] = 0.
                    m[j, 2*i+1] = 0.
                    m[j, 2*i] = 0.
            else:
                for j in range(m.shape[0]):
                    m[j, i] = 0.
                    m[i, j] = 0.
    return m


def test_remove_hopping_matches_reference_spinless():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    assert not h.has_spin
    rng = np.random.RandomState(0)
    dim = h.intra.shape[0]
    m0 = rng.random((dim, dim)) + 1j*rng.random((dim, dim))
    h.intra = m0.copy()

    def f(r): return r[0] > 0  # remove roughly half the sites

    from pyqula.htk.hamiltonianmodify import remove_hopping
    h2 = remove_hopping(h, f)
    ref = _reference_remove_hopping(m0, h.geometry.r, False, f)
    assert np.allclose(h2.intra, ref)


def test_remove_hopping_matches_reference_spinful():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.001])  # forces a spinful Hamiltonian
    assert h.has_spin
    rng = np.random.RandomState(1)
    dim = h.intra.shape[0]
    m0 = rng.random((dim, dim)) + 1j*rng.random((dim, dim))
    h.intra = m0.copy()

    def f(r): return r[0] > 0

    from pyqula.htk.hamiltonianmodify import remove_hopping
    h2 = remove_hopping(h, f)
    ref = _reference_remove_hopping(m0, h.geometry.r, True, f)
    assert np.allclose(h2.intra, ref)
