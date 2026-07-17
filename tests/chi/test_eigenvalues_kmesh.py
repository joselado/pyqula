import numpy as np

from pyqula import geometry, algebra
from pyqula.spectrum import eigenvalues_kmesh


def _serial_reference(h, nk):
    """Verbatim pre-refactor eigenvalues_kmesh: a plain nested loop, no
    parallelism at all."""
    ne = h.intra.shape[0]
    es = np.zeros((nk, nk, ne))
    hkgen = h.get_hk_gen()
    kx = np.linspace(0., 1., nk, endpoint=False)
    ky = np.linspace(0., 1., nk, endpoint=False)
    for i in range(nk):
        for j in range(nk):
            hk = hkgen([kx[i], ky[j]])
            es[i, j, :] = algebra.eigvalsh(hk)
    return es


def test_eigenvalues_kmesh_matches_serial_reference():
    """eigenvalues_kmesh (used by get_qpi's default "response" mode via
    epsilonk) is now batched through numba prange; check it still agrees
    with the plain nested-loop implementation it replaced."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    nk = 6
    new = eigenvalues_kmesh(h, nk=nk, batch_size=5) # not a divisor of nk*nk
    old = _serial_reference(h, nk)
    assert new.shape == old.shape
    assert np.allclose(np.sort(new, axis=2), np.sort(old, axis=2))


def test_eigenvalues_kmesh_independent_of_batch_size():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    nk = 6
    outs = [eigenvalues_kmesh(h, nk=nk, batch_size=bs) for bs in (1, 4, 36, 100)]
    for o in outs[1:]:
        assert np.allclose(o, outs[0])
