import numpy as np

from pyqula import geometry, algebra
from pyqula import klist
from pyqula.filling import eigenvalues


def _serial_reference(h0, nk):
    h = h0.copy(); h = h.get_dense()
    ks = klist.kmesh(h.dimensionality, nk=nk)
    hkgen = h.get_hk_gen()
    es = np.array([algebra.eigvalsh(hkgen(k)) for k in ks])
    return es.reshape(es.shape[0] * es.shape[1])


def test_filling_eigenvalues_matches_serial_reference():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    new = eigenvalues(h, nk=6, batch_size=5) # not a divisor of nk^2
    old = _serial_reference(h, nk=6)
    assert np.allclose(np.sort(new), np.sort(old))


def test_filling_eigenvalues_independent_of_batch_size():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    outs = [eigenvalues(h, nk=6, batch_size=bs) for bs in (1, 5, 36, 100)]
    for o in outs[1:]:
        assert np.allclose(np.sort(o), np.sort(outs[0]))
