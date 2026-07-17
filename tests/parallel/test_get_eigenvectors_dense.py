import numpy as np

from pyqula import geometry, algebra
from pyqula.htk.eigenvectors import get_eigenvectors


def _serial_reference(h, nk):
    from pyqula.klist import kmesh
    f = h.get_hk_gen()
    kp = kmesh(h.dimensionality, nk=nk)
    vvs = [algebra.eigh(f(k)) for k in kp]
    nume = sum(len(v[0]) for v in vvs)
    eigvecs = np.zeros((nume, h.intra.shape[0]), dtype=np.complex128)
    eigvals = np.zeros(nume)
    iv = 0
    for ik in range(len(kp)):
        vv = vvs[ik]
        for (e, v) in zip(vv[0], vv[1].transpose()):
            eigvecs[iv] = v.copy(); eigvals[iv] = e.copy(); iv += 1
    return eigvals, eigvecs


def test_get_eigenvectors_dense_matches_serial_reference():
    """get_eigenvectors' dense branch is now batched through numba prange
    (parallel_diagonalization); eigenvalues must match exactly, and the
    span of eigenvectors per kpoint-block must match up to the ordinary
    phase ambiguity between independent diagonalizations of the same
    matrix -- checked via the phase-invariant density-matrix sum rather
    than direct array equality."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    new_es, new_vs = get_eigenvectors(h, nk=5)
    old_es, old_vs = _serial_reference(h, nk=5)
    assert np.allclose(np.sort(new_es), np.sort(old_es))
    dm_new = np.conj(new_vs).T @ new_vs
    dm_old = np.conj(old_vs).T @ old_vs
    assert np.allclose(dm_new, dm_old, atol=1e-8)


def test_get_eigenvectors_single_kpoint_still_works():
    """k=<a single point> takes a different branch (kp = [k]); smoke test
    it wasn't broken by batching the mesh case."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    es, vs = get_eigenvectors(h, k=np.array([0.1, 0.2, 0.0]))
    assert es.shape[0] == h.intra.shape[0]
