import numpy as np

from pyqula import geometry, algebra
from pyqula.htk.eigenvectors import get_eigenvectors, hk_matrix_batch


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


def max_eigenpair_residual(h, es, vs, ks):
    """Return max_i || H(k_i) v_i - e_i v_i ||.

    This is the oracle for eigenvector CONTENT. Unlike the
    conj(vs).T@vs contraction that used to be used here, it is not a
    completeness identity: it is zero only if every row of vs really is
    an eigenvector of this h at its own k-point, with the eigenvalue
    reported next to it. It is invariant under the phase of each
    eigenvector and under degeneracies, which is what makes it usable
    to compare two independent diagonalizations."""
    hks = hk_matrix_batch(h.get_hk_gen(), ks)  # H(k) for every state's k
    resid = np.einsum("iab,ib->ia", hks, vs) - es[:, None]*vs
    return np.max(np.abs(resid))


def test_get_eigenvectors_dense_matches_serial_reference():
    """get_eigenvectors' dense branch is now batched through numba prange
    (parallel_diagonalization); eigenvalues must match the per-k scipy
    reference exactly, and the eigenvectors must solve the eigenvalue
    problem of the very same h, checked through the residual rather than
    through a contraction that any complete orthonormal set satisfies."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    new_es, new_vs, new_ks = get_eigenvectors(h, nk=5, kpoints=True)
    old_es, old_vs = _serial_reference(h, nk=5)
    assert np.allclose(np.sort(new_es), np.sort(old_es))
    assert max_eigenpair_residual(h, new_es, new_vs, new_ks) < 1e-10


def test_eigenvector_oracle_rejects_random_unitaries():
    """Guard on the guard: the residual oracle above must REJECT a
    complete orthonormal set that has nothing to do with h. The
    contraction it replaced (conj(vs).T@vs, identically nk*Identity by
    completeness) accepted exactly this input."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    es, vs, ks = get_eigenvectors(h, nk=5, kpoints=True)
    n = h.intra.shape[0]
    rng = np.random.default_rng(0)
    fake = np.zeros_like(vs)
    for ik in range(vs.shape[0]//n):  # one random unitary per kpoint
        a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
        q, _ = np.linalg.qr(a)
        fake[ik*n:(ik+1)*n] = q.T
    assert np.allclose(np.conj(fake).T @ fake, np.conj(vs).T @ vs, atol=1e-8)
    assert max_eigenpair_residual(h, es, fake, ks) > 1e-3


def test_get_eigenvectors_kvectors_are_one_per_state():
    """kpoints=True returns one k-vector per EIGENSTATE, as an
    (nstates,3) array whose rows are grouped by k-point: rows
    i*n:(i+1)*n all carry kp[i]. Consumers (densitymatrix.full_dm_simultaneous,
    chitk.magneticresponse, ldostk.atomicmultildos) rely on that layout."""
    from pyqula.klist import kmesh
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    es, vs, ks = get_eigenvectors(h, nk=4, kpoints=True)
    n = h.intra.shape[0]
    kp = kmesh(h.dimensionality, nk=4)
    assert isinstance(ks, np.ndarray)
    assert ks.shape == (len(kp)*n, 3)
    assert vs.shape == (len(kp)*n, n) and es.shape == (len(kp)*n,)
    for i in range(len(kp)):
        assert np.allclose(ks[i*n:(i+1)*n], np.array(kp[i]))


def test_get_eigenvectors_single_kpoint_still_works():
    """k=<a single point> takes a different branch (kp = [k]); smoke test
    it wasn't broken by batching the mesh case."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    es, vs = get_eigenvectors(h, k=np.array([0.1, 0.2, 0.0]))
    assert es.shape[0] == h.intra.shape[0]
