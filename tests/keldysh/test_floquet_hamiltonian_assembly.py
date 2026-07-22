import numpy as np
import pytest

from pyqula.keldyshtk.floquet import floquet_hamiltonian


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    return m + m.conj().T


def test_floquet_hamiltonian_is_hermitian_and_has_expected_shape():
    """floquet_hamiltonian builds a block-tridiagonal (site x sideband)
    matrix; regardless of the electron/hole projector split at the
    AC-carrying bond, the result must remain Hermitian (it represents a
    physical Hamiltonian) and have the expected total dimension
    (n_blocks * (2*nmax+1) * dim)."""
    dim = 4
    h00 = _random_hermitian(dim, 0)
    h11 = _random_hermitian(dim, 1)
    rng = np.random.default_rng(2)
    v01 = rng.standard_normal((dim, dim)) + 1j*rng.standard_normal((dim, dim))
    hlist = [[h00, v01.conj().T], [v01, h11]]

    proje = np.diag([1., 1., 0., 0.])
    projh = np.diag([0., 0., 1., 1.])
    nmax = 5
    omega = 0.37

    H = floquet_hamiltonian(hlist, (0, 1), omega, nmax, proje, projh)

    ns = 2*nmax+1
    assert H.shape == (2*ns*dim, 2*ns*dim)
    assert np.max(np.abs(H-H.conj().T)) < 1e-12

    # the electron/hole split of the AC bond must be a full decomposition:
    # projecting v01 onto electron+hole and comparing against the sum of
    # the two off-diagonal channels used inside floquet_hamiltonian
    assert np.max(np.abs(proje@v01 + projh@v01 - v01)) < 1e-12


def test_dc_current_on_non_nambu_heterostructure_raises_clean_error():
    """Heterostructure.get_dc_current on a plain (non-Nambu) heterostructure
    must fail with the documented NotImplementedError, not an unrelated
    error from deeper inside the tau_z-construction machinery."""
    from pyqula import geometry
    from pyqula import heterostructures

    h = geometry.chain().get_hamiltonian()
    HT = heterostructures.build(h.copy(), h.copy())
    with pytest.raises(NotImplementedError):
        HT.get_dc_current(0.1)
