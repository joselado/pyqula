import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk.spinchi import _require_onsite_only_V


def _hubbard_chain(U=1.0):
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    n = h.intra.shape[0]//2
    m = np.zeros((2*n, 2*n), dtype=np.complex128)
    for i in range(n):
        m[2*i, 2*i+1] = U/2.
        m[2*i+1, 2*i] = U/2.
    h.V = {(0, 0, 0): m}
    return h


def test_require_onsite_only_v_allows_none():
    h = _hubbard_chain()
    h.V = None
    _require_onsite_only_V(h)  # must not raise


def test_require_onsite_only_v_allows_plain_onsite():
    h = _hubbard_chain()
    _require_onsite_only_V(h)  # must not raise


def test_require_onsite_only_v_rejects_nononsite_key():
    h = _hubbard_chain()
    h.V[(1, 0, 0)] = h.V[(0, 0, 0)]*0.1  # add a neighbor-shell key
    with pytest.raises(ValueError):
        _require_onsite_only_V(h)


def test_require_onsite_only_v_rejects_single_nononsite_key():
    h = _hubbard_chain()
    m = h.V.pop((0, 0, 0))
    h.V[(1, 0, 0)] = m  # onsite key replaced entirely by a bond key
    with pytest.raises(ValueError):
        _require_onsite_only_V(h)


def test_get_magnon_bands_raises_immediately_for_nononsite_v_without_a_scf():
    """A dedicated, fast (no SCF) check that the public API surface
    (Hamiltonian.get_magnon_bands) actually raises, not just the internal
    helper -- complements the slow, real-SCF checks in
    tests/scf/test_rpa_nononsite_ferro_chain.py and
    tests/scf/test_rpa_nononsite_rotational_symmetry.py."""
    h = _hubbard_chain()
    h.V[(1, 0, 0)] = h.V[(0, 0, 0)]*0.1
    with pytest.raises(ValueError):
        h.get_magnon_bands(nq=2, energies=np.linspace(-0.1, 0.1, 3), nk=1)


def test_get_spinchi_ladder_raises_for_nononsite_v():
    h = _hubbard_chain()
    h.V[(1, 0, 0)] = h.V[(0, 0, 0)]*0.1
    with pytest.raises(ValueError):
        h.get_spinchi_ladder(energies=np.linspace(-0.1, 0.1, 3), nk=1)
