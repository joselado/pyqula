import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import pairing


def test_every_advertised_pairing_mode_builds_a_bdg_hamiltonian():
    """Every name the mode dispatch advertises must actually build a pairing.
    "deltaud" was listed and dispatched to a function that does not exist, so
    it raised NameError -- and the self-diagnosing error of the else-branch
    pointed the user straight back at the list it was in. The invariant asked
    of each mode is the defining property of a BdG Hamiltonian: H(k) is
    Hermitian and its spectrum is particle-hole symmetric, E_n(k) = -E_n(-k)."""
    g = geometry.honeycomb_lattice()
    k = np.array([0.137, 0.291, 0.])
    assert len(pairing.pairing_modes) > 0
    for mode in pairing.pairing_modes:
        h = g.get_hamiltonian()
        h.add_pairing(delta=0.3, mode=mode, d=[0., 0., 1.])
        hk = h.get_hk_gen()
        m1 = np.array(hk(k))
        assert np.max(np.abs(m1 - np.conjugate(m1.T))) < 1e-10, mode
        e1 = np.sort(np.linalg.eigvalsh(m1))
        e2 = np.sort(np.linalg.eigvalsh(np.array(hk(-k))))
        assert np.max(np.abs(e1 + e2[::-1])) < 1e-10, mode


@pytest.mark.parametrize("mode", ["deltaud", "swaev", "not_a_mode"])
def test_an_unknown_pairing_mode_lists_the_accepted_ones(mode):
    """A mode selected by a string must be self-diagnosing: the error names
    the offending value and enumerates what is accepted. "deltaud" belongs
    here because the branch that claimed to implement it never could."""
    h = geometry.square_lattice().get_hamiltonian()
    with pytest.raises(ValueError) as e:
        h.add_pairing(delta=0.2, mode=mode)
    msg = str(e.value)
    assert mode in msg
    for name in pairing.pairing_modes:
        assert name in msg, (name, msg)
