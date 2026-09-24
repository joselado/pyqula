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


# particle-hole operator tau_y sigma_y in the per-site Nambu order
# (e_up, e_dn, h_dn, -h_up)
_U4 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
               dtype=complex)


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


@pytest.mark.parametrize("lattice", [geometry.honeycomb_lattice,
                                     geometry.square_lattice])
def test_every_pairing_mode_obeys_fermi_antisymmetry(lattice):
    """The spectrum check above cannot see a pairing that breaks Fermi
    antisymmetry: an odd spin-singlet gives E(k) = -E(-k) all the same. The
    defining identity is on the matrix, U H_R^* U^dag = -H_R for every
    real-space block. The part of a BdG matrix that breaks it adds only a
    constant to the many-body Hamiltonian, yet it shows up in the BdG
    spectrum: "swavez" on a single site with mu=0.4 and delta=0.3 gave a gap
    of 0.5 where exact diagonalization of the many-body operator gives 0.4,
    no pairing at all. "haldane", "antihaldane", "swavez" and "SnnAB" (on a
    lattice that is not bipartite) failed this and were removed."""
    for mode in pairing.pairing_modes:
        h = lattice().get_hamiltonian()
        h.add_onsite(0.3)
        h.add_rashba(0.2)
        h.add_zeeman([0.1, -0.2, 0.3])
        try:
            h.add_pairing(delta=0.3, mode=mode, d=[0.3, -0.5, 0.8])
        except ValueError:  # only a sublattice-resolved mode may refuse
            assert not h.geometry.has_sublattice, mode
            continue
        dd = h.get_multihopping().get_dict()
        n = h.intra.shape[0]//4
        U = np.kron(np.identity(n), _U4)
        for key in dd:
            m = _dense(dd[key])
            assert np.max(np.abs(U@np.conj(m)@U.T + m)) < 1e-12, (mode, key)


@pytest.mark.parametrize("mode", ["haldane", "antihaldane", "swavez",
                                  "SnnAB"])
def test_removed_pairing_modes_are_rejected(mode):
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError):
        h.add_pairing(delta=0.2, mode=mode)


@pytest.mark.parametrize("lattice", [geometry.square_lattice,
                                     geometry.triangular_lattice])
@pytest.mark.parametrize("mode", ["swaveA", "swaveB", "swavesublattice"])
def test_sublattice_modes_need_a_sublattice(lattice, mode):
    """Without a labeled sublattice these added zero pairing without saying
    so (square lattice), or died inside get_index with an IndexError or an
    AttributeError (triangular lattice, or any supercell)"""
    for g in [lattice(), lattice().supercell(2)]:
        h = g.get_hamiltonian()
        with pytest.raises(ValueError, match="sublattice"):
            h.add_pairing(delta=0.2, mode=mode)


def test_sublattice_modes_pair_one_sublattice_on_honeycomb():
    """Control for the guard above: on a bipartite lattice swaveA pairs the
    A sites only, and swaveB the B sites only"""
    from pyqula.sctk.extract import extract_singlet_pairing
    g = geometry.honeycomb_lattice()
    for mode, sub in [("swaveA", 1.), ("swaveB", -1.)]:
        h = g.get_hamiltonian()
        h.add_pairing(delta=0.2, mode=mode)
        ud = np.diag(extract_singlet_pairing(h.intra))
        assert np.allclose(ud, 0.2*(np.array(g.sublattice) == sub)), mode


def test_an_electron_hole_sector_index_out_of_range_raises():
    """get_eh_sector returned the NotImplemented singleton for an index
    above 1, which a caller would then have used as a matrix"""
    from pyqula.superconductivity import get_eh_sector
    h = geometry.chain().get_hamiltonian()
    h.add_swave(0.2)
    for (i, j) in [(2, 0), (0, 2), (-1, 0)]:
        with pytest.raises(ValueError):
            get_eh_sector(h.intra, i=i, j=j)


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
