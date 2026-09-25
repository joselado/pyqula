"""Z2 invariant of time-reversal-symmetric (class DIII) superconductors.

has_time_reversal_symmetry used to return False for every Nambu Hamiltonian,
so get_topological_invariant computed the Chern number of a time-reversal
symmetric superconductor, which vanishes identically, and its Z2 was never
reached. The checks below are either an independent code path (the Chern
numbers of the two conserved spin sectors) or a case whose answer follows
from the physics without computing anything (the doubled helical edge of a
quantum spin Hall insulator)."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.operators import Operator
from pyqula.sctk.dvector import dvector2delta
from pyqula.topology import get_chern_operator_sector


def _helical(r1, r2, c=1j):
    """First-neighbor helical p-wave, d(dr) = c*(dr_x, dr_y, 0): spin up
    pairs as p_x - i p_y and spin down as p_x + i p_y. With c imaginary the
    pairing is time-reversal symmetric as written, with any other phase only
    up to a gauge transformation."""
    dr = r1 - r2
    if abs(dr.dot(dr) - 1.) < 1e-4:
        d = dvector2delta([dr[0], dr[1], 0.])
        return c*np.array([[d[2], d[0]], [d[1], -d[2]]], dtype=complex)
    return np.zeros((2, 2), dtype=complex)


def _helical_pwave(mu, c=1j):
    """Square lattice (band from -4 to 4) with chemical potential mu"""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-mu)
    h.add_pairing(mode=lambda r1, r2: _helical(r1, r2, c=c), delta=0.3)
    return h


def _rashba_swave(delta):
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_rashba(0.5)
    h.add_onsite(-5.)
    h.add_swave(delta)
    return h


def test_time_reversal_of_a_superconductor_is_detected(tmp_path, monkeypatch):
    """s-wave and helical p-wave pairing preserve time reversal, a Zeeman
    field breaks it, and a global phase of the pairing is a gauge choice
    that must not change the answer."""
    monkeypatch.chdir(tmp_path)
    assert _rashba_swave(0.2).has_time_reversal_symmetry()
    assert _rashba_swave(0.2j).has_time_reversal_symmetry()
    assert _rashba_swave(0.2*np.exp(0.7j)).has_time_reversal_symmetry()
    h = _rashba_swave(0.2)
    h.add_zeeman([0., 0., 0.5])
    assert not h.has_time_reversal_symmetry()
    for c in [1j, 1., np.exp(0.3j)]:
        assert _helical_pwave(-2., c=c).has_time_reversal_symmetry()
    h = _helical_pwave(-2.)
    h.add_zeeman([0., 0., 0.2])
    assert not h.has_time_reversal_symmetry()


@pytest.mark.parametrize("mu", [-5., -2., 2., 5.])
def test_helical_pwave_z2_matches_the_spin_sector_chern_number(mu, tmp_path,
                                                                monkeypatch):
    """Spin is conserved by an in-plane d-vector, and each spin sector is a
    chiral p-wave superconductor with Chern number C_up = -C_dn, so the Z2
    invariant has to be (-1)^C_up. The spin up sector is the electron up
    and the -c_up^dag component of the Nambu spinor. Inside the band
    (|mu| < 4) the sectors have C = -1 and +1, outside they are trivial."""
    monkeypatch.chdir(tmp_path)
    h = _helical_pwave(mu)
    z2 = h.get_topological_invariant(nk=30, nt=30)
    op = Operator(np.diag([1., -1., -1., 1.]))
    cup = get_chern_operator_sector(h, operator=op, sector=1., nk=20)
    cdn = get_chern_operator_sector(h, operator=op, sector=-1., nk=20)
    assert abs(cup + cdn) < 1e-6
    assert abs(cup - round(cup)) < 1e-6
    assert z2 == (-1)**int(round(abs(cup)))
    assert (z2 == -1) == (abs(mu) < 4.)


def test_z2_does_not_depend_on_the_phase_of_the_pairing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    z2s = [_helical_pwave(-2., c=c).get_topological_invariant(nk=20, nt=20)
           for c in [1j, 1., np.exp(0.3j)]]
    assert z2s == [-1, -1, -1]


def test_quantum_spin_hall_insulator_is_a_trivial_superconductor(tmp_path,
                                                                 monkeypatch):
    """The Nambu doubling of a Kane-Mele insulator carries its helical edge
    twice, and an s-wave pairing gaps the pair, so as a class DIII
    superconductor it is trivial although the normal state is not."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_kane_mele(0.1)
    assert h.get_topological_invariant(nk=20, nt=20) == -1
    h.add_swave(0.01)
    assert h.has_time_reversal_symmetry()
    assert h.get_topological_invariant(nk=20, nt=20) == 1
