"""Loose ends found by the survey of topological invariants
(future_development/topological_invariants.md), each of which used to give
a silent wrong answer: an operator sector that dropped every state of an
operator not commuting with the Hamiltonian, an operator with eigenvalues
+-i that put every state at zero, a nocc that the Wannier-center routines
accepted and ignored, and a complex momentum whose imaginary part the Bloch
Hamiltonian dropped with only a ComplexWarning."""
import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology
from pyqula.topologytk import topologicalsector


def _kane_mele(rashba=0.):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.1)
    if rashba != 0.: h.add_rashba(rashba)
    return h


def test_operator_sector_of_a_conserved_spin():
    h = _kane_mele()
    sz = h.get_operator("sz")
    cup = topology.get_chern_operator_sector(h, operator=sz, sector=1., nk=10)
    cdn = topology.get_chern_operator_sector(h, operator=sz, sector=-1., nk=10)
    assert abs(abs(cup) - 1.) < 1e-6 and abs(cup + cdn) < 1e-6


def test_operator_sector_refuses_a_non_commuting_operator():
    """With Rashba coupling s_z is not conserved, the eigenvalues of P s_z P
    sit away from +-1 and every state used to be dropped, which gave zero"""
    h = _kane_mele(rashba=0.1)
    with pytest.raises(ValueError, match="does not commute"):
        topology.get_chern_operator_sector(h, operator=h.get_operator("sz"),
                sector=1., nk=10)
    assert abs(h.get_spin_chern(nk=20) - 1.) < 1e-6  # the sign split works


def test_sign_sector_refuses_a_non_hermitian_operator():
    """-i s_z is the mirror z -> -z of a single layer, with eigenvalues
    +-i; the states used to be sorted by the real part of those, zero for
    all of them"""
    h = _kane_mele()
    mirror = -1j*h.get_operator("sz")
    with pytest.raises(ValueError, match="not Hermitian"):
        topologicalsector.get_chern_operator_sign_sector(h, operator=mirror,
                sign=1, nk=10)
    c = topologicalsector.get_chern_operator_sign_sector(h,
            operator=1j*mirror, sign=1, nk=10)  # i M is Hermitian
    assert abs(abs(c) - 1.) < 1e-6


def test_nocc_selects_the_lowest_bands(tmp_path, monkeypatch):
    """Shifting every level above zero leaves no occupied state, and nocc
    follows the two lowest bands anyway, which are the occupied ones of the
    unshifted Hamiltonian"""
    monkeypatch.chdir(tmp_path)  # wannier_centers writes WANNIER_CENTERS.OUT
    h = _kane_mele()
    ref = topology.wannier_centers(h, nk=20, nt=10, full=True)
    assert topology.z2_invariant(h, nk=20, nt=20) == -1
    h.add_onsite(10.)
    m = topology.wannier_centers(h, nk=20, nt=10, full=True, nocc=2)
    assert np.max(np.abs(m - ref)) < 1e-10
    assert topology.z2_invariant(h, nk=20, nt=20, nocc=2) == -1
    assert topology.wannier_winding(h, nk=20, nt=20, nocc=2) == 0
    with pytest.raises(ValueError, match="nocc"):
        topology.wannier_centers(h, nk=10, nt=4, nocc=5)


def test_complex_momentum_is_refused():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hk = h.get_hk_gen()
    k = np.array([0.1, 0.2, 0.])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ComplexWarning either
        assert np.max(np.abs(hk(k.astype(complex)) - hk(k))) < 1e-14
        with pytest.raises(TypeError, match="complex"):
            hk(k + 0.3j)
