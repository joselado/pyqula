"""With the kernel switched off the BSE must reduce to the
independent-particle problem: every eigenvalue is a transition energy
e_c(k+Q) - e_v(k) of the underlying mean-field band structure."""
import numpy as np
import pytest

from testutils import gapped_honeycomb, gapped_ionic_chain


@pytest.mark.parametrize("Q", [[0., 0., 0.], [0.25, 0., 0.]])
def test_zero_kernel_gives_the_bare_transitions(Q):
    h = gapped_ionic_chain()
    W = {(0, 0, 0): np.zeros(h.intra.shape, dtype=np.complex128)}
    b = h.get_bse(V=W, Q=Q, nk=8, kernel="none")
    assert np.max(np.abs(np.sort(b.get_energies().real)
                         - np.sort(b.pairs.dE))) < 1e-10


def test_zero_interaction_matrix_and_none_kernel_agree():
    """Passing a vanishing interaction and switching the kernel off are
    two different code paths that must land on the same spectrum."""
    h = gapped_honeycomb()
    zero = {(0, 0, 0): np.zeros(h.intra.shape, dtype=np.complex128)}
    e1 = h.get_exciton_energies(V=zero, nk=6, kernel="full")
    e2 = h.get_exciton_energies(V=zero, nk=6, kernel="none")
    assert np.max(np.abs(np.sort(e1.real) - np.sort(e2.real))) < 1e-10


def test_lowest_binding_energy_vanishes_without_interaction():
    """Only the lowest exciton has zero binding energy with no
    interaction: the binding energy is measured from the lowest
    single-particle transition, so the higher states are above it and
    come out negative by construction."""
    h = gapped_honeycomb()
    zero = {(0, 0, 0): np.zeros(h.intra.shape, dtype=np.complex128)}
    b = h.get_exciton_binding_energies(V=zero, nk=6)
    assert abs(b[0]) < 1e-10
    assert np.all(b[1:] < 1e-10)


def test_tda_and_full_agree_when_there_is_no_interaction():
    """The coupling block B is built entirely from the interaction, so with
    no interaction the Tamm-Dancoff and full problems must coincide."""
    h = gapped_ionic_chain()
    zero = {(0, 0, 0): np.zeros(h.intra.shape, dtype=np.complex128)}
    ef = np.sort(h.get_exciton_energies(V=zero, nk=6, tda=False).real)
    et = np.sort(h.get_exciton_energies(V=zero, nk=6, tda=True).real)
    assert np.max(np.abs(ef - et)) < 1e-10
