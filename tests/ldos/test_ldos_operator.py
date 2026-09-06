import numpy as np

from pyqula import geometry
from pyqula.greentk.selfenergy import bloch_selfenergy
from pyqula.ldos import green2ldos

# get_ldos_tb resolved its `operator` argument and then never used it:
# neither the Green's-function branch nor the diagonalization one saw it,
# so h.get_ldos(operator="sz") returned the plain charge LDOS. dos.get_dos
# honours the same argument, so this was an inconsistency inside the
# library rather than a missing feature.

ARGS = dict(e=0.3, delta=0.1, nk=30, write=False, nrep=1)


def _polarized_lattice():
    h = geometry.square_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.4])
    return h


def test_operator_changes_the_answer():
    h = _polarized_lattice()
    for mode in ["green", "arpack"]:
        plain = h.get_ldos(mode=mode, **ARGS)[2]
        sz = h.get_ldos(mode=mode, operator="sz", **ARGS)[2]
        assert np.max(np.abs(np.array(plain) - np.array(sz))) > 1e-3


def test_the_two_modes_agree_for_a_spin_diagonal_operator():
    """A Zeeman-polarized lattice's eigenstates are eigenstates of sz, so
    the local matrix element and the global-expectation weighting -- the
    two conventions the two modes use -- have to coincide."""
    h = _polarized_lattice()
    green = h.get_ldos(mode="green", operator="sz", **ARGS)[2]
    ed = h.get_ldos(mode="arpack", operator="sz", **ARGS)[2]
    assert np.max(np.abs(np.array(green) - np.array(ed))) < 1e-6


def test_the_two_modes_agree_without_an_operator():
    """The Green's branch's nk sub-branch was missing the 1/pi that both
    its adaptive sibling and the diagonalization path apply, so the same
    LDOS came out a factor pi apart depending on the mode."""
    h = _polarized_lattice()
    green = h.get_ldos(mode="green", **ARGS)[2]
    ed = h.get_ldos(mode="arpack", **ARGS)[2]
    assert np.max(np.abs(np.array(green) - np.array(ed))) < 1e-6


def test_an_absent_observable_maps_to_zero():
    """The Hamiltonian is real and collinear along z, so it carries no sy
    at all. Contracting the Green's function with the operator on one side
    only -- diag(G A) rather than the Hermitian combination -- gives a
    nonzero sy map here, with a sign set by which matrix goes first."""
    h = _polarized_lattice()
    for mode in ["green", "arpack"]:
        sy = h.get_ldos(mode=mode, operator="sy", **ARGS)[2]
        assert np.max(np.abs(np.array(sy))) < 1e-10


def test_the_map_sums_to_the_operator_resolved_dos():
    """Summing the operator-resolved LDOS over the whole Hilbert space
    must give -Im Tr(G A)/pi, which is what green.green_operator returns
    for the total DOS -- an independent reference for the normalization."""
    h = _polarized_lattice()
    gb, gs = bloch_selfenergy(h.get_dense(), energy=0.3, delta=0.1,
                              mode="full", nk=30)
    ops = [np.array(h.get_operator(n).get_matrix().todense())
           for n in ["sz", "sx"]]
    rng = np.random.RandomState(0)  # an arbitrary Hermitian observable too
    a = rng.random_sample(gb.shape) + 1j * rng.random_sample(gb.shape)
    ops.append(a + np.conjugate(a).T)
    for op in ops:
        d = green2ldos(gb, op=op)
        ref = -np.trace(np.array(gb) @ op).imag / np.pi
        assert abs(np.sum(d) - ref) < 1e-10 * max(abs(ref), 1.0)


def test_a_k_dependent_operator_is_refused_by_the_green_mode():
    """mode='green' integrates over the Brillouin zone before the operator
    is applied, so it cannot honour a k-dependent one -- better to say so
    than to drop it."""
    import pytest
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(NotImplementedError):
        h.get_ldos(mode="green", operator="valley", **ARGS)
