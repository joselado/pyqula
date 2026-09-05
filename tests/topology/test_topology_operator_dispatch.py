import numpy as np
import pytest

from pyqula import geometry, topology

KPATH = [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.5, 0.5, 0.0], [0.1, 0.4, 0.0]]


def _hamiltonian():
    """A Kane-Mele Hamiltonian, whose spin- and valley-resolved Berry
    curvatures are genuinely different quantities."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(0.05)
    h.add_sublattice_imbalance(0.2)
    return h


def _curvature(h, operator):
    return np.array(topology.get_berry_curvature_path(h, kpath=KPATH,
                                                      operator=operator,
                                                      nk=len(KPATH)))


def test_string_operator_is_not_always_valley():
    """topology.get_operator's `if op=="valley"` test used to return the
    valley operator in BOTH branches, so an operator-resolved Berry
    curvature asked for "sz" silently computed the valley-projected one.
    The named operator must agree with the explicit Operator instead."""
    h = _hamiltonian()
    sz_by_name = _curvature(h, "sz")
    sz_explicit = _curvature(h, h.get_operator("sz"))
    valley = _curvature(h, "valley")
    assert np.max(np.abs(sz_by_name - sz_explicit)) < 1e-10
    assert np.max(np.abs(sz_by_name - valley)) > 1e-3


def test_valley_string_still_uses_the_projector():
    """The one name that does take projector=True must keep doing so."""
    h = _hamiltonian()
    by_name = _curvature(h, "valley")
    explicit = _curvature(h, h.get_operator("valley", projector=True))
    assert np.max(np.abs(by_name - explicit)) < 1e-10


def test_matrix_operator_is_not_dropped():
    """`type(op)==np.array` is never true (np.array is a function, not a
    type), so a raw matrix used to fall off the end of get_operator and
    come back as None, i.e. unprojected."""
    h = _hamiltonian()
    m = np.array(h.get_operator("sz").get_matrix().todense())
    assert np.max(np.abs(_curvature(h, m) - _curvature(h, "sz"))) < 1e-10
    # and it must differ from the unprojected calculation
    assert np.max(np.abs(_curvature(h, m) - _curvature(h, None))) > 1e-3


def test_unsupported_operator_type_raises():
    """Rather than silently returning None."""
    h = _hamiltonian()
    with pytest.raises(TypeError):
        topology.get_operator(h, 3.5)
