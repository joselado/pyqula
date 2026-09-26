"""An Operator built from a function has no matrix representation.

Some of them cannot have one at all: the unfolding projector acts with a
different matrix at every kpoint. get_matrix() used to return None for
those, and a routine that needed a matrix then carried None into its
calculation, so the operator was quietly dropped instead of refused.
"""

import numpy as np
import pytest

from pyqula import geometry, operators
from pyqula.operators import Operator


def matrix_less(f=lambda v, k=None: v):
    """An operator defined only by its action on a wavefunction"""
    op = Operator(f)
    assert op.matrix is None
    return op


def test_get_matrix_raises_for_an_operator_defined_only_by_its_action():
    with pytest.raises(ValueError):
        matrix_less().get_matrix()


def test_get_matrix_returns_none_when_the_caller_asks_to_handle_it():
    """Three call sites have a better message than the generic one and ask
    for the None back so they can raise it themselves"""
    assert matrix_less().get_matrix(required=False) is None


def test_get_matrix_still_returns_a_matrix_when_there_is_one():
    g = geometry.chain()
    h = g.get_supercell(4).get_hamiltonian(has_spin=True)
    m = h.get_operator("sz").get_matrix()
    assert m is not None
    assert m.shape == (h.intra.shape[0],) * 2


def test_the_unfolding_operator_has_no_matrix_representation():
    """It acts with a different matrix at every kpoint"""
    from pyqula.unfolding import bloch_projector
    g0 = geometry.chain()
    h = g0.get_supercell(3, store_primal=True).get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        bloch_projector(h).get_matrix()


@pytest.mark.parametrize("combine", [lambda a, b: a * b, lambda a, b: a + b])
def test_combining_with_a_matrix_less_operator_leaves_no_matrix(combine):
    """Operator(self) copies self's matrix, and the combination used to keep
    it: get_matrix() then returned the left operand alone and the right one
    was silently dropped"""
    g = geometry.chain()
    h = g.get_supercell(4).get_hamiltonian(has_spin=False)
    n = h.intra.shape[0]
    A = operators.index(h, n=[0])          # a matrix-backed projector
    B = matrix_less(lambda v, k=None: 2.0 * v)  # no matrix
    out = combine(A, B)
    assert out.matrix is None
    with pytest.raises(ValueError):
        out.get_matrix()
    # and the action itself is, and always was, correct
    v = np.ones(n, dtype=np.complex128)
    expected = combine(A, B).m(v)
    assert np.allclose(out.m(v), expected)


def test_multiplying_two_matrix_operators_still_gives_their_product():
    g = geometry.chain()
    h = g.get_supercell(4).get_hamiltonian(has_spin=True)
    A = h.get_operator("sz")
    B = operators.index(h, n=[0])
    prod = (A * B).get_matrix()
    assert np.allclose(np.array((A.get_matrix() @ B.get_matrix()).todense()),
                       np.array(prod.todense()))


def test_get_vev_honours_a_matrix_less_operator():
    """It used to return exactly the numbers it returns with no operator at
    all: the per-site projector composed with the operator kept only the
    projector's matrix, so the operator was dropped. It is now applied
    inside the sum over kpoints (see vev.kresolved_orbital_vev)."""
    g = geometry.chain()
    h = g.get_supercell(4).get_hamiltonian(has_spin=False)
    bare = h.get_vev(nk=4)
    zero = h.get_vev(operator=matrix_less(lambda v, k=None: 0.0 * v), nk=4)
    assert not np.allclose(zero, bare)   # the operator is not ignored
    assert np.allclose(zero, 0.0)        # and it annihilates everything


def test_kpm_kdos_refuses_a_k_dependent_operator():
    """It sampled the operator as a matrix, got None, and produced an
    unweighted KDOS without saying so. The unfolding operator is no longer
    an example, since the KPM kdos takes it exactly from its factor (see
    tests/unfolding/test_unfolding_kpm.py), so the operator here is a
    k-dependent one with neither a matrix nor a factor"""
    from pyqula.operators import Operator
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    op = Operator(lambda v, k=None: np.cos(2. * np.pi * k[0]) * v)
    with pytest.raises(NotImplementedError):
        h.get_kdos_bands(operator=op, mode="KPM", nk=2,
                         energies=np.linspace(-1., 1., 3))


def test_surface_spectral_function_refuses_a_k_dependent_operator():
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    from pyqula.kdos import get_surface_operator
    with pytest.raises(NotImplementedError):
        get_surface_operator(h, "unfold")
