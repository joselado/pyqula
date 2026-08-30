"""The sign of a spin expectation value, component by component.

spectrum.ev contracts the density matrix that densitymatrix.full_dm
returns, and that one is built as dm[i,j] = sum_occ conj(psi_i) psi_j --
the transpose of the usual rho[i,j] = sum_occ psi_i conj(psi_j). Tracing
dm@A therefore evaluates <A^T> = <A*>, which equals <A> for every real
operator (the density, sx, sz, a projector) and flips the sign for a
purely imaginary one. sy is exactly that, so <sy> came back negated while
<sx> and <sz> were right -- invisible along a single axis, and visible
only as a reflection of the moment vector in y for a generic direction.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.spin import sx, sy, sz


def _occupied_sum(h, op, nk=60):
    """<op> summed over the occupied states, straight from eigh."""
    hk = h.get_hk_gen()
    tot = 0.0
    for k in np.linspace(0., 1., nk, endpoint=False):
        es, ws = np.linalg.eigh(np.array(hk([k, 0., 0.])))
        for i, e in enumerate(es):
            if e < 0.:
                tot += np.vdot(ws[:, i], op@ws[:, i]).real
    return tot/nk


@pytest.mark.parametrize("axis,pauli", [(0, sx), (1, sy), (2, sz)])
def test_spin_vev_matches_an_explicit_occupied_state_sum(axis, pauli):
    """get_vev goes through the density matrix; the reference here does not."""
    v = [0., 0., 0.]
    v[axis] = 0.5
    h = geometry.chain().get_hamiltonian()
    h.add_exchange(v)
    ref = _occupied_sum(h, np.array(pauli.todense()))
    got = h.get_vev(["sx", "sy", "sz"][axis], nk=60)[0]
    assert abs(got - ref) < 1e-10, (got, ref)


@pytest.mark.parametrize("v", [[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5],
                               [0.3, 0.4, 0.5], [-0.2, 0.5, -0.1]])
def test_the_moment_is_antiparallel_to_the_exchange_field(v):
    """add_exchange writes +h.sigma, so the occupied states polarize
    against it: the moment must be exactly antiparallel to h, whatever
    direction h points in. A single flipped component leaves this true
    along the three axes and false for any generic direction."""
    h = geometry.chain().get_hamiltonian()
    h.add_exchange(v)
    m = h.get_magnetization(nk=60)[0]
    v = np.array(v, dtype=float)
    cos = m.dot(v)/np.linalg.norm(m)/np.linalg.norm(v)
    assert abs(cos + 1.) < 1e-8, (m, v, cos)


def test_moment_direction_agrees_with_the_independent_implementation():
    """magnetism.compute_magnetization reads the density-matrix elements
    directly instead of tracing against an operator, so it is an
    independent check on the sign of all three components (its
    normalization is <S>=<sigma>/2, hence the factor of two)."""
    from pyqula.magnetism import compute_magnetization
    h = geometry.chain().get_hamiltonian()
    h.add_exchange([0.3, 0.4, 0.5])
    mx, my, mz = compute_magnetization(h, nk=60)
    other = 2.*np.array([mx[0], my[0], mz[0]]).real
    assert np.allclose(h.get_magnetization(nk=60)[0], other, atol=1e-8)
