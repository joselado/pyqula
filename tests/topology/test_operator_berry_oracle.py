"""External-oracle checks for topology.operator_berry / topologytk.operatorberry.

Everything that previously exercised this path asserted against constants
recorded from the code's own earlier run (tests/topology/test_berry_valley.py,
test_berry_valley_spin.py, test_quantum_geometry.py,
test_berry_curvature_disentangle_strained.py). That kind of test pins whatever
the code did when the number was recorded, so it could not -- and did not --
detect that operator_berry was returning garbage. Two independent bugs were
found here by checking against something outside the code instead:

1. `multicell.derivative` (current.py) returns np.matrix whenever h.intra is
   one (add_haldane routes through a scipy sparse build, so any Haldane/
   Kane-Mele Hamiltonian hits this). `*` on np.matrix is MATRIX
   multiplication, so operatorberry's `prod = h1.T*h2` -- documented and
   intended as an elementwise product -- silently computed a matrix product.
2. The Kubo formula's overall minus sign was missing, so even with (1) fixed
   the BZ integral came out as -2*pi*Chern.

The oracle here is the analytically known Chern number of the Haldane model,
cross-checked against topology.chern's independent Fukui-Hatsugai-Suzuki
Wilson-loop implementation. Assert on BOTH sign and magnitude: bug (2) was
invisible to any magnitude-only check.

SIGN CONVENTION. The signs asserted below are pyqula's, not Xiao/Chang/Niu's
(RMP 82, 1959 (2010)). topology.berry_curvature returns -Omega in the RMP
convention and every Chern number in the package inherits that global sign --
see its SIGN CONVENTION docstring. So "C = +1 for t2 = +0.1" here means +1 as
pyqula reports it; the point of these tests is that operator_berry agrees with
its siblings (h.get_chern, topology.spin_chern, bandstructure.berry_bands),
which is what was broken. If someone ever flips the package to the RMP
convention, these expected values flip with it -- they are not independent of
that choice, and should not be used to argue operator_berry's sign in
isolation.
"""
import numpy as np
import pytest

from pyqula import geometry, topology, parallel
from testutils import temporary_attr


def _haldane(t2=0.1, mass=0.0):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    if mass != 0.0:
        h.add_sublattice_imbalance(mass)
    return h


def _chern_from_operator_berry(h, nk=30, operator=None):
    """Chern number as the BZ average of operator_berry over the fractional
    unit cell, divided by 2*pi -- operator_berry's own documented
    normalization ("normalize so the sum is 2pi Chern")."""
    vals = [topology.operator_berry(h, k=[(i + 0.5) / nk, (j + 0.5) / nk],
                                    operator=operator)
            for i in range(nk) for j in range(nk)]
    return np.mean(vals) / (2.0 * np.pi)


@pytest.mark.parametrize("t2,mass,expected", [
    (0.1, 0.0, +1),    # topological
    (0.1, 0.2, +1),    # still topological, gapped away from the Dirac points
    (-0.1, 0.0, -1),   # reversed Haldane flux -> opposite Chern
    (0.1, 1.5, 0),     # mass beyond 3*sqrt(3)*t2 -> trivial
])
def test_operator_berry_integrates_to_the_analytic_chern_number(t2, mass, expected):
    """Sign AND magnitude, against the analytic Chern number."""
    h = _haldane(t2=t2, mass=mass)
    with temporary_attr(topology.parallel, "cores", 1):
        c = _chern_from_operator_berry(h, nk=30)
    assert np.isclose(c, expected, atol=2e-2), (t2, mass, c, expected)


def test_operator_berry_matches_wilson_loop_chern():
    """Agreement with topology.chern's independent Wilson-loop path -- two
    unrelated algorithms for the same invariant."""
    h = _haldane(t2=0.1, mass=0.2)
    with temporary_attr(topology.parallel, "cores", 1):
        c_op = _chern_from_operator_berry(h, nk=30)
        c_wilson = topology.chern(h, nk=14)
    assert np.isclose(c_op, c_wilson, atol=2e-2), (c_op, c_wilson)


def test_operator_berry_identity_operator_matches_operator_none():
    """Passing an explicit identity must reduce exactly to operator=None --
    this is what makes the operator-weighted path's symmetrization
    (operator@dhdx + dhdx@operator)/2 meaningful as a generalization."""
    h = _haldane(t2=0.1, mass=0.2)
    k = [0.17, 0.41]
    n = h.intra.shape[0]
    ident = np.identity(n, dtype=np.complex128)
    with temporary_attr(topology.parallel, "cores", 1):
        b_none = topology.operator_berry(h, k=k, operator=None)
        b_ident = topology.operator_berry(h, k=k, operator=ident)
    assert np.isclose(b_none, b_ident, rtol=1e-10, atol=1e-12)


def test_operator_berry_bands_sum_to_operator_berry_over_occupied():
    """operator_berry_bands must be the band resolution of operator_berry:
    summing its occupied (E<=0) entries reproduces the total. Guards the two
    functions against drifting apart -- they share _berry_curvature_bands but
    apply their normalization at separate call sites in topology.py."""
    h = _haldane(t2=0.1, mass=0.2)
    k = [0.23, 0.37]
    with temporary_attr(topology.parallel, "cores", 1):
        total = topology.operator_berry(h, k=k)
        (es, bs) = topology.operator_berry_bands(h, k=k)
    assert np.isclose(np.sum(np.array(bs)[np.array(es) <= 0.]), total,
                      rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("soc,mass,expected", [
    (0.1, 0.0, 2.0),   # QSH: C_up=+1, C_down=-1 -> sz-weighted sum = 2
    (0.1, 1.5, 0.0),   # sublattice mass beyond the gap closing -> trivial
])
def test_spin_chern_is_quantized_on_kane_mele(soc, mass, expected):
    """topology.spin_chern is operator_berry with the sz operator, and is the
    one *public invariant* the operator-weighted path feeds. Its value must be
    quantized: the sz-weighted Berry curvature integrates to C_up - C_down.

    This also covers the Operator-object branch -- spin_chern passes
    operators.get_sz(h), not a raw matrix, so operatorberry must not coerce
    `operator` itself (only its product with dhdx)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(soc)
    if mass != 0.0:
        h.add_sublattice_imbalance(mass)
    with temporary_attr(topology.parallel, "cores", 1):
        c = topology.spin_chern(h, nk=24)
    assert np.isclose(c, expected, atol=5e-2), (soc, mass, c, expected)


def test_operator_berry_returns_plain_float_not_matrix():
    """Regression guard for the np.matrix bug: multicell.derivative returns
    np.matrix for a Haldane Hamiltonian, and operatorberry must coerce before
    its elementwise product. A np.matrix leaking out here is the signature of
    that coercion having been removed."""
    h = _haldane(t2=0.1, mass=0.2)
    with temporary_attr(topology.parallel, "cores", 1):
        b = topology.operator_berry(h, k=[0.17, 0.41])
    assert not isinstance(b, np.matrix)
    assert np.ndim(b) == 0, np.shape(b)
