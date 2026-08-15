"""h.get_hk_gen() must accept an integer k-point.

`h.get_hk_gen()([0,0,0])` is the natural way to write the Gamma point, and it
used to fail with an opaque numba TypingError: htk/bloch.py handed `k` to a
jitted kernel compiled for float64, so an all-integer k arrived as int64. The
neighbouring `ds` array was already coerced explicitly; `k` was not.

This is worth a test out of proportion to its size because get_hk_gen is the
single most central method in the library -- bands, DOS, topology, transport
and the mean-field loops all go through it -- and the failure mode was a
compiler error with no connection to the user's input, on the most obvious
possible argument.

It also went unnoticed because the whole suite happens to build k-points from
np.linspace or literal floats. Nothing was wrong with the physics; the input
type was simply never varied.
"""
import numpy as np
import pytest

from pyqula import geometry


def _h():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_haldane(0.05)
    return h


@pytest.mark.parametrize("k", [
    [0, 0, 0],                       # python ints -- the Gamma point
    np.array([0, 0, 0]),             # int64 numpy array
    [1, 0, 0],                       # nonzero ints
    np.array([0, 1, 0], dtype=int),
])
def test_get_hk_gen_accepts_integer_kpoints(k):
    """Must not raise, and must agree exactly with the float spelling."""
    h = _h()
    f = h.get_hk_gen()
    got = f(k)
    ref = f([float(x) for x in np.asarray(k)])
    assert np.array_equal(got, ref), k


def test_get_gk_gen_accepts_integer_kpoints():
    """The Green's-function generator wraps get_hk_gen, so it inherited the
    same failure."""
    h = _h()
    g = h.get_gk_gen()
    got = g(e=0.1, k=[0, 0, 0])
    ref = g(e=0.1, k=[0., 0., 0.])
    assert np.array_equal(got, ref)


def test_integer_k_does_not_perturb_float_results():
    """Guard against a 'fix' that coerces in a lossy way: the float path must
    be untouched, bit for bit."""
    h = _h()
    f = h.get_hk_gen()
    ks = [[0.13, 0.27, 0.], [0.5, 0.5, 0.], [1./3., 1./3., 0.]]
    first = [f(k) for k in ks]
    second = [f(k) for k in ks]
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
