import numpy as np

from pyqula import geometry
from pyqula import specialhopping


def test_entry2matrix_falls_back_when_hopping_is_not_jittable():
    """Regression check for specialhopping.entry2matrix: jit() is lazy, so
    wrapping a non-jittable hopping function (e.g. one that closes over a
    Python callable, like phase_C3's use of a per-call `t`/`phi` closure)
    never actually raised at decoration time -- the try/except around
    jit(f, nopython=True) never triggered, and the real TypingError only
    surfaced later, uncaught, the first time the returned matrix function
    was called. entry2matrix must eagerly trigger compilation with a
    dummy call inside the try block so the fallback to pure Python
    actually engages."""
    g = geometry.triangular_lattice()
    g = g.supercell((2, 1))

    # a hopping function that closes over a Python callable -- numba
    # cannot type this in nopython mode, so it must hit the fallback
    scale = {"value": 1.0}
    def variable_scale(r):
        return scale["value"]

    def fun(r1, r2):
        dr = r1 - r2
        dr2 = dr.dot(dr)
        if 0.99 < dr2 < 1.01:
            return variable_scale(r1) * 1.0
        return 0.0

    fmat = specialhopping.entry2matrix(fun)
    rs = np.array(g.r)
    out = fmat(rs, rs)
    # compare against direct (unjitted) evaluation for correctness
    expected = np.array([[fun(r1, r2) for r1 in rs] for r2 in rs]).T
    assert np.allclose(out, expected)
