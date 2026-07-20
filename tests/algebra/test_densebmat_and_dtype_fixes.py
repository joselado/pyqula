import numpy as np

from pyqula import algebra
from pyqula import geometry


def test_direct_sum_handles_none_off_diagonal_blocks():
    """Regression check for algebra.densebmat: it had a leftover debug
    loop (`for mij in m: for mi in mij: print(mi.shape)`) that crashed
    with AttributeError on the None off-diagonal blocks that
    algebra.direct_sum legitimately builds (a block-diagonal direct sum
    has no off-diagonal blocks). The debug loop was dead code -- todense
    already handles None -- and was removed."""
    m1 = np.eye(2, dtype=np.complex128)
    m2 = 2 * np.eye(3, dtype=np.complex128)
    out = algebra.direct_sum([m1, m2])
    assert out.shape == (5, 5)
    assert np.allclose(out[0:2, 0:2], m1)
    assert np.allclose(out[2:5, 2:5], m2)
    assert np.allclose(out[0:2, 2:5], 0.0)  # off-diagonal block is zero


def test_project_interactions_uses_valid_complex_dtype():
    """Regression check for interactions.vijkl.Vijkl: it used the removed
    np.complex alias (np.zeros(..., dtype=np.complex)), which raises
    AttributeError on any recent numpy. Now uses np.complex128."""
    from pyqula import islands
    g = islands.get_geometry(name="triangular", n=2, nedges=20, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.set_filling(0.5)
    m = h.project_interactions(n=4)
    assert m.shape == (4, 4, 4, 4)
    assert np.all(np.isfinite(m))
