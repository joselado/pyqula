import numpy as np

from pyqula import geometry


def test_diamond_lattice_positions_are_real():
    """Regression check for sculpt.set_xy_plane: algebra.inv always
    returns complex128 (it forces that dtype for every dense input, real
    or not), and set_xy_plane's rotation used that inverse directly
    without casting back to real. That leaked complex128 into a purely
    real geometric transform of real vectors, and the resulting complex
    positions/lattice vectors later crashed numba's real-typed
    close_enough (used when building a multicell Hamiltonian) with
    "lt(complex128, float64)"."""
    g = geometry.diamond_lattice()
    assert not np.iscomplexobj(g.r)
    assert not np.iscomplexobj(g.a1)
    assert not np.iscomplexobj(g.a2)
    assert not np.iscomplexobj(g.a3)
    # this used to raise a numba TypingError via multicell.close_enough
    h = g.get_hamiltonian()
    assert h.intra.shape[0] == 2 * len(g.r)


def test_cubic_lattice_positions_are_an_array():
    """Regression check for geometry.cubic_lattice: g.r was a plain
    Python list containing one array, not an (N,3) numpy array like every
    other geometry factory produces. geometrytk.write.write_positions
    does g.r[:,0]-style fancy indexing, which raised "list indices must
    be integers or slices, not tuple"."""
    g = geometry.cubic_lattice()
    assert isinstance(g.r, np.ndarray)
    assert g.r.shape == (1, 3)
    assert g.r[:, 0][0] == 0.0  # the fancy indexing that used to crash
