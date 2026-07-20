import numpy as np

from pyqula.geometrytk.bloch import bloch_phase


class _FakeGeometry:
    def __init__(self, dimensionality):
        self.dimensionality = dimensionality


def test_bloch_phase_2d_accepts_scalar_k():
    """Regression check: bloch_phase's 1D branch already tolerated a bare
    scalar k ("ups, assume that it is a float"), but the 2D and 3D
    branches did np.array(k)[0:2]/[0:3] unconditionally, which raised
    IndexError on a 0-dimensional array. A scalar k is now treated as the
    first component with the rest zero, matching the 1D fallback."""
    g = _FakeGeometry(2)
    scalar_phase = bloch_phase(g, [1, 0, 0], 0.25)
    vector_phase = bloch_phase(g, [1, 0, 0], [0.25, 0.])
    assert np.isclose(scalar_phase, vector_phase)


def test_bloch_phase_3d_accepts_scalar_k():
    g = _FakeGeometry(3)
    scalar_phase = bloch_phase(g, [1, 0, 0], 0.25)
    vector_phase = bloch_phase(g, [1, 0, 0], [0.25, 0., 0.])
    assert np.isclose(scalar_phase, vector_phase)
