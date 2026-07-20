import numpy as np

from pyqula import phasediagram


def test_selected_interpolation_runs_without_removed_scipy_function(tmp_path, monkeypatch):
    """Regression check for phasediagram.selected_interpolation: it used
    scipy.interpolate.interp2d, removed in SciPy >=1.14, so any call with
    nite>=1 (the default) raised. It was replaced with
    RegularGridInterpolator; check the refined grid is finite and the
    same shape as before."""
    monkeypatch.chdir(tmp_path)  # diagram2d writes PHASE_DIAGRAM.OUT to cwd
    def getquantity(x, y):
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    phasediagram.diagram2d(getquantity, x=np.linspace(0., 1., 6),
            y=np.linspace(0., 1., 6), nite=2)
    d = np.genfromtxt("PHASE_DIAGRAM.OUT")
    assert d.shape[0] == 12 * 12  # grid doubled once (nite=2 refines, then stops at nite=1)
    assert np.all(np.isfinite(d[:, 2]))
