import numpy as np

from pyqula import geometry
from pyqula import topology


def test_omega_rmap_runs_and_writes_output(tmp_path, monkeypatch):
    """Regression check for topology.Omega_rmap (the function
    examples used to call under its old, since-removed name
    berry_green_map): it must run to completion and write a spatially
    resolved Berry-density profile to BERRY_RMAP.OUT."""
    monkeypatch.chdir(tmp_path)  # writes BERRY_RMAP.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    out = topology.Omega_rmap(h, k=[0., 0., 0.], nrep=1, integral=False)
    assert out is not None
    d = np.genfromtxt("BERRY_RMAP.OUT")
    assert d.shape[0] > 0


def test_chern_density_runs_and_integrates(tmp_path, monkeypatch):
    """Regression check for topology.chern_density: it used to call the
    since-removed scipy.integrate.cumtrapz, which raised an ImportError
    on any recent SciPy. Also a smoke check that the energy-resolved
    Berry density integrates to something finite."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.1)
    es, cs, csi = topology.chern_density(h, nk=4, es=np.linspace(-1.0, 1.0, 10))
    assert len(es) == len(cs) == len(csi)
    assert np.all(np.isfinite(csi))
