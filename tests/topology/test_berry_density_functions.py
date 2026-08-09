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


def test_chern_density_plateau_matches_wilson_chern_sign_and_scale(tmp_path, monkeypatch):
    """Regression check for topology.chern_density/topologytk.green.dOmega_dE:
    the energy-resolved Berry density was evaluated on the real energy axis
    with the opposite sign of the independently Wilson-loop-validated
    complex-contour method (topology.berry_green, used by
    write_berry(mode="Green") and cross-checked against Wilson loops in
    tests/topology and the wilson_green_formalism notebook) -- verified at
    a single k-point, where the real-axis integral of dOmega/dE reproduced
    -berry_green(...) to 5 significant figures before the sign fix. As a
    result, chern_density's cumulative, energy-integrated Berry density
    used to plateau at close to *minus* the Chern number inside the gap
    (e.g. -1.8 instead of +2.0 for this Haldane model), rather than
    matching it."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_haldane(0.2)
    C = h.get_chern(nk=20)
    assert np.isclose(C, 2.0, atol=1e-3)

    es, cs, csi = topology.chern_density(h, nk=6, delta=0.05, dk=0.05,
            es=np.linspace(-3.5, 0.3, 120))
    gap_idx = np.argmin(np.abs(es))  # es[0]=-3.5 clears the valence band
    # below, so csi should already have plateaued near the (positive) Chern
    # number by the time the scan reaches E~0, deep in the Haldane gap
    assert np.isclose(csi[gap_idx], C, atol=0.5)
