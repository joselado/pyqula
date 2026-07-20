import numpy as np

from pyqula import geometry
from pyqula import heterostructures


def test_get_kdos_accepts_scalar_kpoints(tmp_path, monkeypatch):
    """Regression check for transporttk.kdos.kdos: it calls self.generate(k)
    with k coming straight from a plain np.linspace (bare scalars), which
    eventually reaches geometrytk.bloch.bloch_phase and did
    np.array(k)[0:2] -- IndexError on a 0-dimensional array. bloch_phase's
    2D and 3D branches must tolerate a scalar k, the same way the 1D
    branch already did."""
    monkeypatch.chdir(tmp_path)  # writes KDOS.OUT to cwd
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.remove_spin()
    h1 = h.copy()
    h2 = h.copy()
    h1.add_sublattice_imbalance(0.3)
    h2.add_sublattice_imbalance(-0.3)
    HT = heterostructures.build(h1, h2)
    HT.delta = 1e-1
    (k, e, d) = HT.get_kdos(delta=1e-1, kpath=np.linspace(0., 1., 4),
            energies=np.linspace(-1.0, 1.0, 4))
    assert len(k) == len(e) == len(d)
    assert np.all(np.isfinite(d))
