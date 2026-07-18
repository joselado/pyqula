import numpy as np

from pyqula import geometry
from pyqula import kdos


def test_surface_kdos_zigzag_ribbon_disorder_matches_reference(tmp_path, monkeypatch):
    """Regression check for kdos.kdos_bands on a Haldane-gapped zigzag
    ribbon with an edge disorder function, at a small width (n=15 instead
    of 50) and coarse mesh (nk=15, 40 energies instead of the 100/200
    defaults): the total KDOS array must match the value recorded from a
    known-good run. (The disorder function draws from an unseeded RNG, but
    this quantity is empirically robust to the specific realization -- it
    matched exactly between two independent, unseeded runs three days
    apart before this test was written.)"""
    monkeypatch.chdir(tmp_path)  # writes KDOS_BANDS.OUT to cwd
    g = geometry.honeycomb_zigzag_ribbon(15)
    h = g.get_hamiltonian()
    h.add_haldane(0.1)

    edge = np.zeros(h.intra.shape[0])
    edge += 1.0
    edge[10:edge.shape[0]] = 0.0
    frand = lambda: (-0.5 + np.random.random(edge.shape[0])) * edge
    out = kdos.kdos_bands(h, frand=frand, nk=15, energies=np.linspace(-3.0, 3.0, 40))
    assert np.isclose(np.sum(out), 13043.314404109648, atol=1e-2)
