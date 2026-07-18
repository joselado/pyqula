import numpy as np
import scipy.linalg as lg

from pyqula import geometry
from pyqula import green


def get_bulk_green_function(h0, energy=0.0, eta=1e-3):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.intra = np.matrix(h0.intra)
    h.tx = np.matrix(h0.tx)
    h.ty = np.matrix(h0.ty)
    h.txy = np.matrix(h0.txy)
    h.txmy = np.matrix(h0.txmy)
    gf, selfe = green.bloch_selfenergy(h, energy=energy, delta=eta, mode="adaptive")
    return gf


def test_custom_embedding_dos_matches_reference(tmp_path, monkeypatch):
    """Regression check for a custom-Hamiltonian embedding DOS built via
    green.bloch_selfenergy in adaptive mode, at a coarse energy mesh (10
    points instead of 40): the summed DOS must match the value recorded
    from a known-good run."""
    monkeypatch.chdir(tmp_path)

    class H(): pass
    h = H()
    t = np.array([[0., 1.], [0., 0.]])
    h.intra = t + t.T
    h.tx = t
    h.ty = t
    h.txy = 0. * t
    h.txmy = 0. * t
    eta = 1e-2

    def f(e):
        gf = get_bulk_green_function(h, energy=e, eta=eta)
        selfe = t @ gf @ t.T
        ons = np.array([[10000.0, 0.], [0., 0.]])
        gg = lg.inv(ons - np.array([[1., 0.], [0., 1.]]) * (e - 1j * 1e-2) - selfe)
        return -np.trace(gg).imag

    es = np.linspace(-2., 2., 10)
    ds = [f(e) for e in es]
    assert np.isclose(np.sum(ds), 0.47863499138998467, atol=1e-4)
