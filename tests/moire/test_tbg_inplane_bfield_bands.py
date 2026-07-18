import numpy as np

from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix


def test_tbg_inplane_bfield_bands_match_reference(tmp_path, monkeypatch):
    """Regression check for twisted bilayer graphene with an in-plane
    magnetic field and interlayer bias, at the smallest commensurate moire
    index (n=1): the band energies and z-position expectation values must
    match the values recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    g = specialgeometry.twisted_bilayer(1)
    ti = 0.0
    h = g.get_hamiltonian(is_sparse=True, has_spin=False,
                           mgenerator=twisted_matrix(ti=ti))
    b = 0.02
    bphi = 0.828
    bias = 0.0
    h.add_inplane_bfield(b=b, phi=bphi)
    h.add_onsite(lambda r: np.sign(r[2]) * bias)
    (k, e, c) = h.get_bands(num_bands=20, operator="zposition")
    assert np.isclose(np.sum(e), 1.6288937685633536, atol=1e-6)
    assert np.isclose(np.sum(c), 1.1368683772161603e-13, atol=1e-6)
