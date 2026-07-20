import numpy as np

from pyqula import geometry
from pyqula.specialhamiltonian import H2HFH


def test_h2hfh_builds_a_heavy_fermion_hamiltonian():
    """Regression check for the full path that motivated the densebmat
    fix: H2HFH fuses a dispersive-electron Hamiltonian with a flat
    localized band via htk.fusion.hamiltonian_fusion ->
    algebra.direct_sum, which used to crash inside densebmat's leftover
    debug print loop on the None off-diagonal blocks."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(1.0)
    h = H2HFH(h, JK=0.2)
    (k, e, c) = h.get_bands(operator="dispersive_electrons", nk=10)
    assert len(e) > 0
    assert np.all(np.isfinite(e))
