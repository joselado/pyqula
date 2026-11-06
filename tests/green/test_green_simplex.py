# The bulk Green's function of the unit cell from the triangle integration
# (dostk/simplexdos.green_triangle), checked against the decimation plus
# adaptive k-integral of bloch_selfenergy(mode="adaptive"), an independent
# code path, and against the triangle DOS, which ties the two together.
import numpy as np

from pyqula import geometry
from pyqula.dostk import simplexdos
from pyqula.greentk.selfenergy import bloch_selfenergy


def _gapped_haldane():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.2)
    h.add_haldane(0.05)
    return h


def test_triangle_green_agrees_with_adaptive():
    """The full matrix, off-diagonal elements included, in the gap, in the
    band and near the van Hove energy of the M point"""
    h = _gapped_haldane()
    es = [0.1,0.5,1.0,2.0]
    gt = simplexdos.green_triangle(h,es,nk=201,delta=1e-2)
    for e,g in zip(es,gt):
        ga = bloch_selfenergy(h,energy=e,delta=1e-2,mode="adaptive",
                error=1e-6)[0]
        assert np.max(np.abs(g-ga)) < 2e-3*np.max(np.abs(ga))


def test_trace_of_the_green_function_is_the_dos():
    """-Im Tr G/pi is the DOS, the weights being the band projectors in one
    case and one in the other; the multiplet-averaged projectors have
    trace one, so the two agree to rounding on the same mesh"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_rashba(0.2)
    es = np.linspace(-3.,3.,13)+0.0037
    g = simplexdos.green_triangle(h,es,nk=31,delta=1e-2)
    _,d = h.get_dos(integration="triangle",energies=es,nk=31,delta=1e-2,
            write=False)
    trace = -np.trace(g,axis1=1,axis2=2).imag/np.pi
    assert np.max(np.abs(trace-d)) < 1e-10
