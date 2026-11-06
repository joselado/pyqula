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


def _diamond():
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    return h


def test_triangle_selfenergy_in_3d_converges_to_a_direct_k_sum():
    """bloch_selfenergy(mode="triangle") of a two-site 3d crystal, the full
    matrix, against a plain sum of (z - H_k)^-1 over a uniform mesh at a
    broadening that the uniform mesh resolves (converged there to 1e-4).
    The tetrahedron result converges at roughly second order in 1/nk, so
    the error has to drop by more than 2.5 from nk=16 to nk=32"""
    h = _diamond()
    delta = 0.3
    hk = h.get_hk_gen()
    kk = np.linspace(0.,1.,30,endpoint=False)
    ks = np.array([[a,b,c] for a in kk for b in kk for c in kk])
    hc = np.array(h.intra.todense()) if hasattr(h.intra,"todense") \
            else np.array(h.intra)
    for e in [-2.,0.2,1.5]: # in the band, in the gap, in the band
        z = (e+1j*delta)*np.identity(2)
        gref = np.mean([np.linalg.inv(z-hk(k)) for k in ks],axis=0)
        sc = np.max(np.abs(gref))
        err = []
        for nk in (16,32):
            g,s = bloch_selfenergy(h,energy=e,delta=delta,mode="triangle",
                    nk=nk)
            err.append(np.max(np.abs(g-gref))/sc)
            # the selfenergy closes the Dyson equation of the unit cell
            assert np.allclose(np.linalg.inv(z-hc-s),g)
        assert err[1] < 2e-2
        assert err[0]/err[1] > 2.5


def test_triangle_selfenergy_is_3d_only():
    import pytest
    h = _gapped_haldane()
    with pytest.raises(ValueError,match="only for 3d"):
        bloch_selfenergy(h,energy=0.1,mode="triangle")
    with pytest.raises(ValueError,match="accepted values"):
        bloch_selfenergy(h,energy=0.1,mode="trianlge")
