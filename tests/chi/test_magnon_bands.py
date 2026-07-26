import numpy as np
import pytest

from pyqula import islands
from pyqula.chitk.chiAB import chiAB
from pyqula.chitk.spinchi import _full_spin_U
from pyqula.chitk.rpa import rpa_kernel_poles_ops
from testutils import SCF_MAXERROR

ENERGIES = np.linspace(-0.3, 0.3, 61)


def _converged_island():
    """A small zigzag-edged honeycomb island, converged from an explicit
    initial guess, following the same recipe as
    tests/chi/test_spinchi_rotation.py (see that file for why this specific
    filling/seed combination is needed to get a genuinely magnetized
    solution rather than the trivial paramagnetic one)."""
    g = islands.get_geometry(name="honeycomb", n=1.2, nedges=3)  # 6 sites
    h = g.get_hamiltonian()
    v = np.random.random(3) - .5
    v = v / np.linalg.norm(v)
    h.add_exchange(1e-2*v)
    mf = h.copy()
    mf.add_exchange(0.5*v)
    hmf = h.get_mean_field_hamiltonian(U=3.0, filling=0.3, mf=mf,
                                        maxerror=SCF_MAXERROR)
    return hmf


@pytest.mark.slow
def test_magnon_bands_shape_and_requires_interaction():
    h = _converged_island()
    qs, ws, gammas = h.get_magnon_bands(nq=2, energies=ENERGIES, delta=2e-2, nk=1)
    assert qs.shape == ws.shape == gammas.shape
    assert set(np.unique(qs)) <= {0, 1}  # nq=2 q-points, indexed 0..nq-1

    h0 = islands.get_geometry(name="honeycomb", n=1.2, nedges=3).get_hamiltonian()
    with pytest.raises(ValueError):
        h0.get_magnon_bands(nq=2, energies=ENERGIES, delta=2e-2, nk=1)


@pytest.mark.slow
def test_magnon_bands_matches_direct_kernel_poles():
    """get_magnon_bands must agree exactly with calling the lower-level
    rpa_kernel_poles_ops directly at q=0, using the same Sx,Sy,Sz operators
    and interaction matrix as spinchi_full/get_iets_ldos -- this checks the
    Hamiltonian.get_magnon_bands -> chi.magnon_bands wiring and the q-path
    loop don't diverge from the underlying pole finder."""
    h = _converged_island()
    qs, ws, gammas = h.get_magnon_bands(nq=1, energies=ENERGIES, delta=2e-2, nk=1)

    from pyqula.chitk.spinchi import _full_spin_operators
    Ss = _full_spin_operators(h)
    U = _full_spin_U(h)
    q0 = h.geometry.get_kpath(None, nk=1)[0]
    direct_poles = rpa_kernel_poles_ops(h, ops=Ss, V=U, q=q0,
                                         energies=ENERGIES, delta=2e-2, nk=1)

    assert np.all(qs == 0)
    assert len(ws) == len(direct_poles)
    order = np.argsort(ws)
    direct_order = np.argsort(direct_poles[:, 0])
    assert np.allclose(ws[order], direct_poles[direct_order, 0])
    assert np.allclose(gammas[order], direct_poles[direct_order, 1])
