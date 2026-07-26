import numpy as np
import pytest

from pyqula import islands
from pyqula.chitk.chiAB import chiAB


def _test_island():
    """A small finite (0d) honeycomb island -- fast to diagonalize, no SCF
    needed since this test only exercises the generic RPA-kernel pole
    finder with a hand-picked interaction matrix V."""
    g = islands.get_geometry(name="honeycomb", n=1.2, nedges=3)  # 6 sites
    return g.get_hamiltonian()


def _critical_V(h, energies, nk=1):
    """Pick a scalar-times-identity V just barely below the threshold at
    which a (charge-channel) RPA kernel eigenvalue 1-V*chi(0,omega=0)
    touches zero: since chi0(omega)'s magnitude falls off away from
    omega=0, sitting just below threshold makes the kernel eigenvalue dip
    through zero and back on both sides of omega=0, giving two clean,
    nearby poles to exercise the finder against (this is a purely
    numerical construction to exercise the pole finder, not a physical
    Hubbard-U channel; chi0's static eigenvalues are negative here, so
    V/U_c come out negative too)."""
    N = len(h.geometry.r)
    es, chis = chiAB(h, mode="matrix", energies=np.array([0.0]), delta=1e-2, nk=nk)
    chi0_eig = np.min(np.linalg.eigvals(chis[0]).real)  # most negative eigenvalue
    U_c = 1.0 / chi0_eig  # 1 - U_c*chi0_eig == 0
    return 0.9999 * U_c * np.eye(N)  # just below threshold


def test_rpa_kernel_poles_matches_bruteforce_determinant():
    """Every pole reported by rpa_kernel_poles must sit at a genuine local
    minimum of |det(kernel(omega))|, verified independently on a much finer
    local frequency grid that the pole finder itself never saw."""
    h = _test_island()
    N = len(h.geometry.r)
    energies = np.linspace(-0.6, 0.6, 121)
    delta = 1e-2
    V = _critical_V(h, energies)

    poles = h.get_rpa_kernel_poles(V=V, energies=energies, delta=delta, nk=1)
    assert len(poles) > 0, "expected at least one pole for a super-critical V"

    iden = np.identity(N, dtype=np.complex128)
    for w0, gamma0 in poles:
        fine = np.linspace(w0 - 0.02, w0 + 0.02, 81)
        es, chis = chiAB(h, mode="matrix", energies=fine, delta=delta, nk=1)
        dets = np.array([np.linalg.det(iden - V @ chi) for chi in chis])
        i0 = np.argmin(np.abs(dets))
        # the finer, independently-computed grid must find its minimum
        # within one coarse grid step of the reported pole
        assert abs(fine[i0] - w0) < 0.01
        assert np.abs(dets[i0]) < 1e-3
        # the reported residual imaginary part must match the fine-grid one
        assert abs(gamma0 - dets[i0].imag) < 0.05 or abs(gamma0) < 0.05


def test_rpa_kernel_poles_requires_interaction():
    h = _test_island()
    energies = np.linspace(-0.5, 0.5, 51)
    with pytest.raises(ValueError):
        h.get_rpa_kernel_poles(V=None, energies=energies, delta=1e-2, nk=1)


def test_rpa_kernel_poles_none_for_weak_interaction():
    """A tiny interaction can never push a kernel eigenvalue through zero
    within a bounded frequency window, so no poles should be reported."""
    h = _test_island()
    N = len(h.geometry.r)
    energies = np.linspace(-0.6, 0.6, 121)
    V = 1e-4 * np.eye(N)
    poles = h.get_rpa_kernel_poles(V=V, energies=energies, delta=1e-2, nk=1)
    assert len(poles) == 0
