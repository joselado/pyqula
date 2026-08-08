import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk.spinchi import _full_spin_U, _full_spin_operators
from pyqula.chitk.rpa import rpa_kernel_ops


def _q0_kernel_residual(hmf, delta, nk):
    """Smallest |eigenvalue| of the full spin RPA kernel 1-U(q)*chi0(q,w)
    at q=0, w=0. Exactly gapless (Goldstone) means this -> 0 as delta -> 0;
    a genuine gap would instead plateau at a delta-independent value."""
    Ss = _full_spin_operators(hmf)
    Uv = _full_spin_U(hmf)
    q0 = [0., 0., 0.]
    _, kernels = rpa_kernel_ops(hmf, ops=Ss, V=Uv, q=q0,
                                 energies=np.array([0.0]), delta=delta, nk=nk)
    return np.min(np.abs(np.linalg.eigvals(kernels[0])))


@pytest.mark.slow
@pytest.mark.parametrize("U,filling", [(10.0, 0.2), (6.0, 0.25)])
def test_doped_ferro_chain_magnon_is_gapless_at_q0(U, filling):
    """A doped (filling != 0.5), fully spin-polarized (saturated) plain-
    onsite-Hubbard ferromagnetic chain must have an exactly gapless
    (Goldstone) transverse magnon at q=0: the RPA kernel's q=0,w=0 residual
    is a pure finite-broadening artifact of delta (the Lorentzian
    regularization chiAB uses), not a real energy gap, so residual/delta
    must stay roughly CONSTANT as delta shrinks -- that delta-linearity is
    the actual Goldstone signature (a real gap would instead leave the
    residual itself roughly delta-independent, so residual/delta would grow
    as delta shrinks).

    This is the RPA/Dyson resummation (chi0 -> chi0(1-U*chi0)^-1) doing its
    job: the BARE (RPA=False) susceptibility chi0 alone has essentially no
    weight at w=0 -- its spectral weight sits near the Stoner spin-flip
    gap set by the mean-field exchange splitting -- and only after
    dressing by the interaction does the exact w=0 pole reappear (see
    spinchi_ladder's RPA=False vs RPA=True below). Confirms
    chitk.rpa.chi_AB_RPA/chi_ops_RPA correctly "renormalize" the bare
    bubble with the SAME U that produced the mean-field splitting, which
    is what the Goldstone/Ward identity requires.

    Deliberately restricted to SATURATED (half-metallic, one spin channel
    fully empty) configurations: a partially-polarized itinerant
    ferromagnet on this plain nearest-neighbor chain sits close enough to
    the Stoner threshold that the SCF fixed point itself does not converge
    with the k-mesh (Delta drifts by ~10% between nk=1600 and nk=3200 in
    exploratory checks) -- a k-mesh-fragility issue with the mean-field
    solver on this particular 1D model, not something this test (which
    targets the RPA kernel, not the SCF) can responsibly probe. Partial
    polarization is therefore untested here."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    hmf = h.get_mean_field_hamiltonian(U=U, filling=filling, mf="ferro", nk=300)
    mz = hmf.get_vev("sz", nk=300)[0]
    assert abs(mz) > 0.1, f"expected a sizable ferromagnetic moment, got {mz}"

    r1 = _q0_kernel_residual(hmf, delta=0.02, nk=300)
    r2 = _q0_kernel_residual(hmf, delta=0.005, nk=300)
    ratio1, ratio2 = r1 / 0.02, r2 / 0.005
    # both ratios bounded (residual doesn't blow up) and mutually consistent
    # (delta-linear, not e.g. delta-independent -- the signature of a real gap)
    assert ratio1 < 1.0 and ratio2 < 1.0, \
        f"residual/delta not bounded: {ratio1} (delta=0.02), {ratio2} (delta=0.005)"
    assert abs(ratio2 - ratio1) < 0.2 * ratio1, \
        f"residual/delta not constant across delta ({ratio1} -> {ratio2}); " \
        "looks like a real gap, not a pure broadening artifact"


@pytest.mark.slow
def test_bare_susceptibility_lacks_the_q0_pole_rpa_restores():
    """The RPA dressing is load-bearing for the q=0 Goldstone pole, not
    incidental: the bare (non-interacting) S+/- response of the SAME
    spin-split mean-field bands has essentially no spectral weight at
    w=0 (it sits near the Stoner gap instead), while chi_RPA develops a
    sharp peak exactly at w=0. This is the numerical signature of "not
    renormalizing the interaction" gapping the mode."""
    from pyqula.chitk.spinchi import spinchi_ladder
    g = geometry.chain()
    h = g.get_hamiltonian()
    hmf = h.get_mean_field_hamiltonian(U=10.0, filling=0.2, mf="ferro", nk=300)

    energies = np.linspace(-1.0, 1.0, 81)
    delta = 0.05
    _, chi_bare = spinchi_ladder(hmf, RPA=False, q=[0., 0., 0.],
                                  energies=energies, delta=delta, nk=300)
    _, chi_rpa = spinchi_ladder(hmf, RPA=True, q=[0., 0., 0.],
                                 energies=energies, delta=delta, nk=300)
    im_bare = np.array([np.trace(c).imag for c in chi_bare])
    im_rpa = np.array([np.trace(c).imag for c in chi_rpa])
    i0 = len(energies) // 2  # index closest to w=0

    assert abs(np.argmax(np.abs(im_rpa)) - i0) <= 1, \
        "RPA-dressed response should peak at (or immediately next to) w=0 " \
        "(the Goldstone pole)"
    assert np.abs(im_rpa[i0]) > 50 * np.abs(im_bare[i0]), \
        "RPA dressing should massively enhance the w=0 weight relative " \
        "to the bare susceptibility"
