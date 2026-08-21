"""The transverse RPA with the interaction's pair index kept.

The site-basis RPA of chitk/rpa.py has one vertex number per site, which
is exact for an onsite Hubbard U and empty for anything longer ranged --
not approximate, empty: the extraction maps a spin-independent V_ij to
exactly zero. The reason is that the transverse ladder rung
K_{(ij),(kl)} = -V_ij delta_ik delta_jl is diagonal in the PAIR index,
which collapses onto sites only when V is onsite.

chitk/pairchi.py keeps that index. These tests pin the three things that
makes it worth having:

  - it reduces to the site-basis answer for an onsite U, so nothing that
    worked before is a different number now;
  - it carries a neighbour-shell density-density interaction, with an
    exact Goldstone mode where the site basis has no vertex at all;
  - it agrees with the independent pair-basis TDHF of bsetk/spinflip.py,
    and with a closed-form exact answer where one exists.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from pyqula import geometry
from pyqula.bsetk.interaction import bare_interaction
from pyqula.bsetk.spinflip import magnon_energies
from pyqula.chitk import pairchi
from pyqula.meanfield import VJinteraction

NK = 6


def _neel(nk=NK, **kw):
    g = geometry.honeycomb_lattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                          nk=nk, maxerror=1e-10, mix=0.3, maxite=3000,
                          **kw).hamiltonian


def _kernel_min_eigenvalue(h, nk=NK, delta=1e-4, q=(0., 0., 0.),
                            channel="+-"):
    _, K = pairchi.pair_rpa_kernel(h, q=list(q), energies=np.array([0.0]),
                                    delta=delta, nk=nk, channel=channel)
    return np.min(np.abs(np.linalg.eigvals(K[0])))


def test_the_pair_basis_is_the_support_of_the_interaction():
    """Not N^2: the kernel is diagonal in the pair index, so only pairs the
    interaction actually couples enter, plus the diagonal ones the physical
    response is read off. A honeycomb cell with a nearest-neighbour V has
    eight, not sixteen."""
    h = _neel(U=3.0)
    pairs, values, diag = pairchi.build_pairs(bare_interaction(h), 2)
    assert len(pairs) == 2 and np.allclose(values.real, [3., 3.])  # onsite U
    assert list(diag) == [0, 1]
    h2 = _neel(U=3.0, V1=0.5)
    pairs2, values2, _ = pairchi.build_pairs(bare_interaction(h2), 2)
    assert len(pairs2) == 8  # 2 onsite + 6 bond
    assert any(p[0] != p[1] for p in pairs2)  # genuinely off-diagonal pairs


@pytest.mark.slow
def test_goldstone_for_an_onsite_hubbard_antiferromagnet():
    """The case the site basis already did, which must not change. Both
    spin channels, and the controls: at finite q, and away from a magnetic
    state, the kernel is nowhere near singular."""
    h = _neel(U=3.0)
    for ch in ("+-", "-+"):
        assert _kernel_min_eigenvalue(h, channel=ch) < 1e-6
    assert _kernel_min_eigenvalue(h, q=(0.1, 0., 0.)) > 1e-2


@pytest.mark.slow
def test_goldstone_with_a_neighbour_shell_density_density_interaction():
    """The case the site basis cannot do at all. Its vertex for V1 is
    exactly zero; here V1 enters as six off-diagonal pairs and the
    Goldstone mode survives."""
    h = _neel(U=3.0, V1=0.5)
    assert len(h.V) > 1
    assert _kernel_min_eigenvalue(h) < 1e-6
    assert _kernel_min_eigenvalue(h, q=(0.1, 0., 0.)) > 1e-2


@pytest.mark.slow
def test_it_agrees_with_the_independent_tdhf_pair_basis():
    """Two implementations that share no code below the Hamiltonian: a
    frequency scan for zeros of 1 + V chi0 in the interaction's pair basis,
    and a Casida eigenproblem in the electron-hole pair basis. Measured
    agreement to 5 decimals, with and without V1."""
    grid = np.linspace(1e-4, 3.0, 3000)
    for kw in ({"U": 3.0}, {"U": 3.0, "V1": 0.5}):
        h = _neel(**kw)
        q = [0.1, 0., 0.]
        tdhf = magnon_energies(h, nk=NK, Q=q, n=1)[0].real
        poles = pairchi.pair_rpa_poles(h, q=q, energies=grid, delta=1e-4,
                                        nk=NK)
        sharp = sorted(p[0] for p in poles if abs(p[1]) < 0.05)
        assert abs(sharp[0] - tdhf) < 1e-4, f"{kw}: {sharp[0]} vs {tdhf}"


@pytest.mark.slow
def test_goldstone_of_a_metal_ordered_by_v1_alone():
    """No gap requirement anywhere in this route, so the case that needed
    bsetk/spinflip.py's metal mode works here directly. The residual is
    limited by the broadening and by nothing else: it is exactly
    proportional to delta (4.614e-3, -4, -5, -6 at delta 1e-3 ... 1e-6),
    which is what an exact zero looks like through a frequency scan."""
    g = geometry.chain()
    nk = 200
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 0.5])
    hm = VJinteraction(h, V1=1.1, filling=0.1, mf=mf, nk=nk, mix=0.2,
                        maxerror=1e-10, maxite=3000).hamiltonian
    assert abs(hm.get_gap()) < 1e-6  # metallic
    assert abs(hm.get_vev("sz")[0]) > 0.05  # magnetic
    res = [_kernel_min_eigenvalue(hm, nk=nk, delta=d)
           for d in (1e-3, 1e-4, 1e-5)]
    assert res[0] < 1e-2
    for a, b in zip(res, res[1:]):
        assert abs(b/a - 0.1) < 0.01  # strictly proportional to delta


@pytest.mark.slow
def test_it_reproduces_the_exact_saturated_ferromagnet_dispersion():
    """The closed-form two-body answer, in a METAL, which is the strongest
    check any of the three routes gets. Measured to 5 decimals:
    0.00173, 0.01291, 0.07756 at q = 0.02, 0.05, 0.1."""
    nk, nocc, U = 200, 41, 3.0  # odd nocc: symmetric occupied set, see below
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 0.5])
    hm = VJinteraction(h, U=U, filling=nocc/(2.0*nk), mf=mf, nk=nk, mix=0.2,
                        maxerror=1e-10, maxite=3000).hamiltonian
    ks = np.arange(nk)/nk
    eps = lambda k: -2.0*np.cos(2*np.pi*k)
    occ = np.argsort(eps(ks))[:nocc]

    def exact(q):
        dE = eps(ks[occ] + q) + U*nocc/nk - eps(ks[occ])
        return brentq(lambda w: 1.0 + U*np.sum(1.0/(w - dE))/nk,
                       1e-12, np.min(dE) - 1e-9, xtol=1e-13)

    grid = np.linspace(1e-5, 0.3, 3000)
    for q in (0.02, 0.05, 0.1):
        poles = pairchi.pair_rpa_poles(hm, q=[q, 0., 0.], energies=grid,
                                        delta=1e-5, nk=nk, channel="+-")
        sharp = sorted(p[0] for p in poles if abs(p[1]) < 0.02)
        assert abs(sharp[0] - exact(q)) < 1e-4


@pytest.mark.slow
def test_the_response_itself_comes_back_site_resolved():
    """The point of this route over the eigenproblem one: a chi(omega), in
    the site basis, which is what get_spinchi_full and the IETS maps
    consume. Its poles are where the kernel is singular."""
    h = _neel(U=3.0, V1=0.5)
    es = np.linspace(0.3, 0.9, 61)
    out, chi = h.get_transverse_spinchi(energies=es, q=[0.1, 0., 0.],
                                         delta=1e-2, nk=NK)
    chi = np.array(chi)
    assert chi.shape == (len(es), 2, 2)
    weight = np.array([abs(np.trace(c).imag) for c in chi])
    peak = es[np.argmax(weight)]
    assert abs(peak - 0.5949) < 0.05  # the magnon this state has at q=0.1
