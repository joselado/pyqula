"""The spin RPA on a neighbor-shell EXCHANGE interaction.

For a long time this was refused along with everything else non-onsite,
and the reason given was that the vertex could not be trusted. That reason
was too broad. The mean field for an isotropic J is not Ising-like at all:
scftk.spinspin's SCF decouples the x and y channels as well, by rotating
the density matrix into the frame where that axis is the computational z
(see _run_anisotropic_scf), so the converged state is a genuinely
SU(2)-symmetric Hartree-Fock one. What was missing was only a vertex to
match it, and h.V could not supply one -- an isotropic J1 and an
anisotropic J1z leave exactly the same z-channel matrix there.

The SCF now records the three channels separately in h.Vchannels, the
vertex is built per channel (chitk.spinchi._channel_spin_U), and the two
cases are both correct and distinguishable. The test that this is right is
the Goldstone theorem: the RPA kernel must be singular at q=0, w=0.

That vertex is still only the site-separable part of an exchange between
different sites: the Fock rung of the bonds lives on the electron-hole pair
index, so the public entry points now sum such an interaction in the pair
basis (chitk.pairchi), and the vertex itself is exercised here directly.
A neighbor-shell DENSITY-DENSITY interaction, whose spin response is that
rung and nothing else, goes the same way, see
test_density_density_goes_through_the_pair_basis below.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk.rpa import (_chi_ops_matrix_vectorized,
                               build_ops_projectors, interaction_at_q)
from pyqula.chitk.spinchi import _full_spin_operators, _full_spin_U
from pyqula.meanfield import VJinteraction

NK = 6


def _kernel_min_eigenvalue(h, nk=NK, delta=1e-4, q=(0., 0., 0.)):
    """Smallest eigenvalue of the RPA kernel 1 - V*chi0(q,w=0). Zero at
    q=0 is the Goldstone mode."""
    Ss = _full_spin_operators(h)
    V = _full_spin_U(h)
    pAs, pBs = build_ops_projectors(h, Ss)
    q = list(q)
    _, chis = _chi_ops_matrix_vectorized(h, ops=Ss, pAs=pAs, pBs=pBs, q=q,
                                          energies=np.array([0.0]),
                                          delta=delta, nk=nk)
    Vq = interaction_at_q(V, h, q)
    K = np.identity(Vq.shape[0], dtype=np.complex128) - Vq@chis[0]
    return np.min(np.abs(np.linalg.eigvals(K)))


def _neel(nk=NK, **kw):
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                         nk=nk, maxerror=1e-10, mix=0.3, maxite=3000, **kw)
    return scf.hamiltonian


def test_the_scf_records_the_three_exchange_channels():
    h = _neel(J1=3.0)
    assert h.Vchannels is not None
    assert set(h.Vchannels) == {"x", "y", "z", "d"}
    assert len(h.V) > 1  # h.V itself is non-onsite, i.e. the old gate's target


@pytest.mark.slow
def test_isotropic_exchange_keeps_its_goldstone_mode():
    """The measurement that justifies letting exchange through the gate.
    The controls are what make it a statement about the Goldstone mode
    rather than about a small number: the same kernel at finite q, and on
    a non-magnetic reference carrying the same interaction, are order
    one."""
    h = _neel(J1=3.0)
    assert abs(h.get_vev("sz")[0]) > 0.1
    assert _kernel_min_eigenvalue(h) < 1e-6
    assert _kernel_min_eigenvalue(h, q=(0.1, 0., 0.)) > 1e-2
    bare = geometry.honeycomb_lattice().get_hamiltonian()
    bare.V, bare.Vchannels = h.V, h.Vchannels
    assert _kernel_min_eigenvalue(bare) > 1e-2


@pytest.mark.slow
def test_exchange_alongside_an_onsite_hubbard_u_also_works():
    h = _neel(U=2.0, J1=1.0)
    assert _kernel_min_eigenvalue(h) < 1e-6


@pytest.mark.slow
def test_magnon_bands_run_and_reach_zero_at_gamma_for_isotropic_exchange():
    """End to end through the public API, which used to raise here."""
    h = _neel(J1=3.0)
    energies = np.linspace(0.02, 6., 300)
    qs, ws, gammas = h.get_magnon_bands(nq=6, energies=energies, delta=3e-2,
                                         nk=NK)
    at_gamma = [w for q, w, g in zip(qs, ws, gammas)
                if q == qs.max() and abs(g) < 0.1]
    assert len(at_gamma) > 0
    # the acoustic branch is at zero up to the resolution of this grid
    # (the scan starts at 0.02 and is broadened by delta=0.03)
    assert min(at_gamma) < 0.1


def test_an_anisotropic_exchange_gives_a_different_vertex_than_an_isotropic_one():
    """The reason h.V alone was not enough: J1=1 and J1z=1 leave the same
    z-channel matrix in it, and only the per-channel vertex tells them
    apart. The isotropic one must be proportional to the identity in the
    spin-channel index, the anisotropic one must not."""
    def vertex(**kw):
        h = _neel(nk=3, **kw)
        d = sorted(h.Vchannels["x"].keys())[-1]  # some neighbor-shell key
        return _full_spin_U(h)[d]
    n = 2  # sites per cell
    iso = vertex(J1=1.0)
    aniso = vertex(J1z=1.0)
    blocks_iso = [iso[i*n:(i+1)*n, i*n:(i+1)*n] for i in range(3)]
    blocks_ani = [aniso[i*n:(i+1)*n, i*n:(i+1)*n] for i in range(3)]
    assert np.allclose(blocks_iso[0], blocks_iso[2])  # x channel == z channel
    assert not np.allclose(blocks_ani[0], blocks_ani[2])  # x is empty, z is not
    assert np.max(np.abs(blocks_ani[0])) < 1e-12


@pytest.mark.slow
def test_density_density_goes_through_the_pair_basis():
    """A neighbor-shell density-density interaction enters the spin
    response only through a Fock rung on the electron-hole pair index,
    which a site-separable vertex cannot carry (V2K_matrix maps it to
    exactly zero). So the default method sums it in the pair basis, and
    gives exactly what method="pair" gives with the same arguments."""
    h = _neel(U=3.0, V1=0.5)
    assert h.Vchannels is not None
    kw = dict(nq=2, energies=np.linspace(0.02, 3., 50), delta=3e-2, nk=NK)
    a = h.get_magnon_bands(**kw)
    b = h.get_magnon_bands(method="pair", T=3e-2, **kw)
    assert len(a[0]) > 0
    for x, y in zip(a, b):
        assert np.allclose(x, y)


@pytest.mark.slow
def test_the_ladder_response_of_an_in_plane_anisotropic_exchange():
    """With Jx != Jy, S+/S- is not an eigen-channel of the interaction, so
    no site vertex exists for chi_{+-}. In the pair basis the ladder
    response is still defined, and it must be the combination of the
    full blocks that its operators say it is,
    chi(S+,S-) = chi_xx + chi_yy - i chi_xy + i chi_yx.
    Measured 5e-16 on the chain."""
    h = _neel(J1x=1.0)  # x channel only, y and z left at zero
    kw = dict(energies=np.linspace(0., 2., 5), delta=0.05, nk=NK,
              q=[0.2, 0., 0.])
    _, L = h.get_spinchi_ladder(**kw)
    _, F = h.get_spinchi_full(**kw)
    n = len(h.geometry.r)
    def b(i, j): return F[:, i*n:(i+1)*n, j*n:(j+1)*n]
    comb = b(0, 0) + b(1, 1) - 1j*b(0, 1) + 1j*b(1, 0)
    assert np.max(np.abs(L)) > 0.1
    assert np.max(np.abs(L - comb)) < 1e-10


@pytest.mark.slow
def test_an_easy_axis_anisotropy_gaps_the_magnon():
    """The physical consequence of building the vertex per channel rather
    than replicating one. An easy-axis anisotropy (J1z on top of an
    isotropic J1) breaks the continuous symmetry explicitly, so there is
    no Goldstone mode any more and the kernel stops being singular at q=0
    -- by more and more as the anisotropy grows. A replicated vertex
    cannot show this at all: it would report the isotropic answer for
    every J1z, since h.V is identical in all four of these runs.

    Measured: 4.3e-10, 3.2e-2, 9.1e-2, 2.5e-1 at J1z = 0, 0.1, 0.3, 1.0."""
    res = []
    for J1z in (0.0, 0.1, 0.3, 1.0):
        h = _neel(J1=3.0, J1z=J1z)
        res.append(_kernel_min_eigenvalue(h))
    assert res[0] < 1e-6  # isotropic: gapless
    assert np.all(np.diff(res) > 0)  # and gapping out monotonically
    assert res[-1] > 0.1


@pytest.mark.slow
def test_the_ladder_and_the_full_kernel_find_the_same_magnon():
    """The transverse (S+/S-) response and the poles of the (Sx,Sy,Sz)
    kernel are separate code paths, for the site vertex (U) and for the
    pair basis (J1) alike. On an isotropic interaction they describe the
    same excitation and must peak at the same energy.

    Measured at q=0.1 on the honeycomb Neel state: 0.4917 vs 0.4900 at U=3.
    At J1=3 both come from the pair basis, 1.3687 from the poles, which is
    the time-dependent Hartree-Fock value, against 1.3700 from the ladder
    peak, agreement to the 0.005 spacing of the grid; the site vertex put
    this magnon at 1.3296, without the Fock rung of the exchange bonds."""
    energies = np.linspace(0.005, 4., 800)
    q = [0.1, 0., 0.]
    for kw in ({"U": 3.0}, {"J1": 3.0}):
        h = _neel(**kw)
        _, ws, gammas = h.get_magnon_bands(qpath=[q], nq=1,
                                           energies=energies, delta=2e-3,
                                           nk=NK)
        acoustic = min(w for w, g in zip(ws, gammas) if abs(g) < 0.02)
        es, chis = h.get_spinchi_ladder(energies=energies, q=q, delta=2e-3,
                                         nk=NK)
        peak = es[np.argmax(np.abs([np.trace(c).imag for c in chis]))]
        assert abs(acoustic - peak) < 0.01, f"{kw}: {acoustic} vs {peak}"
        if "J1" in kw: assert abs(acoustic - 1.3687) < 5e-3
