"""The Goldstone theorem, which is the whole test of the TDHF magnon route.

A mean-field state that orders magnetically without spin-orbit coupling
breaks SU(2) spontaneously, so its spin response must have a mode at
exactly zero energy at Q=0: rotating every spin by the same angle costs
nothing. Time-dependent Hartree-Fock reproduces that exactly when the same
interaction generates the mean field and the kernel, so the residual
measured here is a statement about nothing but the convergence of the SCF
-- which is what makes it sharp enough to be worth asserting on.

The measurement is ||M v||, with v the spin generator written in the
electron-hole pair basis, rather than "the eigenvalue closest to zero".
The zero eigenvalue is defective (the generator and its conjugate span a
Jordan block), so it converges only as the square root of the same error:
at maxerror 1e-10 the eigenvalue sits at 4e-5 while the residual is at
2e-10. Asserting on the eigenvalue would therefore be a much weaker test.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction

NK = 6  # the mean field and the magnon problem MUST share this, see below
TOL = 1e-6  # a real Goldstone mode lands 4 orders of magnitude below this


def _neel_honeycomb(nk=NK, U=3.0, maxerror=1e-10):
    """A Neel-ordered honeycomb Hubbard insulator, converged on the same
    mesh the magnon problem will use"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=U, filling=0.5, mf="antiferro",
                                        nk=nk, maxerror=maxerror)


def _saturated_ferromagnet(nk=NK, U=10.0):
    """A fully polarized Hubbard chain: one electron per site, an
    exchange splitting larger than the bandwidth, so the majority band is
    full and the minority one empty and the state is an insulator.

    The SCF needs a strong explicit guess to find it -- the unpolarized
    solution is also a fixed point, and mf="ferro" alone converges to that
    one. This state is a saddle rather than a minimum (at half filling in
    one dimension the ground state is the antiferromagnet, and the magnon
    energies away from Q=0 come out negative accordingly), which is fine
    for the Goldstone theorem: it applies to any stationary point of the
    mean-field problem, stable or not."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 3.0])
    return h.get_mean_field_hamiltonian(U=U, filling=0.5, mf=mf, nk=nk,
                                        maxerror=1e-10)


def test_goldstone_of_a_hubbard_antiferromagnet():
    h = _neel_honeycomb()
    assert abs(h.get_vev("sz")[0]) > 0.1  # actually ordered
    assert h.get_goldstone_residual(nk=NK) < TOL


def test_goldstone_with_a_neighbor_shell_density_density_interaction():
    """The case the site-basis spin RPA of chitk/spinchi.py cannot do at
    all: a mean field ordered with a V1 neighbor-shell interaction
    alongside the Hubbard U. Its spin vertex there is V2K_matrix, which
    maps a spin-independent V_ij to exactly zero -- so V1 would contribute
    nothing to the magnons -- while here it enters the pair-index rung
    where it belongs and the Goldstone mode survives."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    scf = VJinteraction(h, U=3.0, V1=0.5, filling=0.5, mf="antiferro",
                         nk=NK, maxerror=1e-10, mix=0.3, maxite=2000)
    hmf = scf.hamiltonian
    assert len(hmf.V) > 1  # genuinely non-onsite
    assert abs(hmf.get_vev("sz")[0]) > 0.1
    assert hmf.get_goldstone_residual(nk=NK) < TOL


def test_goldstone_of_a_saturated_ferromagnet():
    """The other extreme: no minority electrons at all, so the magnon
    problem has no de-excitation half (n2 = 0) and is an ordinary
    Hermitian eigenproblem. The Goldstone mode is then exact to machine
    precision rather than to the SCF tolerance."""
    h = _saturated_ferromagnet()
    assert abs(h.get_vev("sz")[0]) > 0.99  # fully polarized
    from pyqula.bsetk.spinflip import magnon_matrix
    p = magnon_matrix(h, Q=[0., 0., 0.], nk=NK)
    assert p.n2 == 0  # nothing to de-excite
    assert h.get_goldstone_residual(nk=NK) < 1e-12


def test_goldstone_does_not_care_which_axis_the_state_ordered_along():
    """A magnetic state pointing along an arbitrary axis is not an Sz
    eigenstate, so the pair basis has no spin-flip block to restrict to
    and the whole basis is kept instead. That is a different code path,
    and it has to give the same answer: the Goldstone theorem is about a
    rotation of the state, and cannot know which axis was written down
    first."""
    h = _neel_honeycomb()
    tilted = h.copy()
    tilted.global_spin_rotation(vector=[1., 0., 0.], angle=0.5)
    assert abs(tilted.get_vev("sz")[0]) < 1e-8  # no longer along z
    assert abs(tilted.get_vev("sy")[0]) > 0.1
    assert tilted.get_goldstone_residual(nk=NK) < TOL
    # and the whole spectrum is the same, not just the zero mode
    ez = np.sort(h.get_magnon_energies(nk=NK, channel="all").real)
    et = np.sort(tilted.get_magnon_energies(nk=NK, channel="all").real)
    assert np.max(np.abs(ez - et)) < 1e-6


@pytest.mark.slow
def test_goldstone_residual_tracks_the_scf_tolerance():
    """Nothing but the convergence of the mean field is allowed to
    contribute to the residual, so tightening the SCF has to tighten it
    proportionally. Measured: 1.8e-6, 1.8e-8, 1.8e-10 at maxerror 1e-6,
    1e-8, 1e-10."""
    res = [_neel_honeycomb(maxerror=tol).get_goldstone_residual(nk=NK)
           for tol in (1e-6, 1e-8, 1e-10)]
    assert res[0] > res[1] > res[2]
    assert res[2] < res[0]/1e3  # four orders of SCF buys four of residual


@pytest.mark.slow
def test_a_mismatched_k_mesh_breaks_the_goldstone_mode():
    """The Ward identity behind the Goldstone mode holds between a mean
    field and a kernel evaluated on the SAME mesh. A mean field converged
    at a different nk is not self-consistent on the magnon mesh, and the
    acoustic branch acquires a real gap. This is a documented constraint
    of the method rather than a bug, and it is pinned here because it is
    the single easiest way to get a plausible-looking wrong dispersion."""
    h = _neel_honeycomb(nk=20)  # converged on a mesh the magnon will not use
    assert h.get_goldstone_residual(nk=4) > 1e-3


@pytest.mark.slow
def test_goldstone_of_a_spontaneously_non_collinear_magnet():
    """The case with no collinear frame at all: the 120-degree spiral of
    the triangular-lattice Hubbard model, on a 3x3 supercell at large U.
    Its bands are not Sz eigenstates in any frame, so there is no
    spin-flip block to restrict to and the whole pair basis is used -- and
    the Goldstone theorem still has to hold, since the state breaks SU(2)
    just as completely as a collinear one does.

    The initial guess is an explicit 120-degree spiral rather than
    mf="random": a random guess also converges here, but not always to the
    non-collinear solution, and a test that silently checks a collinear
    state instead would be checking nothing new."""
    g = geometry.triangular_lattice().get_supercell([3, 3])
    h = g.get_hamiltonian()

    def spiral(r):  # 120 degrees in the xy plane, by sublattice
        ph = 2*np.pi*(r[0] + 2*r[1])/3.
        return [np.cos(ph), np.sin(ph), 0.]

    mf = h.copy()
    mf.add_exchange(spiral)
    nk, tol = 3, 1e-9  # a 9-site cell: 729 pairs already at nk=3
    hmf = h.get_mean_field_hamiltonian(U=8.0, filling=0.5, mf=mf, nk=nk,
                                        maxerror=tol, mix=0.2, maxite=3000)
    m = np.array([hmf.get_vev("sx"), hmf.get_vev("sy"), hmf.get_vev("sz")]).T
    n = m/np.linalg.norm(m, axis=1)[:, None]
    assert np.min(np.linalg.norm(m, axis=1)) > 0.5  # it ordered
    assert 1 - np.min(np.abs(n@n[0])) > 0.4  # and genuinely non-collinearly
    assert hmf.get_goldstone_residual(nk=nk) < 1e-6
