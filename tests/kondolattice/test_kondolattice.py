import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

# KondoLatticeHamiltonian implements the large-N Abrikosov-pseudofermion
# (Read-Newns) mean-field theory of the Kondo lattice/periodic Anderson
# model, following P. Coleman, "Heavy Fermions: electrons at the edge of
# magnetism", arXiv:cond-mat/0612006, Eq. 65-99 -- see kondolattice.py's
# module docstring. These tests check the two defining physical invariants
# (the local <n_f>=1 constraint, and V=0 vs. a genuine Kondo hybridization
# both being self-consistent solutions) end to end through the class.
#
# filling=0.15 (rather than e.g. 0.5) is used throughout for any test that
# needs a metallic conduction sector: the bare (V=0) f-sector is a
# perfectly flat, macroscopically degenerate band, so a filling target that
# lands the chemical potential inside that degenerate block (roughly
# 0.25-0.75 for this one-orbital-per-site chain) is a numerically
# ill-posed starting point for the SCF loop -- see
# scftk.kondolattice.kondo_lattice_mean_field's docstring.


def _chain():
    gc = geometry.chain()
    hc = gc.get_hamiltonian(has_spin=True)
    return KondoLatticeHamiltonian(hc)


def test_v_zero_flat_f_band_is_self_consistent():
    """With J=0 there is no Kondo coupling at all: the self-consistent
    hybridization must vanish, leaving the bare conduction dispersion
    untouched and a flat (dispersionless) f-band pinned at lam=0 by
    particle-hole symmetry -- the cheapest possible smoke test for the
    geometry/fusion wiring (a real, but nonzero, coupling could still hide
    an indexing bug that this catches for free)."""
    h = _chain()
    h2 = h.get_mean_field_hamiltonian(J=0.0, filling=0.5, nk=60, mix=0.5,
            maxerror=1e-7, maxite=500)
    assert h2 is not None, "SCF did not converge"
    assert np.allclose(h2.hybridization, 0.0, atol=1e-8)
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3)


def test_v_zero_is_also_a_fixed_point_at_nonzero_J():
    """Like the BCS gap equation, V=0 is always a self-consistent solution
    of the hybridization saddle point (Eq. 77): with no seed, the SCF loop
    starts at V=0 and A=<f^dagger c>=0 there too (no c-f coupling to
    generate a nonzero expectation value from), so it stays at V=0 even for
    a J that does support a genuine, distinct Kondo state (see
    test_hybridization_grows_with_J below) -- a finite seed is needed to
    find that other state, exactly as with BCS pairing."""
    h = _chain()
    h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=100, mix=0.3,
            maxerror=1e-6, maxite=1000)
    assert h2 is not None, "SCF did not converge"
    assert np.allclose(h2.hybridization, 0.0, atol=1e-6)
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3)


def _hyb(h, J, filling=0.15, nk=150):
    seed = (np.array([0.3+0.0j]), np.array([0.0]))
    h2 = h.get_mean_field_hamiltonian(J=J, filling=filling, nk=nk, mf=seed,
            mix=0.3, maxerror=1e-6, maxite=3000)
    assert h2 is not None, "SCF did not converge"
    return h2


def test_hybridization_grows_with_J():
    """Away from the flat-band-degenerate filling, a large-enough seed
    converges to a genuine Kondo state (nonzero hybridization, distinct
    from the trivial V=0 solution above) whose magnitude grows with the
    Kondo coupling J, as Eq. 88/97 predicts (a wider hybridization gap at
    larger J/rho)."""
    h = _chain()
    v1 = abs(_hyb(h, 1.2).hybridization[0])
    v2 = abs(_hyb(h, 2.0).hybridization[0])
    assert v2 > 2*v1, (v1, v2)


def test_subcritical_J_does_not_falsely_converge():
    """Regression guard: with a nonzero seed, a J just below the sharp
    hybridization threshold (Eq. 77's saddle point has no other root than
    V=0 there) makes |V| decay geometrically toward 0 iteration by
    iteration rather than settling at a genuine nonzero fixed point. An
    earlier version of the SCF loop compared this residual to an ABSOLUTE
    tolerance, so it falsely reported "converged" at whatever small value
    |V| happened to have reached once it dropped below maxerror -- a
    tolerance/iteration-count artifact, not a physical answer (it changed
    with maxerror). The residual is now relative to |V| itself, so this
    case should consistently fail to converge (report None) regardless of
    how tight or loose maxerror is, rather than agreeing on some nonzero
    value by coincidence."""
    h = _chain()
    seed = (np.array([0.3+0.0j]), np.array([0.0]))
    for maxerror in (1e-4, 1e-8):
        h2 = h.get_mean_field_hamiltonian(J=0.9, filling=0.15, nk=100,
                mf=seed, mix=0.3, maxerror=maxerror, maxite=500)
        assert h2 is None, (maxerror, h2.hybridization if h2 else None)


def test_local_occupation_is_one_nonuniform_supercell():
    """A 2-site chain supercell with inequivalent conduction-site onsite
    energies: unlike the uniform case, a generic mean field has no reason
    to keep the two f-sites equivalent, so this checks the constraint is
    enforced independently at every f-site, not just on lattice average."""
    gc = geometry.chain().get_supercell(2)
    hc = gc.get_hamiltonian(has_spin=True)
    hc.add_onsite([0.3, -0.3])
    h = KondoLatticeHamiltonian(hc)
    seed = (np.array([0.3+0.0j, 0.3+0.0j]), np.array([0.0, 0.0]))
    h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=100, mf=seed,
            mix=0.3, maxerror=1e-6, maxite=3000)
    assert h2 is not None, "SCF did not converge"
    assert len(h2.local_occupation) == 2
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3), h2.local_occupation
    # the two f-sites are inequivalent (different conduction onsite energy
    # underneath them), so nothing forces their hybridization to agree
    assert not np.isclose(h2.hybridization[0], h2.hybridization[1], atol=1e-2)


def test_kondo_branch_has_lower_energy_than_trivial_branch():
    """Above the hybridization threshold, V=0 (unseeded) and a genuine
    Kondo state (seeded) are BOTH self-consistent solutions of Eq. 77 (see
    test_v_zero_is_also_a_fixed_point_at_nonzero_J) -- but only one of them
    should be the actual ground state. This is also the first real exercise
    of return_total_energy=True's Eq. 83 constant terms
    (sum_j |V_j|^2/J - lam_j*Q) and the mu*n_electrons un-shift: get either
    sign wrong and this assertion is the one thing here that would catch
    it (every other test only checks hybridization/local_occupation, never
    total_energy)."""
    h = _chain()
    h0, e0 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150,
            mix=0.3, maxerror=1e-6, maxite=1000, return_total_energy=True)
    assert h0 is not None and np.allclose(h0.hybridization, 0.0, atol=1e-6)
    h1 = _hyb(h, 1.5)
    _, e1 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150,
            mf=(h1.hybridization, h1.constraint_lambda), mix=0.3,
            maxerror=1e-6, maxite=100, return_total_energy=True)
    assert not np.allclose(h1.hybridization, 0.0, atol=1e-6) # still the Kondo branch
    assert e1 < e0, (e0, e1)


def test_bdg_conduction_hamiltonian_rejected():
    gc = geometry.chain()
    hc = gc.get_hamiltonian(has_spin=True)
    hc.turn_nambu()
    with pytest.raises(ValueError):
        KondoLatticeHamiltonian(hc)
