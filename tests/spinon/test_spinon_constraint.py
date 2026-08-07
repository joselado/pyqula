import numpy as np
import pytest

from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

# SpinonHamiltonian implements Abrikosov-pseudofermion mean-field theory for
# a spin-1/2 Heisenberg model (Savary & Balents, arXiv:1601.03742, Sec. 4):
# the local constraint <n_i>=1 (exactly one auxiliary fermion per site) must
# hold at EVERY site, enforced via VJinteraction's per-site `filling` array
# path (selfconsistency.spinspin._run_anisotropic_scf) -- these tests check
# that guarantee end to end through the class, not just the lower-level SCF
# machinery (already covered by tests/scf/test_vjinteraction_local_filling.py).


def test_local_occupation_is_one_everywhere_uniform_chain():
    """A translationally-uniform 1-site-unit-cell Heisenberg chain: the
    RVB bond order parameter does not need to break any symmetry, so this
    isolates the local-constraint machinery (the actual point of this
    class) from any mean-field convergence difficulty."""
    np.random.seed(0) # reproducible: an unseeded random mf guess can
                       # occasionally fail to converge within maxite,
                       # unrelated to what this test checks
    g = geometry.chain()
    h = SpinonHamiltonian(g)
    h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=24, mix=0.3,
            maxerror=1e-5, maxite=2000)
    assert h2 is not None, "SCF did not converge"
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3), h2.local_occupation


def test_local_occupation_is_one_everywhere_nonuniform_supercell():
    """A 2-site chain supercell: unlike the uniform case above, a generic
    random SCF seed has no reason to keep the two sites equivalent, so this
    checks the constraint is enforced independently at every site, not just
    on lattice average."""
    np.random.seed(0) # see the uniform-chain test above for why
    g = geometry.chain().get_supercell(2)
    h = SpinonHamiltonian(g)
    h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=24, mix=0.3,
            maxerror=1e-5, maxite=2000)
    assert h2 is not None, "SCF did not converge"
    assert len(h2.local_occupation) == 2
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3), h2.local_occupation


def _run_chain(seed, J1, maxerror=1e-6):
    """Shared helper: converged (Hamiltonian, total_energy/site) for the
    1-site-unit-cell chain -- translationally uniform, so it has a unique
    self-consistent RVB solution (see test_energy_scales_linearly_with_J's
    docstring for why this matters and why the frustrated/triangular case
    is deliberately NOT used in these tests). return_total_energy=True is
    required here, not h2.get_total_energy(): that method alone is just the
    sum of occupied spinon-band energies, missing the Hartree-Fock
    double-counting (and lambda grand-potential) correction
    _run_anisotropic_scf's own scf.total_energy already includes -- using
    it directly would silently double-count the interaction."""
    g = geometry.chain()
    np.random.seed(seed)
    h = SpinonHamiltonian(g)
    h2, etot = h.get_mean_field_hamiltonian(J1=J1, nk=24, mix=0.3,
            maxerror=maxerror, maxite=2000, return_total_energy=True)
    assert h2 is not None, "SCF did not converge"
    return h2, etot/len(g.r)


def test_energy_per_site_independent_of_scf_seed():
    """The converged ground-state energy per site (a physical observable)
    must not depend on the random mean-field guess used to seed the SCF
    loop -- the same invariant this repo's other SCF tests check (see
    CLAUDE.md), applied here to the RVB spinon mean field. Only exercised on
    the chain (a unique self-consistent solution) -- a frustrated lattice
    (e.g. triangular) can have several distinct self-consistent parton
    ansatze (different flux sectors) at the same J, so a random seed
    landing on different ones is expected physics, not a bug (Savary &
    Balents, arXiv:1601.03742, Sec. 4.1: "it is not possible to search for
    all possible self-consistent mean field solutions... calculations are
    usually carried out by assuming a particular decoupling scheme")."""
    _, e1 = _run_chain(1, J1=1.0)
    _, e2 = _run_chain(2, J1=1.0)
    assert np.isclose(e1, e2, atol=1e-3), (e1, e2)


def test_rvb_bond_order_is_nonzero():
    """A regression guard against a silently dead RVB channel: every test
    above (constraint satisfied, energy seed-independent) would still pass
    if chi_ij were identically zero on every bond (a flat, chi=0 spinon band
    trivially satisfies <n_i>=1 via lambda alone, and 0==0 is seed-
    independent) -- so explicitly check the converged mean field actually
    produced a nonzero bond order parameter."""
    h2, _ = _run_chain(0, J1=1.0)
    hoppings = h2.get_multihopping().get_dict()
    max_offsite = max(np.max(np.abs(m))
            for d, m in hoppings.items() if d != (0, 0, 0))
    assert max_offsite > 1e-3, "converged RVB bond order parameter is zero"


def test_energy_scales_linearly_with_J():
    """With zero bare hopping, J is the only energy scale in the problem:
    the converged (dimensionless, J-independent) bond order parameter chi
    sets a mean-field Hamiltonian proportional to J, so the ground-state
    energy per site must be EXACTLY linear in J -- a strong,
    convention-free correctness check that does not depend on any
    literature reference value."""
    _, e1 = _run_chain(0, J1=1.0)
    _, e2 = _run_chain(0, J1=2.0)
    assert np.isclose(e2, 2*e1, rtol=1e-3), (e1, e2)


def test_filling_kwarg_cannot_be_overridden():
    """The auxiliary-fermion representation is only physically valid at
    exactly one fermion per site -- filling= is not a free SCF parameter
    here, unlike for a plain get_mean_field_hamiltonian call."""
    g = geometry.chain()
    h = SpinonHamiltonian(g)
    with pytest.raises(ValueError):
        h.get_mean_field_hamiltonian(J1=1.0, filling=0.3)


def test_bare_hopping_is_zero():
    """A pure spin model has no bare electron kinetic term -- all "hopping"
    in the converged Hamiltonian must come from the RVB mean field itself,
    not from SpinonHamiltonian's own starting point."""
    # note: tij=[0.0] (see SpinonHamiltonian.__init__) prunes exactly-zero
    # bonds rather than keeping explicit zero-valued Hopping entries around
    # -- h.hopping is empty at construction, not a list of zero matrices.
    # That's fine: the exchange decoupling (selfconsistency.spinspin._build_v)
    # rebuilds its own neighbor-shell bond structure directly from the
    # geometry, never from self.hopping (see test_rvb_bond_order_is_nonzero,
    # which confirms the mean field is nonzero despite this).
    g = geometry.chain()
    h = SpinonHamiltonian(g)
    assert np.allclose(h.intra, 0.0)
    for hop in h.hopping:
        assert np.allclose(hop.m, 0.0)
