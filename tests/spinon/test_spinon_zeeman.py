import numpy as np

from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

# An external Zeeman field couples EXACTLY to the Abrikosov pseudofermion
# (S_i = 1/2 f_i^dagger sigma f_i is already bilinear in f, see spinon.py's
# class docstring) -- it is a single-particle term added to the auxiliary
# Hamiltonian before the RVB exchange mean field, via the ordinary inherited
# Hamiltonian.add_zeeman/add_exchange, not a new code path. These tests lock
# in that behavior end to end: the field survives into the converged
# Hamiltonian, the local one-fermion-per-site constraint stays intact under
# it, and the induced response is isotropic and saturates correctly.


def _magnetization(g, J1, zeeman, seed=0, nk=24):
    np.random.seed(seed)  # see test_spinon_constraint.py for why
    h = SpinonHamiltonian(g)
    h.add_zeeman(zeeman)
    h2 = h.get_mean_field_hamiltonian(J1=J1, nk=nk, mix=0.3,
            maxerror=1e-6, maxite=2000)
    assert h2 is not None, "SCF did not converge"
    return h2.local_occupation, h2.get_magnetization()


def test_zero_exchange_field_reproduces_bare_zeeman_exactly():
    """With J1=0 the RVB mean field is identically zero (no exchange to
    decouple), so the converged Hamiltonian's intra is exactly hop0 (the
    bare Zeeman term) modulo a spin-symmetric constraint shift that cancels
    in the up/down difference get_magnetization reads -- i.e. this is a
    convention-free, exactly-solvable check that also pins down
    get_magnetization's sign/normalization for the field-only case: the
    readout must equal the input field component for component, with zero
    on the other two axes."""
    g = geometry.chain()
    occ, m = _magnetization(g, J1=0.0, zeeman=[0., 0., 0.3])
    assert np.allclose(occ, 1.0, atol=1e-6)
    assert np.allclose(m, [[0., 0., 0.3]], atol=1e-6), m


def test_local_occupation_stays_one_under_a_field():
    """The local constraint enforced by the array-filling machinery is a
    total-occupation (not a spin) constraint -- a field is free to induce a
    net polarization while <n_i>=1 stays exact at every site, on a
    non-uniform (2-site supercell) chain too, not just a translationally
    uniform one. Uses a non-frustrated geometry deliberately (see
    test_spinon_constraint.py's uniform/non-uniform pair) -- a frustrated
    lattice's random mf seed can land on a non-converging ansatz within
    maxite regardless of the field, which is a pre-existing SCF-seed
    property this test is not about."""
    g = geometry.chain().get_supercell(2)
    occ, m = _magnetization(g, J1=1.0, zeeman=[0., 0., 0.3])
    assert len(occ) == 2
    assert np.allclose(occ, 1.0, atol=1e-3), occ


def test_response_is_isotropic():
    """A field of the same magnitude along z vs x must induce the same
    magnitude of response -- exercises the SzSz-direct-decouple code path
    (z) against the rotate-decouple-rotate-back path (x/y) with a
    field-breaking bare term now present in both, not just the pure
    exchange case those two code paths are already separately tested
    against each other for."""
    g = geometry.chain()
    occ_z, m_z = _magnetization(g, J1=1.0, zeeman=[0., 0., 0.3])
    occ_x, m_x = _magnetization(g, J1=1.0, zeeman=[0.3, 0., 0.])
    assert np.allclose(occ_z, 1.0, atol=1e-6)
    assert np.allclose(occ_x, 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(m_z), np.linalg.norm(m_x), rtol=1e-3)
    # and the response lies along the applied field's own axis
    assert np.allclose(m_z[0][:2], 0.0, atol=1e-3), m_z
    assert np.allclose(m_x[0][1:], 0.0, atol=1e-3), m_x


def test_saturation_at_large_field():
    """At a field much larger than J1, exchange feedback is negligible and
    the auxiliary fermion is (nearly) fully polarized into the lower Zeeman
    level -- the response must approach the same bare-field value the J1=0
    case reproduces exactly, from below (a finite J1 partially screens/
    resists the field, see test_response_is_isotropic's smaller-than-bare
    result at J1=1, zeeman=0.3)."""
    g = geometry.chain()
    occ, m = _magnetization(g, J1=1.0, zeeman=[0., 0., 50.0])
    assert np.allclose(occ, 1.0, atol=1e-3), occ
    assert np.isclose(m[0][2], 50.0, atol=0.5), m


def test_add_exchange_matches_add_zeeman():
    """add_exchange (-> Hamiltonian.add_magnetism, a different function in
    magnetism.py than add_zeeman) is the idiom used elsewhere in this
    codebase and in the user guide -- check it lands on the same converged
    state as add_zeeman for a plain constant-vector field, not a
    diverging/differently-normalized one."""
    g = geometry.chain()
    occ_zeeman, m_zeeman = _magnetization(g, J1=1.0, zeeman=[0., 0., 0.3])
    np.random.seed(0)
    h = SpinonHamiltonian(g)
    h.add_exchange([0., 0., 0.3])
    h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=24, mix=0.3,
            maxerror=1e-6, maxite=2000)
    assert h2 is not None
    assert np.allclose(h2.local_occupation, occ_zeeman, atol=1e-6)
    assert np.allclose(h2.get_magnetization(), m_zeeman, atol=1e-6)
