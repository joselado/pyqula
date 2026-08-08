import numpy as np

from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

# An external Zeeman field couples EXACTLY to both fermion species here --
# the conduction electron (an ordinary electron) and the localized moment
# (S_j = 1/2 f_j^dagger sigma f_j, already bilinear in f, see
# kondolattice.py's module docstring) -- so it is a single-particle term
# added to h0 via the ordinary inherited Hamiltonian.add_zeeman/add_exchange
# BEFORE get_mean_field_hamiltonian, exactly like SpinonHamiltonian (see
# tests/spinon/test_spinon_zeeman.py); kondo_lattice_mean_field's own
# hop0 = h1.get_dict() carries it through every SCF iteration unchanged, no
# new code path needed. What IS specific to this class: the local
# constraint only applies to the f-sites (a total-occupation constraint,
# unaffected by the field), and -- genuine physics, not a bug -- a strong
# enough field destroys the Kondo hybridization, the same "decays toward
# the trivial V=0 branch" non-convergence signal
# test_subcritical_J_does_not_falsely_converge already exercises for
# subcritical J.


def _chain():
    gc = geometry.chain()
    hc = gc.get_hamiltonian(has_spin=True)
    return KondoLatticeHamiltonian(hc)


def test_zero_J_field_reproduces_bare_zeeman_on_both_sublattices():
    """With J=0 there is no Kondo coupling at all: the converged
    Hamiltonian is exactly the bare (field-split) conduction + f bands, so
    get_magnetization must reproduce the input field exactly on BOTH the
    conduction site and the f site -- a convention-free sign/normalization
    check, same idea as the analogous spinon test."""
    h = _chain()
    h.add_zeeman([0., 0., 0.3])
    h2 = h.get_mean_field_hamiltonian(J=0.0, filling=0.5, nk=60, mix=0.5,
            maxerror=1e-7, maxite=500)
    assert h2 is not None, "SCF did not converge"
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3)
    assert np.allclose(h2.hybridization, 0.0, atol=1e-8)
    assert np.allclose(h2.get_magnetization(), [[0., 0., 0.3], [0., 0., 0.3]],
            atol=1e-6)


def test_local_occupation_stays_one_under_a_field():
    """The f-site constraint is a total-occupation constraint, unaffected
    by a (weak) field -- checked on the same non-uniform 2-site supercell
    (inequivalent conduction onsite energies) as
    test_local_occupation_is_one_nonuniform_supercell, now with a field
    added on top."""
    gc = geometry.chain().get_supercell(2)
    hc = gc.get_hamiltonian(has_spin=True)
    hc.add_onsite([0.3, -0.3])
    h = KondoLatticeHamiltonian(hc)
    h.add_zeeman([0., 0., 0.02])
    seed = (np.array([0.3+0.0j, 0.3+0.0j]), np.array([0.0, 0.0]))
    h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=100, mf=seed,
            mix=0.3, maxerror=1e-6, maxite=3000)
    assert h2 is not None, "SCF did not converge"
    assert len(h2.local_occupation) == 2
    assert np.allclose(h2.local_occupation, 1.0, atol=1e-3), h2.local_occupation


def test_hybridization_is_suppressed_by_a_field():
    """A field competes with the Kondo singlet: the self-consistent
    hybridization amplitude |V| must shrink (not grow or stay flat) as the
    field grows, at fixed J -- the large-N-mean-field-level signature of
    field-driven Kondo screening suppression."""
    h = _chain()
    seed = (np.array([0.3+0.0j]), np.array([0.0]))

    def hyb(b):
        h = _chain()
        h.add_zeeman([0., 0., b])
        h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150,
                mf=seed, mix=0.3, maxerror=1e-6, maxite=3000)
        assert h2 is not None, "SCF did not converge for zeeman=%s" % b
        return abs(h2.hybridization[0])

    v0 = hyb(0.0)
    v1 = hyb(0.04)
    v2 = hyb(0.08)
    assert v0 > v1 > v2, (v0, v1, v2)


def test_strong_field_destroys_the_kondo_state():
    """Push the field past the (large-N mean-field) critical value where
    the hybridized branch stops being a genuine fixed point: the SCF must
    correctly report non-convergence (None), exactly like a subcritical J
    (test_subcritical_J_does_not_falsely_converge) -- the relative-residual
    convergence check is what makes this the correct answer here rather
    than a silently wrong small-but-nonzero V, since a field-destroyed
    hybridization decays geometrically toward the always-self-consistent
    V=0 trivial branch, the same trajectory shape a subcritical J
    produces."""
    h = _chain()
    seed = (np.array([0.3+0.0j]), np.array([0.0]))
    h.add_zeeman([0., 0., 0.2])
    h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150, mf=seed,
            mix=0.3, maxerror=1e-6, maxite=3000)
    assert h2 is None, (
        "expected the Kondo state to be destroyed by a strong field",
        h2.hybridization if h2 else None)


def test_add_exchange_matches_add_zeeman():
    """add_exchange (-> Hamiltonian.add_magnetism, a different function
    than add_zeeman) is the idiom used elsewhere in the codebase and the
    user guide -- check it lands on the same converged state."""
    seed = (np.array([0.3+0.0j]), np.array([0.0]))

    h_zeeman = _chain()
    h_zeeman.add_zeeman([0., 0., 0.04])
    h2_zeeman = h_zeeman.get_mean_field_hamiltonian(J=1.5, filling=0.15,
            nk=150, mf=seed, mix=0.3, maxerror=1e-6, maxite=3000)

    h_exchange = _chain()
    h_exchange.add_exchange([0., 0., 0.04])
    h2_exchange = h_exchange.get_mean_field_hamiltonian(J=1.5, filling=0.15,
            nk=150, mf=seed, mix=0.3, maxerror=1e-6, maxite=3000)

    assert h2_zeeman is not None and h2_exchange is not None
    assert np.allclose(h2_zeeman.local_occupation, h2_exchange.local_occupation)
    assert np.allclose(h2_zeeman.hybridization, h2_exchange.hybridization)
    assert np.allclose(h2_zeeman.get_magnetization(), h2_exchange.get_magnetization())
