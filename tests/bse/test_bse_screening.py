"""The static RPA screened interaction W = eps^-1 v, and the BSE built on
top of it (bsetk/screening.py).

There is no benchmark to run this against. Xatu (arXiv:2307.01572), the
code the rest of this BSE follows, does not compute an RPA W at all -- it
substitutes a phenomenological Rytova-Keldysh form -- and no comparable
open tight-binding implementation of a TB-RPA W was found. So the whole
burden falls on internal cross-checks, and they are ordered below roughly
by how much they are worth: the exact identities first, the independent
code path next, and the physical sanity checks last.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula.bsetk import screening as sc
from pyqula.bsetk.interaction import density_interaction, interaction_at_q, qkey

NK = 8


def _coulomb(e2):
    """A soft-cutoff Coulomb tail, as tests/bse/test_bse_physics.py uses"""
    return lambda r1, r2: e2 / np.sqrt((r1 - r2).dot(r1 - r2) + 0.25)


def _nondegenerate_chain():
    """A gapped spinless chain with four DIFFERENT onsite energies in the
    cell, so no two bands touch anywhere on the mesh.

    Needed because cRPA is only meaningful with a genuine nv/nc subset of
    the bands, and the usual test models cannot supply one: every band of
    a spinful non-magnetic model is two-fold degenerate, and a supercell of
    a smaller cell folds bands onto each other. Either way a small window
    would cut a multiplet in half, which pairbasis.select_bands rightly
    warns about and which would make the exciton energies arbitrary."""
    from pyqula import geometry
    onsite = [1.3, -0.9, 0.6, -1.1]
    h = geometry.chain().supercell(4).get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: onsite[int(round(r[0] - 0.5)) % 4])
    return h.get_multicell().get_dense()


def _models():
    return [("chain", gapped_ionic_chain()),
            ("honeycomb", gapped_honeycomb(mass=1.0))]


def _reverse(obj, qs):
    """Pair each q with the index of -q on the same mesh"""
    index = {qkey(q): i for i, q in enumerate(qs)}
    return [(i, index[qkey(-q)]) for i, q in enumerate(qs)]


# --- exact identities -------------------------------------------------

@pytest.mark.parametrize("name,h", _models())
def test_polarizability_is_hermitian_and_negative(name, h):
    """chi0 = sum of w * rho outer conj(rho) with w real, so it is
    Hermitian by construction; and the static density response of a gapped
    system is negative (density moves away from a raised potential), so it
    must also be negative semidefinite."""
    qs, chi0 = sc.static_polarizability(h, nk=NK)
    assert np.max(np.abs(chi0 - np.conj(np.transpose(chi0, (0, 2, 1))))) < 1e-12
    assert max(np.max(np.linalg.eigvalsh(c)) for c in chi0) < 1e-10


@pytest.mark.parametrize("name,h", _models())
def test_uniform_charge_mode_is_unscreened_at_q0(name, h):
    """At q=0 the form factor sums to sum_a rho^{nm}_a = <n|m> = delta_nm,
    which vanishes for every transition entering chi0 (they all change the
    occupation, so n != m). So chi0(0) annihilates the uniform vector,
    exactly: a rigid shift of every orbital energy cannot polarize
    anything. This is the cheapest test here and the one that catches an
    index or conjugation slip."""
    qs, chi0 = sc.static_polarizability(h, nk=NK)
    i0 = {qkey(q): i for i, q in enumerate(qs)}[qkey([0., 0., 0.])]
    ones = np.ones(chi0.shape[1], dtype=np.complex128)
    assert np.max(np.abs(chi0[i0] @ ones)) < 1e-12
    assert np.max(np.abs(ones @ chi0[i0])) < 1e-12


def test_reciprocity_holds_on_a_magnetic_model():
    """chi0(-q) = conj(chi0(q)), and therefore W(-q) = conj(W(q)).

    This is not decoration: kernel.build_blocks evaluates the antiresonant
    exchange block as conj(W(Q)) rather than transforming at -Q, so a
    screened W that did not obey this would silently break the finite-Q
    BSE. The identity follows from summing BOTH orderings of the (n,m)
    band pair -- the "twice one ordering" shortcut needs time-reversal
    symmetry -- so it is checked on a model with none: a Zeeman field
    plus a non-collinear exchange field."""
    h = gapped_ionic_chain()
    h.add_zeeman([0., 0., 0.3])
    h.add_exchange(lambda r: [0.15, 0.05, 0.25])
    V = density_interaction(h, U=0.6, V1=0.4)
    W = sc.screened_interaction(h, V=V, nk=NK)
    for i, j in _reverse(W, W.qs):
        assert np.max(np.abs(W.chi0[j] - np.conj(W.chi0[i]))) < 1e-12
        assert np.max(np.abs(W.Wq[j] - np.conj(W.Wq[i]))) < 1e-12


@pytest.mark.parametrize("name,h", _models())
def test_screened_interaction_is_hermitian(name, h):
    """W = v + v chi0 v + ... is Hermitian term by term"""
    W = sc.screened_interaction(h, V=density_interaction(h, U=0.6, V1=0.4),
                                nk=NK)
    assert np.max(np.abs(W.Wq - np.conj(np.transpose(W.Wq, (0, 2, 1))))) < 1e-12


def test_screened_interaction_matches_the_weak_coupling_series():
    """W = eps^-1 v = v + v chi0 v + O(v^3). Scaling the interaction down
    by a factor s scales the neglected remainder by s^3 while the term it
    is being compared against scales as s^2, so the ratio of the two must
    fall like s. This tests the inversion and the matrix ordering
    independently of chi0 itself."""
    h = gapped_ionic_chain()
    ratios = []
    for e2 in (0.4, 0.1, 0.025):  # each a quarter of the last
        W = sc.screened_interaction(h, V=density_interaction(h, Vr=_coulomb(e2)),
                                    nk=NK)
        first, resid = 0., 0.
        for i, q in enumerate(W.qs):
            vq = interaction_at_q(W.bare, h.geometry, q)
            first = max(first, np.max(np.abs(W.Wq[i] - vq)))
            resid = max(resid, np.max(np.abs(W.Wq[i] - (vq + vq @ W.chi0[i] @ vq))))
        ratios.append(resid / first)
    # the remainder shrinks relative to the first-order term, by about the
    # factor the interaction was scaled down by
    assert np.all(np.diff(ratios) < 0.)
    assert ratios[-1] < 0.02


# --- against an independent code path ---------------------------------

@pytest.mark.parametrize("iq", [0, 1, 3])
def test_polarizability_matches_the_independent_response_function(iq):
    """chitk/chiAB.py computes the same density-density response through
    completely different code (a frequency scan with a finite broadening,
    site-resolved rather than spin-orbital-resolved). Collapsing this
    module's spin-orbital chi0 onto sites must reproduce it at omega=0.

    The tolerance is loose on purpose: chiAB carries a finite delta, so
    the two agree in the delta -> 0 limit and not to machine precision."""
    from pyqula.chitk.chiAB import chiAB
    h = gapped_ionic_chain()
    nk = 12
    qs, chi0 = sc.static_polarizability(h, nk=nk)
    ns = len(h.geometry.r)
    mine = np.array([[np.sum(chi0[iq][2 * i:2 * i + 2, 2 * j:2 * j + 2])
                      for j in range(ns)] for i in range(ns)])
    _, chis = chiAB(h, mode="matrix", q=qs[iq], nk=nk,
                    energies=np.array([0.0]), delta=0.02)
    assert np.max(np.abs(mine - np.array(chis)[0])) < 1e-3


# --- the BSE built on top of it ---------------------------------------

@pytest.mark.parametrize("nsuper", [2, 4])
def test_supercell_folding_with_screening(nsuper):
    """The test that has historically caught every finite-Q bug in this
    BSE, now with the direct term screened. The supercell computes its own
    screening at Gamma over a coarser mesh; the base cell computes a
    different one over a finer mesh and is then evaluated at every Q that
    folds onto Gamma. That the two agree exercises the momentum bookkeeping
    of chi0, of W(q), and of the direct kernel's lookup into it."""
    def spectrum(ns, nk, Q):
        h = gapped_ionic_chain(nsuper=ns)
        V = density_interaction(h, U=1.0, V1=0.4)
        return np.sort(h.get_bse(V=V, Q=Q, nk=nk,
                                 screening="rpa").get_energies().real)
    sup = spectrum(nsuper, NK // nsuper, [0., 0., 0.])
    ref = np.sort(np.concatenate([spectrum(1, NK, [i / nsuper, 0., 0.])
                                  for i in range(nsuper)]))
    assert len(sup) == len(ref)
    assert np.max(np.abs(sup - ref)) < 1e-9


def test_screening_none_reproduces_the_bare_result():
    """Regression guard: the default must be bit-for-bit what it was
    before screening existed."""
    h = gapped_honeycomb(mass=1.0)
    V = density_interaction(h, Vr=_coulomb(0.6))
    a = h.get_bse(V=V, nk=6).get_energies()
    b = h.get_bse(V=V, nk=6, screening=None).get_energies()
    assert np.max(np.abs(a - b)) == 0.


def test_the_exchange_term_stays_bare_under_screening():
    """The standard GW-BSE split: the ladder is screened, the exchange
    (local-field) term keeps the bare interaction. Screening the latter
    would resum the same RPA bubbles twice, so switching screening on must
    leave a kernel="exchange" calculation completely untouched."""
    h = gapped_honeycomb(mass=1.0)
    V = density_interaction(h, Vr=_coulomb(0.6))
    bare = h.get_bse(V=V, nk=6, kernel="exchange").get_energies()
    with pytest.warns(UserWarning, match="no effect"):
        screened = h.get_bse(V=V, nk=6, kernel="exchange",
                             screening="rpa").get_energies()
    assert np.max(np.abs(bare - screened)) == 0.


def test_screening_changes_the_direct_term():
    """The complement of the test above: with the ladder switched on,
    screening must actually do something."""
    h = gapped_honeycomb(mass=1.0)
    V = density_interaction(h, Vr=_coulomb(0.6))
    bare = h.get_bse(V=V, nk=6).get_energies(1)[0].real
    screened = h.get_bse(V=V, nk=6, screening="rpa").get_energies(1)[0].real
    assert abs(bare - screened) > 1e-4


def test_crpa_screens_less_than_rpa():
    """Constrained RPA leaves the transitions inside the BSE window out of
    the polarization, so it has strictly fewer screening channels than the
    full RPA and its dielectric matrix must sit closer to unity."""
    h = _nondegenerate_chain()  # 4 bands, so nv=nc=1 is a genuine subset
    V = density_interaction(h, Vr=_coulomb(0.5))
    rpa = h.get_bse(V=V, nk=NK, nv=1, nc=1, screening="rpa")
    crpa = h.get_bse(V=V, nk=NK, nv=1, nc=1, screening="crpa")
    assert 1.0 > crpa.W.epsmin > rpa.W.epsmin


# --- guards -----------------------------------------------------------

def test_a_vacuous_crpa_window_is_refused():
    """With the default nv=nc=None the BSE window is the whole spectrum,
    so cRPA excludes every transition and chi0 would be identically zero.
    Silently returning the bare interaction there would be a trap."""
    h = gapped_honeycomb(mass=1.0)
    with pytest.raises(ValueError, match="excludes every"):
        h.get_bse(V=density_interaction(h, Vr=_coulomb(0.6)), nk=6,
                  screening="crpa")


def test_an_rpa_divergence_is_reported():
    """eps(q) reaching zero is a charge/spin instability of the mean field
    at that wavevector -- the same 1 - V chi = 0 condition chitk.rpa
    reports as a collective mode -- and W is infinite there. It must raise
    rather than return a huge number."""
    h = gapped_ionic_chain()
    with pytest.raises(ValueError, match="diverges"):
        h.get_screened_interaction(V=density_interaction(h, U=6., V1=3.),
                                   nk=NK)


def test_off_mesh_q_is_refused():
    """A tabulated W exists only on its mesh; snapping to the nearest
    point would be a silent wrong answer."""
    h = gapped_ionic_chain()
    W = h.get_screened_interaction(V=density_interaction(h, U=1.0), nk=NK)
    W.at(W.qs[1])  # a mesh point is fine
    with pytest.raises(ValueError, match="tabulated only"):
        W.at([0.077, 0., 0.])


def test_screened_interaction_round_trips_through_real_space():
    """get_dict() inverse Fourier transforms the tabulated W back to a
    real-space interaction. On the mesh the round trip is exact, and the
    result must be real -- W(-q) = conj(W(q)) is what makes it so."""
    h = gapped_ionic_chain()
    W = h.get_screened_interaction(V=density_interaction(h, U=0.6, V1=0.4),
                                   nk=NK)
    d = W.get_dict()
    for q in W.qs:
        assert np.max(np.abs(interaction_at_q(d, h.geometry, q) - W.at(q))) < 1e-10
    assert max(np.max(np.abs(m.imag)) for m in d.values()) < 1e-10


def test_a_denser_screening_mesh_must_be_a_multiple():
    h = gapped_ionic_chain()
    V = density_interaction(h, U=1.0)
    h.get_bse(V=V, nk=4, screening="rpa", nkW=8)  # a multiple is fine
    with pytest.raises(ValueError, match="integer multiple"):
        h.get_bse(V=V, nk=4, screening="rpa", nkW=6)


def test_a_precomputed_screened_interaction_can_be_reused():
    """Building W once and handing it to several BSE calls (a Q scan, say)
    must give the same answer as letting each build its own."""
    h = gapped_ionic_chain()
    V = density_interaction(h, U=1.0, V1=0.4)
    W = h.get_screened_interaction(V=V, nk=NK)
    a = h.get_bse(V=V, nk=NK, screening="rpa").get_energies()
    b = h.get_bse(V=V, nk=NK, screening=W).get_energies()
    assert np.max(np.abs(a - b)) < 1e-12


# --- a known limitation, pinned so it cannot change silently -----------

def test_screening_breaks_spin_rotation_invariance():
    """The RPA in this density-density representation is NOT spin-rotation
    invariant, and this test records that rather than asserting it away.

    On a non-magnetic reference the Bloch states are spin-diagonal, so
    chi0 is proportional to the identity in spin. The bare interaction's
    spin structure is spanned by {1, sigma_x} (a site pair couples every
    spin combination equally; a site's own block is the up-down-only
    Hubbard term), and that algebra is commutative, so eps and W stay
    inside it -- but with the same-spin and opposite-spin entries now
    DIFFERENT, where the bare interaction had them equal. Splitting
    W into those two parts,

        A n_iu n_ju + B n_iu n_jd + ... = (A+B)/2 n_i n_j + 2(A-B) Sz_i Sz_j

    so the screened interaction carries an Ising Sz-Sz coupling. That is
    not SU(2) invariant, and the visible consequence is that the lowest
    spin multiplet of the exciton spectrum splits.

    The fix is the standard GW one -- build the dielectric matrix in the
    charge channel alone, over site indices, and left-multiply the bare
    interaction by it so that its spin structure is untouched. That is a
    different approximation, not a bug fix, and is deliberately not
    implemented here. Until it is, prefer spinless models (or read the
    spin structure of the result with this in mind)."""
    h = gapped_honeycomb(mass=1.0)  # spin-rotation invariant reference
    V = density_interaction(h, U=1.0, Vr=_coulomb(0.6))
    W = h.get_screened_interaction(V=V, nk=6)
    vq, Wq = interaction_at_q(W.bare, h.geometry, W.qs[5]), W.at(W.qs[5])
    # the bare interaction couples both spin combinations of a site pair
    # equally; the screened one no longer does
    assert abs(vq[0, 2] - vq[0, 3]) < 1e-12
    assert abs(Wq[0, 2] - Wq[0, 3]) > 1e-3
    # and the lowest multiplet, four-fold degenerate with the bare
    # interaction, splits
    bare = h.get_bse(V=V, nk=6).get_energies(4).real
    screened = h.get_bse(V=V, nk=6, screening="rpa").get_energies(4).real
    assert bare[3] - bare[0] < 1e-10
    assert screened[3] - screened[0] > 1e-3
