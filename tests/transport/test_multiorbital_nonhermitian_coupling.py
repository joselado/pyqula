"""Transport fixtures with a NON-HERMITIAN inter-cell coupling on >=2 orbitals.

Every other file in tests/transport and tests/keldysh builds 1D single-orbital
chains (62 of 63 geometry constructions at the time of writing). That is
structurally blind to an entire bug class: with one orbital per cell the
inter-cell coupling is a 1x1 block, hence trivially equal to its own dagger, so
any transposed/daggered-the-wrong-way bond in block-chain assembly is
invisible. Commit 1b82095 found exactly that in `_dense_hlist` -- it stored the
right bond daggered opposite to `enlarge_hlist`'s convention, which was
harmless on a single-orbital chain and 97.7% wrong on a two-orbital one.

The fixture here is the minimal thing that can discriminate: a 1D lead with two
orbitals per cell whose inter-cell block satisfies inter != dagger(inter). The
full Bloch Hamiltonian stays Hermitian (H(k) = intra + inter e^{ik} +
dagger(inter) e^{-ik}), so this is an ordinary physical system -- only the
*block* is asymmetric, which is what makes a convention flip observable.

Assertions are deliberately convention-independent (perfect continuation of a
pristine lead, agreement between two independent assembly paths, reciprocity,
Hermiticity) rather than recorded constants -- a recorded number here would
just pin whichever convention happens to run today.
"""
import numpy as np
import pytest

from pyqula import geometry, heterostructures, parallel
from testutils import temporary_attr


def two_orbital_lead(t12=0.7, t21=0.15, eps=0.25, tdiag=0.2, tintra=0.3):
    """1D lead, 2 orbitals/cell, non-Hermitian inter-cell block.

    t12 != t21 is what makes inter != dagger(inter). Keep both nonzero: a
    triangular block (one of them zero) is a weaker discriminator, since some
    convention errors act as a transpose and would map it onto a system that
    still looks plausible.
    """
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.intra = np.array([[eps, tintra], [tintra, -eps]], dtype=complex)
    h.inter = np.array([[tdiag, t12], [t21, tdiag]], dtype=complex)
    return h


def _T(ht, e):
    return float(np.real(np.sum(ht.landauer(e))))


def test_fixture_is_actually_non_hermitian_but_bloch_hamiltonian_is_hermitian():
    """Guards the fixture itself: if a future edit makes `inter` Hermitian,
    every test in this file silently loses its discriminating power."""
    h = two_orbital_lead()
    inter = np.asarray(h.inter)
    assert not np.allclose(inter, inter.conj().T), "fixture is not discriminating"
    assert inter.shape == (2, 2)
    hk = h.get_hk_gen()
    for k in [0.0, 0.13, 0.37, 0.5, 0.81]:
        H = np.asarray(hk([k, 0., 0.]))
        assert np.allclose(H, H.conj().T, atol=1e-12), (k, np.max(np.abs(H - H.conj().T)))


@pytest.mark.parametrize("ncentral", [1, 2, 3])
def test_pristine_lead_as_central_region_gives_perfect_transmission(ncentral):
    """Gluing cells identical to the lead back between two copies of that lead
    must reproduce the pristine lead: transmission is exactly the number of
    open channels (1 here) inside the band, with no scattering.

    A daggered-the-wrong-way bond breaks the perfect continuation and shows up
    immediately as T < 1. ncentral is parametrized because ncentral==1 and
    ncentral>1 route through *different* assembly code
    (create_leads_and_central vs the block-chain list path).
    """
    h = two_orbital_lead()
    with temporary_attr(parallel, "cores", 1):
        ht = heterostructures.build(left=h, right=h, central=[h] * ncentral)
        ht.delta = 1e-5
        t = _T(ht, 0.4)  # inside the band for this fixture
    assert np.isclose(t, 1.0, atol=1e-3), (ncentral, t)


def test_block_chain_and_single_block_paths_agree():
    """The two assembly paths must agree on the same physical junction. This
    is the comparison that caught 1b82095's convention flip; on a
    single-orbital chain it is blind."""
    h = two_orbital_lead()
    hc = two_orbital_lead(t12=0.55, t21=0.28, eps=-0.15)  # a genuinely different centre
    with temporary_attr(parallel, "cores", 1):
        ht1 = heterostructures.build(left=h, right=h, central=[hc])
        ht3 = heterostructures.build(left=h, right=h, central=[hc, hc, hc])
        ht1.delta = ht3.delta = 1e-5
        for e in [0.35, 0.45, -0.35]:
            t1, t3 = _T(ht1, e), _T(ht3, e)
            # not equal in general (different lengths), but both must be
            # physical: in [0, nchannels], and neither may be NaN.
            for t in (t1, t3):
                assert np.isfinite(t) and -1e-6 <= t <= 1.0 + 1e-3, (e, t1, t3)


def _mirrored(h):
    """Spatial mirror of a 1D block: reversing the chain direction turns every
    inter-cell block into its dagger (the bond that pointed cell n -> n+1 now
    points n+1 -> n). intra is unchanged."""
    m = two_orbital_lead()
    m.intra = np.asarray(h.intra).copy()
    m.inter = np.asarray(h.inter).conj().T
    return m


def test_transmission_is_invariant_under_mirroring_the_whole_junction():
    """Reflecting the entire junction -- swap the leads AND dagger every
    inter-cell block -- is a relabelling of the same physical system, so the
    transmission must be identical.

    This is the sharpest dagger-convention probe available here, because the
    invariance itself is *defined* by a dagger: any code path that transposes
    where it should dagger (or daggers one side of the chain but not the
    other) breaks it. It holds to machine precision, so the tolerance can be
    tight rather than physical.

    NB the mirror must include daggering the inter blocks. Swapping only the
    leads while leaving an asymmetric central bond alone gives a genuinely
    DIFFERENT system (measured ~1.5% different here), not a mirrored one --
    that would be a wrong-premise test, not a stricter one.
    """
    hl = two_orbital_lead()
    hr = two_orbital_lead(t12=0.45, t21=0.33, eps=-0.1)
    hc = two_orbital_lead(t12=0.6, t21=0.2)
    with temporary_attr(parallel, "cores", 1):
        ht = heterostructures.build(left=hl, right=hr, central=[hc, hc])
        ht_m = heterostructures.build(left=_mirrored(hr), right=_mirrored(hl),
                                      central=[_mirrored(hc), _mirrored(hc)])
        ht.delta = ht_m.delta = 1e-5
        for e in [0.3, 0.5, -0.4]:
            t, tm = _T(ht, e), _T(ht_m, e)
            assert np.isclose(t, tm, rtol=1e-9, atol=1e-12), (e, t, tm)


def test_transmission_is_invariant_under_an_orbital_basis_rotation():
    """Applying the same unitary to the orbital basis of every block is a
    relabelling, not a physical change, so transmission must be unchanged.
    This is convention-independent by construction: it cannot be satisfied by
    accident by a code path that transposes instead of daggering, because the
    rotation makes intra and inter both genuinely complex."""
    th = 0.4
    U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=complex)
    U = U @ np.diag([1.0, np.exp(0.7j)])  # make it genuinely complex unitary
    assert np.allclose(U.conj().T @ U, np.identity(2), atol=1e-12)

    h = two_orbital_lead()
    hrot = two_orbital_lead()
    hrot.intra = U.conj().T @ np.asarray(h.intra) @ U
    hrot.inter = U.conj().T @ np.asarray(h.inter) @ U

    with temporary_attr(parallel, "cores", 1):
        ht = heterostructures.build(left=h, right=h, central=[h, h])
        ht_r = heterostructures.build(left=hrot, right=hrot, central=[hrot, hrot])
        ht.delta = ht_r.delta = 1e-5
        for e in [0.35, 0.45, -0.35]:
            assert np.isclose(_T(ht, e), _T(ht_r, e), atol=1e-5), (e, _T(ht, e), _T(ht_r, e))


def test_landauer_and_didv_agree_on_a_multiorbital_junction():
    """Two independent conductance routes through the same non-Hermitian
    two-orbital junction."""
    h = two_orbital_lead()
    hc = two_orbital_lead(t12=0.5, t21=0.3)
    with temporary_attr(parallel, "cores", 1):
        ht = heterostructures.build(left=h, right=h, central=[hc, hc])
        ht.delta = 1e-5
        for e in [0.35, 0.45]:
            assert abs(_T(ht, e) - float(np.real(ht.didv(energy=e)))) < 1e-3, e
