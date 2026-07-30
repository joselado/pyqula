import numpy as np

from pyqula import geometry
from pyqula import heterostructures


def _finite_chain(n, has_spin=True):
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=has_spin)


def test_central_heterostructure_matches_build_for_ns_junction():
    """A one-site-wide central region attached to a normal lead on one
    side and a superconducting lead (add_swave) on the other must
    reproduce heterostructures.build(h_normal, h_sc)'s own Andreev
    conductance -- both describe the exact same physical NS junction
    through two independent code paths (build's own two-lead constructor
    vs. get_central_heterostructure's bare-Heterostructure-plus-coupling
    construction)."""
    g = geometry.chain()
    h_normal = g.get_hamiltonian()
    h_sc = g.get_hamiltonian()
    h_sc.add_swave(0.3)

    HTref = heterostructures.build(h_normal, h_sc)
    HTref.delta = 1e-4

    hc = _finite_chain(1)
    ht = hc.get_central_heterostructure(0, 0, left=h_normal, right=h_sc)
    ht.delta = 1e-4

    assert ht.has_eh
    for e in [1e-4, 0.1, 0.5]:
        assert abs(ht.didv(energy=e)-HTref.didv(energy=e)) < 1e-3


def test_central_heterostructure_matches_build_with_multisite_center():
    """Same cross-check, but with a genuine multi-site central region (the
    normal lead attaches at site 0, the superconducting lead at the last
    site of a 3-site normal segment) -- exercises the rectangular,
    zero-padded coupling embedding rather than the degenerate single-site
    case above."""
    g = geometry.chain()
    h_normal = g.get_hamiltonian()
    h_sc = g.get_hamiltonian()
    h_sc.add_swave(0.3)

    HTref = heterostructures.build(h_normal, h_sc)
    HTref.delta = 1e-4

    hc = _finite_chain(3)
    nsites = 3
    ht = hc.get_central_heterostructure(0, nsites-1, left=h_normal, right=h_sc)
    ht.delta = 1e-4

    for e in [1e-4, 0.1, 0.5]:
        assert abs(ht.didv(energy=e)-HTref.didv(energy=e)) < 1e-3


def test_two_superconducting_sources_raises():
    """At most one of {central Hamiltonian, left lead, right lead} may
    carry actual pairing -- get_reflection_normal_lead has no way to pick
    a normal lead otherwise, so this must fail loudly and early rather
    than deep inside didv_BdG."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(); h1.add_swave(0.2)
    h2 = g.get_hamiltonian(); h2.add_swave(0.3)
    hc = _finite_chain(3)
    try:
        hc.get_central_heterostructure(0, 2, left=h1, right=h2)
        assert False, "expected a ValueError for two superconducting sources"
    except ValueError:
        pass


def test_superconducting_central_region_with_two_normal_leads():
    """The pairing source can also be the central region itself (a short
    superconducting segment) rather than either lead -- must not raise
    and must yield a subgap Andreev conductance within the BTK bound of
    2 per electron channel.

    Note: even though every input here (both leads and the center) is
    spinless, htk.mode.make_compatible's turn_nambu() has no way to
    produce a purely spinless Nambu Hamiltonian from a plain spinless one
    (its "spinless" branch always calls turn_spinful() first -- the same
    thing heterostructures.build's own pairwise make_compatible calls do
    in this situation, see test_central_heterostructure_matches_build_*
    above using spinful leads from the start) -- so this ends up promoted
    to 2 (spin) electron channels, not 1."""
    g = geometry.chain()
    h_normal = g.get_hamiltonian(has_spin=False)
    hc = _finite_chain(3, has_spin=False)
    hc.add_swave(0.3)
    ht = hc.get_central_heterostructure(0, 2, left=h_normal, right=h_normal)
    ht.delta = 1e-4
    assert ht.has_eh
    n_electron_channels = ht.left_intra.shape[0]//2
    G = ht.didv(energy=0.05)  # well inside the gap
    assert 0.0 <= G <= 2.0*n_electron_channels+1e-6
