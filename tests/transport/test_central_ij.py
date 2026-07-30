import numpy as np

from pyqula import geometry


def _finite_chain(n, has_spin=False):
    """A finite (dimensionality=0) open chain, built the same way
    geometry.dimer() turns a periodic cluster into a finite one: take a
    periodic supercell and flag it as dimensionality=0 so .intra alone
    is the whole (open, no wraparound) Hamiltonian."""
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=has_spin)


def test_identical_chain_segment_gives_perfect_transmission():
    """Gluing a finite chain segment identical to the leads back between
    two copies of that same lead must reproduce a pristine, gapless 1D
    chain: transmission 1 everywhere inside the band."""
    lead = geometry.chain().get_hamiltonian(has_spin=False)
    hc = _finite_chain(6)
    n = hc.intra.shape[0]
    ht = hc.get_central_heterostructure(0, n-1, left=lead, right=lead)
    ht.delta = 1e-6
    es = np.linspace(-1.8, 1.8, 15)  # strictly inside the [-2,2] chain band
    Ts = [ht.landauer(e) for e in es]
    assert np.allclose(Ts, 1.0, atol=1e-3)


def test_landauer_and_didv_agree():
    lead = geometry.chain().get_hamiltonian(has_spin=False)
    hc = _finite_chain(6)
    n = hc.intra.shape[0]
    ht = hc.get_central_heterostructure(0, n-1, left=lead, right=lead)
    ht.delta = 1e-6
    for e in [0.3, 1.1, -0.7]:
        assert abs(ht.landauer(e)-ht.didv(energy=e)) < 1e-3


def test_transmission_is_reciprocal_under_swapping_sites():
    """Swapping which site the left/right lead attaches to must not
    change the transmission (time-reversal/reciprocity)."""
    lead = geometry.chain().get_hamiltonian(has_spin=False)
    hc = _finite_chain(6)
    ht_ij = hc.get_central_heterostructure(1, 4, left=lead, right=lead)
    ht_ji = hc.get_central_heterostructure(4, 1, left=lead, right=lead)
    ht_ij.delta = ht_ji.delta = 1e-6
    assert abs(ht_ij.landauer(0.5)-ht_ji.landauer(0.5)) < 1e-8


def test_default_leads_and_default_j():
    """With no leads/no j given, a plain spinless chain lead is used and
    j defaults to the last site -- must not raise and must give a sane
    (between 0 and number of channels) transmission."""
    hc = _finite_chain(4)
    ht = hc.get_central_heterostructure(0)
    ht.delta = 1e-6
    T = ht.landauer(0.2)
    assert 0.0 <= T <= 1.0+1e-6


def test_site_index_out_of_range_raises():
    hc = _finite_chain(4)
    lead = geometry.chain().get_hamiltonian(has_spin=False)
    try:
        hc.get_central_heterostructure(0, 10, left=lead, right=lead)
        assert False, "expected a ValueError for an out-of-range site index"
    except ValueError:
        pass


def test_lead_hamiltonians_are_exposed_as_Hl_Hr():
    """Heterostructure.Hl/Hr (the full lead Hamiltonian objects, not just
    their .intra/.inter matrices) must be set, exactly as
    heterostructures.build's own constructors do -- several existing
    Heterostructure methods (get_kappa, surface_dos, get_dos, didv's SC
    auto-detection) read ht.Hl/ht.Hr directly and raise AttributeError
    otherwise.

    Uses spinful leads: a plain spinless lead trips an unrelated,
    pre-existing bug in transporttk/kappa.py's generate_HT (it forces a
    zero-pairing "spinless Nambu" lead via setup_nambu_spinor(), and
    hamiltonians.py's remove_nambu() has no branch for that mode -- the
    same class of gap as the turn_nambu() one fixed in superconductivity.py
    for this feature, just in the sibling function, and reproducible with
    plain heterostructures.build(spinless_lead1, spinless_lead2).get_kappa()
    with no central-region involvement at all, so it's out of scope here)."""
    lead = geometry.chain().get_hamiltonian(has_spin=True)
    hc = _finite_chain(4, has_spin=True)
    ht = hc.get_central_heterostructure(0, 3, left=lead, right=lead)
    assert hasattr(ht, "Hl") and hasattr(ht, "Hr")
    ht.delta = 1e-6
    ht.get_kappa(energy=0.2) # must not raise AttributeError
    ht.surface_dos(energies=[0.2]) # must not raise AttributeError
    ht.get_dos(energies=[0.2]) # must not raise AttributeError
