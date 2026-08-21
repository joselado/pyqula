import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction


def _seeded_chain(filling, direction=(0., 0., 1.)):
    """A chain with a small persistent exchange seed field plus a matching
    initial mean-field guess, following the same recipe
    tests/chi/test_magnon_bands.py uses for the honeycomb island: without
    a seed, the trivial (unmagnetized) paramagnetic solution is also a
    valid SCF fixed point, so the seed is what selects the magnetized one."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    v = np.array(direction)
    h.add_exchange(1e-2*v)
    mf = h.copy()
    mf.add_exchange(0.5*v)
    return h, mf


@pytest.mark.slow
def test_vjinteraction_v1_only_converges_ferromagnetic_at_low_filling():
    """A chain with ONLY a nearest-neighbor density-density interaction
    (V1, no onsite U, no explicit exchange J) must be able to spontaneously
    magnetize at low filling: V1's Fock (exchange) contribution to the
    Hartree-Fock mean field favours same-spin alignment on neighboring
    sites (the same mechanism behind extended-Hubbard-model itinerant
    ferromagnetism), and the divergent 1D density of states at the bottom
    of the band makes this instability easy to trigger there. This is the
    physical motivation for supporting non-onsite interactions in the RPA
    machinery at all."""
    h, mf = _seeded_chain(filling=0.1)
    scf = VJinteraction(h, V1=1.1, filling=0.1, mf=mf, nk=200, mix=0.2,
                         maxerror=1e-8, maxite=1000)
    assert scf.converged
    hmf = scf.hamiltonian
    mz = hmf.get_vev("sz")
    assert abs(mz[0]) > 0.05, f"expected a sizable ferromagnetic moment, got {mz}"
    # H.V picks up the neighbor-shell (non-onsite) keys -- this is exactly
    # the H.V shape that chitk.spinchi._require_onsite_only_V now rejects
    # (see test_magnon_bands_raises_on_v1_only_converged_hamiltonian below)
    assert len(hmf.V) > 1
    assert (0, 0, 0) in hmf.V


@pytest.mark.slow
def test_magnon_bands_raises_on_v1_only_converged_hamiltonian():
    """get_magnon_bands must raise ValueError on a Hamiltonian whose H.V
    comes from a folded-in density-density (V1) interaction -- i.e. a
    genuinely non-onsite H.V reached through the normal SCF path, not just
    a hand-built dict. Non-onsite spin-channel RPA (bond exchange alone,
    density-density alone, or a combination) is not yet properly verified
    against an independent reference, so chitk.spinchi._require_onsite_only_V
    rejects it rather than silently returning an unverified number -- see
    that function's docstring for the reasoning."""
    h, mf = _seeded_chain(filling=0.1)
    scf = VJinteraction(h, V1=1.1, filling=0.1, mf=mf, nk=200, mix=0.2,
                         maxerror=1e-8, maxite=1000)
    hmf = scf.hamiltonian
    energies = np.linspace(0.01, 1.0, 40)
    with pytest.raises(ValueError):
        hmf.get_magnon_bands(nq=3, energies=energies, delta=2e-2, nk=100)


@pytest.mark.slow
def test_vjinteraction_j1_ferromagnetic_moment_grows_with_coupling_strength():
    """Same physical check as tests/scf/test_spinspin_ferro_chain.py's
    test_szsz_ferromagnetic_moment_grows_with_coupling_strength (H = J1
    Sz_i Sz_j, J1<0 ferromagnetic), but going through VJinteraction (the
    engine behind get_mean_field_hamiltonian and the one whose H.V feeds
    magnon_bands) instead of SzSz directly, at half filling on the plain
    1-site-unit-cell chain."""
    g = geometry.chain()

    def mz(J1):
        h = g.get_hamiltonian(has_spin=True)
        scf = VJinteraction(h, J1=J1, mf="ferroZ", nk=100, maxerror=1e-8,
                             mix=0.3, maxite=500, filling=0.5)
        assert scf.converged
        return np.mean(np.abs(scf.hamiltonian.get_magnetization()[:, 2]))

    mz_weak = mz(-0.1)
    mz_strong = mz(-2.0)
    assert mz_strong > 5*mz_weak, \
        f"expected the moment to grow with |J1|: {mz_weak} vs {mz_strong}"
    assert mz_strong > 0.005, "no sizable ferromagnetic moment at strong coupling"


@pytest.mark.slow
def test_vjinteraction_j1_antiferromagnetic_moment_grows_on_bichain():
    """The antiferromagnetic sign (J1>0) needs a bipartite (2-site) unit
    cell to represent the staggered Neel order -- see
    test_spinspin_ferro_chain.py's docstring for why the 1-site chain
    cannot converge to it. This is also the more demanding test of the
    V2U_matrix generalization: the bichain's nearest-neighbor bond couples
    sublattice A to sublattice B, i.e. two DIFFERENT orbitals, so H.V's
    bond matrix has genuine off-diagonal (inter-orbital) structure that a
    diagonal-only extraction would silently read as zero -- confirmed
    directly in test_v2u_matrix_offdiagonal.py, exercised end-to-end here
    through the real SCF path."""
    g = geometry.bichain()

    def moments(J1):
        h = g.get_hamiltonian()
        scf = VJinteraction(h, J1=J1, mf="antiferro", nk=100, mix=0.3,
                             maxerror=1e-8, maxite=800, filling=0.5)
        assert scf.converged
        return scf.hamiltonian.get_magnetization()[:, 2], scf.hamiltonian

    m_weak, _ = moments(1.0)
    m_strong, h_strong = moments(3.0)
    # staggered (Neel) pattern: opposite sign on the two sublattices
    assert m_weak[0]*m_weak[1] < 0
    assert m_strong[0]*m_strong[1] < 0
    assert np.mean(np.abs(m_strong)) > 5*np.mean(np.abs(m_weak))

    # H.V must carry the off-diagonal, cross-sublattice bond structure
    assert len(h_strong.V) > 1

    # get_magnon_bands RUNS on this genuinely multi-orbital, non-onsite
    # H.V: an isotropic exchange interaction is one the spin RPA can build
    # a matching vertex for, now that the SCF records its three channels
    # separately in h.Vchannels. See
    # tests/chi/test_exchange_channels_rpa.py for the Goldstone
    # measurement that justifies letting it through, and
    # test_magnon_bands_raises_on_v1_only_converged_hamiltonian above for
    # the case that is still refused (a neighbor-shell density-density
    # interaction, whose Fock rung no site-separable vertex can carry).
    assert h_strong.Vchannels is not None
    energies = np.linspace(0.01, 3.0, 40)
    qs, ws, gammas = h_strong.get_magnon_bands(nq=3, energies=energies,
                                                delta=2e-2, nk=100)
    assert len(qs) == len(ws) == len(gammas)
