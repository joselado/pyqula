import numpy as np

from pyqula import geometry


def _bands(h, nk=61, operator=None):
    """Band energies (and, optionally, an operator expectation value) along
    the ribbon k-path, as flat arrays with one entry per (k, band)."""
    if operator is None:
        (k, e) = h.get_bands(nk=nk)
        return np.array(k), np.array(e)
    (k, e, c) = h.get_bands(nk=nk, operator=operator)
    return np.array(k), np.array(e), np.array(c)


def _armchair_gap(n, nk=61):
    """Gap of a pristine armchair ribbon. The lattice is bipartite with no
    onsite term, so the spectrum is particle-hole symmetric and the gap is
    2*min|E| -- here we only need whether min|E| vanishes."""
    g = geometry.honeycomb_armchair_ribbon(n)
    h = g.get_hamiltonian()
    (k, e) = _bands(h, nk=nk)
    return np.min(np.abs(e))


def test_armchair_ribbon_metallic_when_width_is_3m_plus_2(tmp_path, monkeypatch):
    """An armchair graphene ribbon is metallic when its number of dimer
    lines is N = 3m+2 and semiconducting otherwise (Nakada et al., PRB 54,
    17954 (1996)). geometry.honeycomb_armchair_ribbon(n) puts 4n sites in
    the unit cell, i.e. N = 2n dimer lines, so n=10 (N=20=3*6+2) must be
    gapless while n=9 (N=18) and n=11 (N=22) must not be.

    This replaces a recorded sum(e) constant. sum(e) over a k-path is
    sum_k Tr H(k), which vanishes for *any* width on this bipartite
    lattice, so it pinned nothing about the ribbon."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    assert _armchair_gap(10) < 1e-8
    assert _armchair_gap(9) > 1e-2
    assert _armchair_gap(11) > 1e-2


def test_quantum_spin_hall_armchair_ribbon_has_helical_edge_states(tmp_path,
                                                                   monkeypatch):
    """A Kane-Mele armchair ribbon is a quantum-spin-Hall ribbon: the SOC
    fills the semiconducting gap of the pristine n=8 (N=16) ribbon with
    edge states, and those states are *helical* -- the branch running one
    way along a given edge carries one spin, the branch running the other
    way carries the other. The signature is that sign(<v> * <y> * <sz>) is
    the same for every in-gap edge state, and reverses with the sign of the
    Kane-Mele coupling.

    Replaces a recorded sum(e) constant, which is sum_k Tr H(k) = 0 for any
    Kane-Mele coupling and any width."""
    monkeypatch.chdir(tmp_path)
    lam = .1
    assert _armchair_gap(8) > 1e-2  # the pristine ribbon of this width is gapped

    def helicity(soc):
        g = geometry.honeycomb_armchair_ribbon(8)
        h = g.get_hamiltonian()
        h.add_kane_mele(soc)
        # the three expectation values come from three passes over the same
        # k-path, so the (k, band) ordering of the flat arrays is identical
        (k, e, y) = _bands(h, operator="yposition")
        (k2, e2, sz) = _bands(h, operator="sz")
        (k3, e3, v) = _bands(h, operator=h.get_operator("velocity"))
        assert np.allclose(e, e2) and np.allclose(e, e3)
        return e, y, sz, v

    (e, y, sz, v) = helicity(lam)
    # the SOC closes the pristine gap with edge states
    assert np.min(np.abs(e)) < 1e-2
    # in-gap states that actually sit on an edge (the ribbon half-width is
    # about 5.6, so |<y>| > 3 means edge-localized)
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 3.)
    assert np.sum(edge) > 4
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) > 0.)

    # reversing the Kane-Mele coupling reverses the helicity
    (e, y, sz, v) = helicity(-lam)
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 3.)
    assert np.sum(edge) > 4
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) < 0.)


def test_haldane_armchair_ribbon_has_chiral_edge_states(tmp_path, monkeypatch):
    """A Haldane armchair ribbon is a Chern-insulator ribbon: the
    second-neighbour flux fills the semiconducting gap of the pristine n=8
    (N=16) ribbon with edge states that are *chiral* -- both spins run the
    same way along a given edge, so sign(<v> * <y>) is the same for every
    in-gap edge state and reverses with the sign of the Haldane coupling.
    That sign is exactly what a spectrum-sum reference cannot see: it is
    the difference between Chern number +1 and -1.

    Replaces a recorded sum(e) constant, which is sum_k Tr H(k) = 0 for any
    Haldane coupling and any width."""
    monkeypatch.chdir(tmp_path)
    t2 = .1
    assert _armchair_gap(8) > 1e-2  # the pristine ribbon of this width is gapped

    def chirality(t):
        g = geometry.honeycomb_armchair_ribbon(8)
        h = g.get_hamiltonian(has_spin=True)
        h.add_haldane(t)
        (k, e, y) = _bands(h, operator="yposition")
        (k2, e2, v) = _bands(h, operator=h.get_operator("velocity"))
        assert np.allclose(e, e2)
        return e, y, v

    (e, y, v) = chirality(t2)
    assert np.min(np.abs(e)) < 1e-2  # the flux closes the pristine gap
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 3.)
    assert np.sum(edge) > 4
    assert np.all(np.sign(v[edge] * y[edge]) > 0.)

    (e, y, v) = chirality(-t2)
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 3.)
    assert np.sum(edge) > 4
    assert np.all(np.sign(v[edge] * y[edge]) < 0.)
