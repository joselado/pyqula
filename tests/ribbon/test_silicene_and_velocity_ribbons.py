import numpy as np
import pytest

from pyqula import geometry


def _resolved_bands(h, nk):
    """Band energies together with the <y>, <sz> and <velocity> expectation
    values. get_bands resolves one operator per call, so this makes three
    passes over the same k-path; the (k, band) ordering of the returned flat
    arrays is identical, which the caller asserts."""
    (k, e, y) = h.get_bands(nk=nk, operator="yposition")
    (k2, e2, sz) = h.get_bands(nk=nk, operator="sz")
    (k3, e3, v) = h.get_bands(nk=nk, operator=h.get_operator("velocity"))
    assert np.allclose(e, e2) and np.allclose(e, e3)
    return np.array(e), np.array(y), np.array(sz), np.array(v)


def _silicene_ribbon(ez, soc, n=4):
    """Buckled-honeycomb (silicene) ribbon. The two sublattices sit at
    opposite z, so a perpendicular electric field ez enters as a staggered
    onsite potential +-ez, which competes with the Kane-Mele SOC."""
    g = geometry.buckled_honeycomb_lattice()
    g = geometry.bulk2ribbon(g, n=n)
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: ez * np.sign(r[2]))
    if soc != 0.: h.add_kane_mele(soc)
    return h


@pytest.mark.slow
def test_silicene_ribbon_field_competes_with_soc(tmp_path, monkeypatch):
    """Silicene under a perpendicular field is the textbook competition
    between a staggered sublattice potential and Kane-Mele SOC: the field
    alone gaps the ribbon by exactly 2*ez, while the SOC (when it wins)
    refills that gap with helical edge states -- sign(<v>*<y>*<sz>) the
    same for every one of them, so the two counter-propagating edge
    branches carry opposite spin.

    Replaces recorded sum(e) and sum(c) constants. sum(e) is sum_k Tr H(k),
    which the staggered +-ez potential cancels out of exactly, and sum(c)
    is nk*Tr(sz) = 0; neither responds to ez, to the SOC or to the width.
    Marked slow: runtime here is dominated by fixed overhead (e.g. first-use
    JIT compilation), not the ribbon width."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    ez, soc = 0.2, 0.1

    # the field on its own: a staggered +-ez potential on a bipartite
    # lattice gaps the Dirac point by 2*ez, i.e. min|E| = ez
    (e0, y0, sz0, v0) = _resolved_bands(_silicene_ribbon(ez, 0.), nk=41)
    assert np.isclose(np.min(np.abs(e0)), ez, atol=1e-6)

    # with the SOC on top, the SOC wins here and puts edge states in the gap
    (e, y, sz, v) = _resolved_bands(_silicene_ribbon(ez, soc), nk=41)
    assert np.min(np.abs(e)) < 0.25 * ez  # well inside the 2*ez field gap
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 1.5)
    assert np.sum(edge) > 0
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) > 0.)

    # reversing the SOC reverses the helicity
    (e, y, sz, v) = _resolved_bands(_silicene_ribbon(ez, -soc), nk=41)
    edge = (np.abs(e) < 0.08) & (np.abs(y) > 1.5)
    assert np.sum(edge) > 0
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) < 0.)


def _kane_mele_zigzag_ribbon(soc, n=4):
    g = geometry.honeycomb_zigzag_ribbon(n)
    h = g.get_hamiltonian()
    if soc != 0.: h.add_kane_mele(soc)
    return h


def test_kane_mele_turns_the_zigzag_flat_edge_band_helical(tmp_path, monkeypatch):
    """A pristine zigzag ribbon has the well-known *flat* zero-energy edge
    band: its states sit on the edges but carry no velocity. Kane-Mele SOC
    turns that flat band into a pair of counter-propagating helical edge
    modes, so the edge states acquire a finite velocity whose sign is tied
    to the edge and the spin: sign(<v>*<y>*<sz>) is the same for all of
    them and reverses with the sign of the SOC.

    Replaces recorded sum(e) and sum(c) constants. sum(e) = sum_k Tr H(k) =
    0 on this bipartite lattice for any SOC and any width, and sum(c) for a
    full band structure is nk*Tr(velocity) = 0 whatever the Hamiltonian is
    -- the velocity operator the test names was never actually probed."""
    monkeypatch.chdir(tmp_path)
    soc = 0.2

    # pristine: the edge band is flat, so the edge states have no velocity
    (e0, y0, sz0, v0) = _resolved_bands(_kane_mele_zigzag_ribbon(0.), nk=61)
    edge0 = (np.abs(e0) < 0.1) & (np.abs(y0) > 1.5)
    assert np.sum(edge0) > 0
    assert np.max(np.abs(v0[edge0])) < 1e-8

    (e, y, sz, v) = _resolved_bands(_kane_mele_zigzag_ribbon(soc), nk=61)
    edge = (np.abs(e) < 0.1) & (np.abs(y) > 1.5)
    assert np.sum(edge) > 4
    assert np.max(np.abs(v[edge])) > 0.5  # now dispersive
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) > 0.)

    (e, y, sz, v) = _resolved_bands(_kane_mele_zigzag_ribbon(-soc), nk=61)
    edge = (np.abs(e) < 0.1) & (np.abs(y) > 1.5)
    assert np.sum(edge) > 4
    assert np.all(np.sign(v[edge] * y[edge] * sz[edge]) < 0.)
