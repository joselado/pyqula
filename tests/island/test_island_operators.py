import numpy as np
import pytest

from pyqula import islands
from pyqula import spectrum
from pyqula import operators


@pytest.mark.slow
def test_local_operators_valley_envelope_matches_reference(tmp_path, monkeypatch):
    """Regression check for valley-projected envelope operator expectation
    values on a small honeycomb island (n=2 instead of 5.5) with Peierls
    flux and a sublattice imbalance: the summed expectation values must
    match the value recorded from a known-good run. Marked slow: the
    island size is already small (24 atoms) -- the runtime is dominated by
    fixed overhead, not island size."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=6, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.add_peierls(0.1)
    h.add_sublattice_imbalance(0.1)
    x = np.zeros(h.intra.shape[0])
    h.shift_fermi(x)
    ops = operators.get_envelop(h, sites=range(h.intra.shape[0]), d=0.6)
    fv = h.get_operator("valley")
    ops = [fv * o for o in ops]
    ys = spectrum.ev(h, operator=ops).real
    assert np.isclose(np.sum(ys), -0.7182101377163648, atol=1e-6)


@pytest.mark.slow
def test_valley_texture_real_space_vev_matches_reference(tmp_path, monkeypatch):
    """Regression check for the real-space valley expectation value on a
    small honeycomb island (n=3 instead of 8) with Peierls flux and a
    sublattice imbalance: the summed real-space VEV must match the value
    recorded from a known-good run. Marked slow: the island size is already
    small (42 atoms) -- the runtime is dominated by fixed overhead, not
    island size."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=3, nedges=6, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.add_peierls(0.05)
    h.add_sublattice_imbalance(.2)
    fv = h.get_operator("valley")
    ys = spectrum.real_space_vev(h, operator=fv)
    assert np.isclose(np.sum(ys), -0.8852949826564925, atol=1e-6)


def test_dos_in_site_bulk_vs_edge_matches_reference(tmp_path, monkeypatch):
    """Regression check for site-resolved DOS at a bulk vs. edge site of a
    Haldane-gapped honeycomb island, at a small island size (n=3 instead of
    11): the bulk and edge DOS sums must match the values recorded from a
    known-good run."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=3, nedges=6, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.05)
    ibulk = h.geometry.closest_index([0., 0., 0.])
    iedge = h.geometry.closest_index([-20., 0., 0.])
    opbulk = h.get_operator("site", index=ibulk)
    opedge = h.get_operator("site", index=iedge)
    (e_bulk, d_bulk) = h.get_dos(operator=opbulk, delta=0.02)
    (e_edge, d_edge) = h.get_dos(operator=opedge, delta=0.02)
    assert np.isclose(np.sum(d_bulk), 49.65597424256937, atol=1e-4)
    assert np.isclose(np.sum(d_edge), 49.750612624621795, atol=1e-4)


def test_multildos_atomic_projection_matches_reference(tmp_path, monkeypatch):
    """Regression check for get_multildos(projection="atomic") on a small
    honeycomb island (n=2 instead of 3): the total DOS written to
    MULTILDOS/DOS.OUT must match the value recorded from a known-good run."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3)
    h = g.get_hamiltonian()
    h.get_multildos(projection="atomic")
    dos = np.genfromtxt("MULTILDOS/DOS.OUT")
    assert np.isclose(np.sum(dos), 40231.97213545212, atol=1e-2)
