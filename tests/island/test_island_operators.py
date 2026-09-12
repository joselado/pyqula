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
    fixed overhead, not island size.

    The reference had the wrong sign until spectrum.ev stopped contracting
    the density matrix transposed: the valley operator is purely
    imaginary, exactly the case where <A*> and <A> differ. +0.71821 is
    what an explicit sum over the occupied eigenstates gives."""
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
    assert np.isclose(np.sum(ys), 0.7182101377163648, atol=1e-6)


@pytest.mark.slow
def test_valley_texture_real_space_vev_matches_reference(tmp_path, monkeypatch):
    """Regression check for the real-space valley expectation value on a
    small honeycomb island (n=3 instead of 8) with Peierls flux and a
    sublattice imbalance: the summed real-space VEV must match an explicit
    sum of <psi|A|psi> over the occupied eigenstates. Marked slow: the
    island size is already small (42 atoms) -- the runtime is dominated by
    fixed overhead, not island size.

    The reference used to be a pinned -0.88529, which had the wrong sign:
    real_space_vev contracted the density matrix untransposed, so for the
    purely imaginary valley operator it evaluated <A*> rather than <A>.
    That is the same defect fbee7c9 fixed in spectrum.ev, and it survived
    in this sibling. The assertion is against a reference computed here
    rather than a literal, so it cannot silently regress in either
    direction."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=3, nedges=6, rot=0.0)
    h = g.get_hamiltonian(has_spin=False)
    h.add_peierls(0.05)
    h.add_sublattice_imbalance(.2)
    fv = h.get_operator("valley")
    ys = spectrum.real_space_vev(h, operator=fv)
    (es, ws) = h.get_eigenvectors()
    m = fv.get_matrix()
    ref = sum([np.conjugate(w).dot(m @ w) for (e, w) in zip(es, ws)
               if e < 0.]).real
    assert np.isclose(np.sum(ys), ref, atol=1e-6)
    assert ref > 0.  # the sign the untransposed contraction got wrong


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
    """get_multildos(projection="atomic") on a small honeycomb island
    (n=2 instead of 3).

    This used to pin sum(DOS.OUT) == 40231.97213545212, which was the
    un-normalized value: ldostk/atomicmultildos handed the eigenvalues to
    calculate_dos raw, without the 1/pi of the Lorentzian that
    dos.dos_kmesh applies, so the recorded constant locked the error in.
    The island is 0d, so only that half of the defect bit here -- there is
    no k-mesh to divide by as well.

    The invariant replacing it is that MULTILDOS/DOS.OUT is a density of
    states: the one h.get_dos computes from the same eigenvalues on the
    refined grid multi_ldos_tb builds internally. Its sum is 12806.23,
    which is the old constant divided by pi."""
    monkeypatch.chdir(tmp_path)
    g = islands.get_geometry(name="honeycomb", n=2, nedges=3)
    h = g.get_hamiltonian()
    energies = np.linspace(-2.0, 2.0, 100)
    delta = 0.05
    es2 = np.linspace(min(energies), max(energies), len(energies)*10)
    ref = h.get_dos(energies=es2, delta=delta, write=False)[1]
    h.get_multildos(projection="atomic", energies=energies, delta=delta)
    dos = np.genfromtxt("MULTILDOS/DOS.OUT").T
    assert np.max(np.abs(dos[0]-es2)) < 1e-12  # same energy grid
    assert np.max(np.abs(dos[1]-ref)) < 1e-10*np.max(np.abs(ref))
