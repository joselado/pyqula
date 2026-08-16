"""Li-Haldane correspondence (Li & Haldane, PRL 101, 010504 (2008)): the
k-resolved entanglement spectrum of a Chern insulator carries chiral
branches that flow across xi=0, in one-to-one correspondence with the edge
states of the same model, while a trivial insulator has a gapped
entanglement spectrum.

The cut used here is a ring of unit cells (see entanglement.py), so it has
TWO entanglement boundaries and the counting of mid-gap branches is 2|C|,
with C the Chern number computed independently by topology.chern.
"""

import numpy as np

from pyqula import geometry


def _haldane(t2=0.1, mass=0.0, has_spin=False):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    if t2 != 0.0:
        h.add_haldane(t2)
    if mass != 0.0:
        h.add_sublattice_imbalance(mass)
    return h


def _midgap_count(xi, window=0.5):
    """Number of entanglement levels sitting in the middle of the
    entanglement gap"""
    return int(np.sum(np.abs(xi) < window))


def test_chern_insulator_entanglement_spectrum_counts_two_per_chern(
        tmp_path, monkeypatch):
    """At the crossing momentum, the number of entanglement levels at
    xi=0 must equal 2|C| (|C| chiral branches per entanglement boundary,
    two boundaries), and those levels must be well separated from the
    bulk of the entanglement spectrum."""
    monkeypatch.chdir(tmp_path)  # topology.chern writes *.OUT files to cwd
    for has_spin, expected_chern in [(False, 1), (True, 2)]:
        h = _haldane(has_spin=has_spin)
        chern = h.get_chern(nk=10)
        assert abs(chern - expected_chern) < 1e-3
        ks, xis = h.get_entanglement_spectrum(nsuper=8, nk=41)
        i0 = int(np.argmin(np.abs(ks - 0.5)))  # the crossing, pinned by symmetry
        assert abs(ks[i0] - 0.5) < 1e-9  # the mesh really samples it
        midgap = np.sort(np.abs(xis[i0]))
        assert _midgap_count(xis[i0]) == 2 * expected_chern
        # the mid-gap levels are at xi=0, the next ones are far away
        assert midgap[2 * expected_chern - 1] < 1e-6
        assert midgap[2 * expected_chern] > 3.0


def test_trivial_insulator_has_a_gapped_entanglement_spectrum():
    """With no Haldane flux and a sublattice mass the Chern number is
    zero, and no entanglement level comes anywhere near xi=0 anywhere in
    the Brillouin zone."""
    h = _haldane(t2=0.0, mass=0.4)
    ks, xis = h.get_entanglement_spectrum(nsuper=8, nk=41)
    assert np.min(np.abs(xis)) > 0.5
    assert all(_midgap_count(xi) == 0 for xi in xis)


def test_entanglement_spectrum_flows_across_zero(tmp_path, monkeypatch):
    """The mid-gap branches are not flat: they sweep from the top to the
    bottom of the entanglement spectrum across the Brillouin zone, which
    is the spectral flow that mirrors the chiral edge dispersion. Away
    from the crossing the entanglement spectrum is gapped."""
    monkeypatch.chdir(tmp_path)
    h = _haldane()
    ks, xis = h.get_entanglement_spectrum(nsuper=8, nk=41)
    gaps = np.min(np.abs(xis), axis=1)  # entanglement gap at each momentum
    assert np.min(gaps) < 1e-6  # closes at the crossing
    assert np.max(gaps) > 3.0  # wide open away from it
    # the two branches span the whole entanglement spectrum: the smallest
    # positive level goes from ~0 at the crossing to a large value at the
    # zone center, monotonically in between
    positive = np.array([np.min(np.abs(xi)) for xi in xis])
    i0 = int(np.argmin(np.abs(ks - 0.5)))
    assert positive[0] > positive[i0 // 2] > positive[i0]
