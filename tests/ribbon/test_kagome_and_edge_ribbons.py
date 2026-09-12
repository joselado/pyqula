import numpy as np

from pyqula import geometry
from pyqula import ribbon
from pyqula import multicell


def _particle_hole_symmetric(e):
    """A BdG spectrum comes in +-E pairs, so the sorted spectrum must equal
    minus its own reverse. This is the identity that made the old sum(e)
    reference zero; asserting it directly says what the old reference only
    implied."""
    e = np.sort(np.array(e))
    return np.allclose(e, -e[::-1], atol=1e-8)


def _kagome_frustration_ribbon(mm, delta=0.3, n=6):
    """Kagome ribbon whose lower half carries a 120-degree (frustrated)
    three-sublattice texture of magnitude mm and whose upper half is an
    s-wave superconductor of gap delta."""
    g = geometry.kagome_lattice()
    g.has_sublattice = True
    g.sublattice = [-1, 1, 0]
    g = ribbon.bulk2ribbon(g, n=n)
    h = g.get_hamiltonian()
    m1 = np.array([1., 0., 0.])
    m2 = np.array([-.5, np.sqrt(3.) / 2., 0.])
    m3 = np.array([-.5, -np.sqrt(3.) / 2., 0.])
    ms = []
    for (r, s) in zip(g.r, g.sublattice):
        if r[1] < 0.0:
            if s == -1: ms.append(m1 * mm)
            if s == 1: ms.append(m2 * mm)
            if s == 0: ms.append(m3 * mm)
        else:
            ms.append([0., 0., 0.])

    def fs(r):
        if r[1] > 0.0: return delta
        else: return 0.0

    h.add_magnetism(ms)
    h.add_swave(fs)
    h.shift_fermi(fs)
    return h


def test_kagome_frustration_ribbon_binds_states_at_the_magnet_sc_interface(
        tmp_path, monkeypatch):
    """A 120-degree frustrated kagome magnet in contact with an s-wave
    superconductor binds subgap states *at the interface between the two
    halves*, which is what the "interface" operator is there to show.

    The old assertions were sum(e) and sum(c) over the full band structure.
    Both are Hamiltonian-independent: sum_n e_n(k) = Tr H(k) = 0 here, and
    sum_n <n|O|n> = Tr O, so sum(c) is just nk*Tr(interface) = 400*24 =
    9600 whatever the moments, the pairing or the ribbon width are. What
    actually depends on the texture is that subgap states exist at all and
    that they sit on the interface: with no moments the magnetic half is a
    normal metal whose low-energy states are spread over it (interface
    weight ~0.4), and with moments far above the bandwidth the subgap
    window empties out entirely."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    mm = 3.0
    h = _kagome_frustration_ribbon(mm)
    # nk=31 instead of the 400-point default: this is a localization
    # statement about the subgap states, not a k-resolved one
    (k, e, c) = h.get_bands(nk=31, operator="interface")
    e, c = np.array(e), np.array(c)
    assert _particle_hole_symmetric(e)

    subgap = np.abs(e) < 0.15
    assert np.sum(subgap) > 0
    assert np.max(c[subgap]) > 0.9  # bound to the interface, not spread out

    # with no texture the magnetic half stays a gapless normal metal and its
    # low-energy states are not interface states
    h0 = _kagome_frustration_ribbon(0.0)
    (k, e0, c0) = h0.get_bands(nk=31, operator="interface")
    e0, c0 = np.array(e0), np.array(c0)
    assert np.max(c0[np.abs(e0) < 0.15]) < 0.6


def _rashba_zeeman_sc_ribbon(bz, n=15):
    """Triangular-lattice Rashba superconductor with an out-of-plane
    Zeeman field -- the standard recipe for a two-dimensional topological
    superconductor -- cut into a ribbon."""
    g = geometry.triangular_lattice()
    h = g.get_hamiltonian()
    h.add_rashba(1.0)
    h.add_zeeman([0., 0., bz])
    h.add_onsite(-6.0)
    h.add_swave(0.4)
    return multicell.bulk2ribbon(h, n=n)


def _edge_weight(y, ld):
    """Fraction of a ribbon LDOS carried by the outer quarter of the sites
    (by transverse coordinate). A uniform distribution gives 0.25."""
    o = np.argsort(np.array(y))
    ld = np.array(ld)[o]
    m = len(ld) // 8
    return (ld[:m].sum() + ld[-m:].sum()) / ld.sum()


def test_topological_sc_ribbon_has_zero_energy_edge_states(tmp_path, monkeypatch):
    """Rashba + out-of-plane Zeeman + s-wave pairing makes a topological
    superconductor, so the ribbon carries Majorana edge modes: the bulk
    stays gapped, a handful of states sit at E=0, and the zero-energy LDOS
    piles up on the two edges. Without the Zeeman field the same ribbon is
    a trivial superconductor -- fully gapped, with essentially no weight at
    E=0 and none of it on the edges.

    Replaces sum(eb) and sum(cb) references. sum(eb) is sum_k Tr H(k) = 0
    for a BdG Hamiltonian regardless of Rashba, Zeeman, pairing or width,
    and sum(cb) is nk*Tr(yposition) = 0 because the ribbon is centred on
    y=0 -- neither sees the model."""
    monkeypatch.chdir(tmp_path)
    hr = _rashba_zeeman_sc_ribbon(1.0)
    (kb, eb, cb) = hr.get_bands(nk=41, operator="yposition")
    eb = np.array(eb)
    assert _particle_hole_symmetric(eb)
    # the only states in the trivial gap are the edge modes: few of them,
    # and at E=0
    assert np.min(np.abs(eb)) < 0.05
    assert 0 < np.sum(np.abs(eb) < 0.05) < 10
    (x, y, ld) = hr.get_ldos(e=0.0, delta=1e-3, nk=20, nrep=5)
    assert np.sum(ld) > 0.5
    assert _edge_weight(y, ld) > 0.45  # 0.25 would be uniform

    # the trivial (no Zeeman) superconductor: gapped, no zero-energy weight
    hr0 = _rashba_zeeman_sc_ribbon(0.0)
    (kb, eb0, cb0) = hr0.get_bands(nk=41, operator="yposition")
    assert np.min(np.abs(np.array(eb0))) > 0.2
    (x, y, ld0) = hr0.get_ldos(e=0.0, delta=1e-3, nk=20, nrep=5)
    assert np.sum(ld0) < 0.2
