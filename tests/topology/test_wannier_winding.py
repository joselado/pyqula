"""Chern number and Z2 invariant from the winding of the hybrid Wannier
centers, the signed number of times they cross a fixed line over the whole
Brillouin zone, or its parity over half of it (topology.wannier_winding).

The Chern branch used to raise, with the crossing count behind it marked as
wrong, and z2_wannier_centers dropped full=True, so the full flow was never
returned. The Z2 used to count crossings of the largest gap between the
centers instead, and with several occupied bands the answer changed with the
resolution. The Chern number is checked against the plaquette sum of
mesh_chern, an independent evaluation of the same integer, and both
invariants against the fact that a supercell describes the same system."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology


def _haldane(t2, has_spin):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    return h


def _trivial():
    h = _haldane(0.1, False)
    h.add_sublattice_imbalance(1.0)
    return h


def _rashba_exchange():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0., 0., 0.3])
    return h


def _topological_superconductor():
    h = geometry.square_lattice().get_hamiltonian()
    h.add_rashba(0.5)
    h.add_zeeman([0., 0., 0.6])
    h.add_onsite(-3.5)
    h.add_swave(0.3)
    return h


CASES = {
    "haldane spinful": lambda: _haldane(0.2, True),
    "haldane spinful reversed": lambda: _haldane(-0.2, True),
    "haldane spinless": lambda: _haldane(0.2, False),
    "haldane spinless reversed": lambda: _haldane(-0.2, False),
    "trivial": _trivial,
    "rashba and exchange": _rashba_exchange,
    "topological superconductor": _topological_superconductor,
}


@pytest.mark.parametrize("name", list(CASES))
def test_wannier_winding_matches_the_plaquette_chern(name, tmp_path,
                                                     monkeypatch):
    monkeypatch.chdir(tmp_path)  # both routes write *.OUT files to cwd
    h = CASES[name]()
    cw = h.get_chern(integration="wannier", nk=20, nt=60)
    cg = h.get_chern(nk=30)
    assert isinstance(cw, int)
    assert abs(cg - round(cg)) < 1e-6
    assert cw == round(cg)


@pytest.mark.parametrize("n,nk", [(1, 60), (2, 60), (3, 100)])
def test_z2_does_not_depend_on_the_unit_cell(n, nk, tmp_path, monkeypatch):
    """A supercell of the Kane-Mele model is the same quantum spin Hall
    insulator, with the same single helical pair at each edge. The
    largest-gap count returned +1 for n=2 at nk=nt=60 and for n=3 at
    nk=nt=100 (and -1 for n=3 at 60)."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice().get_supercell(n)
    h = g.get_hamiltonian()
    h.add_kane_mele(0.1)
    assert topology.z2_invariant(h, nk=nk, nt=nk) == -1


def test_chern_winding_does_not_depend_on_the_unit_cell(tmp_path,
                                                       monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(0.2)
    assert h.get_chern(integration="wannier", nk=20, nt=60) == \
        _haldane(0.2, False).get_chern(integration="wannier", nk=20, nt=60)


def test_z2_is_the_parity_of_the_half_zone_winding(tmp_path, monkeypatch):
    """The crossing count over half of the zone is an integer whose parity
    is the invariant, in both phases of the Kane-Mele model"""
    monkeypatch.chdir(tmp_path)
    for mass, z2 in [(0., -1), (1.0, 1)]:
        h = geometry.honeycomb_lattice().get_hamiltonian()
        h.add_kane_mele(0.05)
        h.add_sublattice_imbalance(mass)
        w = topology.wannier_winding(h, nk=40, nt=40, full=False)
        assert (-1)**abs(w) == z2
        assert topology.z2_wannier_winding(h, nk=40, nt=40) == z2


def test_full_wannier_flow_covers_the_whole_brillouin_zone(tmp_path,
                                                          monkeypatch):
    """z2_wannier_centers(full=True) used to return the half flow of the Z2
    invariant regardless of full, and the half flow stopped short of the
    time-reversal-invariant momentum t=1/2 it has to end on"""
    monkeypatch.chdir(tmp_path)
    h = _haldane(0.2, False)
    half = topology.z2_wannier_centers(h, nk=10, nt=20)
    full = topology.z2_wannier_centers(h, nk=10, nt=20, full=True)
    assert np.isclose(np.max(half[0]), 0.5)  # ends on the TRIM t=1/2
    assert 0.9 < np.max(full[0]) < 1.  # the loop closes on t=0


def test_unknown_integration_lists_the_accepted_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="wannier"):
        _haldane(0.2, False).get_chern(integration="mesh")


def test_a_metal_has_no_wannier_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _haldane(0.1, False)
    h.shift_fermi(2.5)  # the Fermi energy inside the upper band
    with pytest.raises(ValueError, match="number of occupied states"):
        h.get_chern(integration="wannier", nk=10, nt=10)


def test_atomic_gauge_centers_are_the_orbital_positions(tmp_path,
                                                       monkeypatch):
    """With a large sublattice imbalance the occupied band sits on one
    site, so its hybrid Wannier center is that site: at the origin of the
    cell in the lattice gauge, at the fractional coordinate of the site in
    the atomic gauge, along either loop direction"""
    monkeypatch.chdir(tmp_path)
    from pyqula.topologytk.qgt import _orbital_fractions
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(20.)
    e, v = np.linalg.eigh(np.array(h.get_hk_gen()([0.1, 0.2, 0.])))
    occ = np.argmax(np.abs(v[:, 0])**2)  # the occupied orbital
    x = _orbital_fractions(h, 2)[occ]
    for (loop, pump) in [(0, 1), (1, 0)]:
        for gauge, x0 in [("lattice", 0.), ("atomic", x[loop])]:
            m = topology.wannier_centers(h, nk=30, nt=4, full=True,
                                         loop=loop, pump=pump, gauge=gauge)
            d = np.angle(np.exp(1j*(m[1] - 2*np.pi*x0)))  # distance mod 2 pi
            assert np.max(np.abs(d)) < 2*np.pi*2e-3, (loop, gauge, m[1])


def test_exchanging_loop_and_pump_reverses_the_chern_number(tmp_path,
                                                           monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _haldane(0.2, False)
    c = topology.wannier_winding(h, nk=20, nt=60)
    assert c == round(h.get_chern(nk=30))
    assert topology.wannier_winding(h, nk=20, nt=60, loop=1, pump=0) == -c


def test_wannier_flow_arguments_are_checked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _haldane(0.2, False)
    with pytest.raises(ValueError, match="different directions"):
        topology.wannier_centers(h, loop=0, pump=0)
    with pytest.raises(ValueError, match="different directions"):
        topology.wannier_centers(h, loop=2, pump=0)
    with pytest.raises(ValueError, match="three-dimensional"):
        topology.wannier_centers(h, kfix=0.5)
    with pytest.raises(ValueError, match="gauge"):
        topology.wannier_centers(h, gauge="cell")
