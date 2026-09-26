"""The nested Wilson loop: the polarization of a sector of Wannier bands and
the quadrupole moment (topologytk/nestedwilson.py).

The Benalcazar-Bernevig-Hughes model is specialhamiltonian.square_2OTI,
four orbitals with pi flux, intracell hopping 1-delta and intercell 1+delta,
which is a quadrupole insulator for delta > 0. The mirrors quantize every
sector polarization to 0 or 1/2 there, so those values only say that the
answer is right up to a sign and an offset; the sign and the offset are
fixed by an atomic limit, whose sector polarizations are the orbital
positions, and by a model without mirrors, whose non-quantized values are
compared with an independent implementation. The nested loop over all the
occupied states is the ordinary Wilson loop, which the existing
wannier_centers computes by another route."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhamiltonian
from pyqula import topology
from pyqula.multihopping import MultiHopping
from pyqula.topologytk import nestedwilson


def _distance(a, b):
    """Distance between two polarizations defined modulo 1"""
    return abs((a - b + 0.5) % 1. - 0.5)


def _bbh(gx, gy, lx=1., ly=1.):
    """Benalcazar-Bernevig-Hughes model with intracell hoppings gx, gy and
    intercell hoppings lx, ly along x and y, built from square_2OTI by
    scaling each bond by its direction and by whether it leaves the cell"""
    h = specialhamiltonian.square_2OTI(delta=0.)
    g = h.geometry
    r = np.array(g.r)
    A = np.array([g.a1, g.a2, g.a3])
    out = {}
    for (R, m) in h.get_multihopping().dict.items():
        m = np.array(m, dtype=complex)
        for (i, j) in zip(*np.nonzero(np.abs(m) > 1e-9)):
            dr = r[j] + np.array(R)@A - r[i]
            intra = tuple(R) == (0, 0, 0)
            if abs(dr[1]) < 1e-6: m[i, j] *= gx if intra else lx
            else: m[i, j] *= gy if intra else ly
        out[R] = m
    h.set_multihopping(MultiHopping(out))
    return h


def _mirror_broken():
    """The quadrupole insulator with onsite energies and intracell hoppings
    that break both mirrors, so that no polarization is quantized"""
    h = specialhamiltonian.square_2OTI(delta=0.3)
    p = np.zeros((4, 4), dtype=complex)
    p[0, 1] = 0.1 + 0.05j
    p[2, 3] = -0.08j
    p[0, 3] = 0.07
    p[1, 2] = 0.04 - 0.03j
    h.intra = h.intra + p + p.conj().T + np.diag([0.2, -0.1, 0.05, -0.15])
    return h


def _atomic_limit():
    """Four uncoupled orbitals, the two occupied ones at the fractional
    positions (0.2,0.3) and (0.7,0.6)"""
    g = geometry.square_lattice().get_supercell((2, 2))
    h = g.get_hamiltonian(has_spin=False)
    h.set_multihopping(MultiHopping({(0, 0, 0): np.diag([-1., -1., 1., 1.])}))
    frac = np.array([[0.2, 0.3], [0.7, 0.6], [0.5, 0.1], [0.1, 0.8]])
    h.geometry.r = np.array([f[0]*g.a1 + f[1]*g.a2 for f in frac])
    return h


@pytest.mark.parametrize("delta", [0.3, 0.6])
@pytest.mark.parametrize("nk", [20, 31])
def test_quadrupole_phase(delta, nk):
    h = specialhamiltonian.square_2OTI(delta=delta)
    for loop in (0, 1):
        for sector in ("+", "-"):
            p = h.get_wannier_sector_polarization(loop=loop, sector=sector,
                    nk=nk)
            assert abs(p - 0.5) < 1e-8
    assert abs(h.get_quadrupole_moment(nk=nk) - 0.5) < 1e-8


@pytest.mark.parametrize("nk", [20, 31])
def test_trivial_phase(nk):
    h = specialhamiltonian.square_2OTI(delta=-0.3)
    for loop in (0, 1):
        for sector in ("+", "-"):
            p = topology.wannier_sector_polarization(h, loop=loop,
                    sector=sector, nk=nk)
            assert abs(p) < 1e-8
    assert abs(topology.quadrupole_moment(h, nk=nk)) < 1e-8


def test_the_four_classes_of_wannier_bands():
    """With lx=ly=1, a direction is dimerized the topological way when its
    intracell hopping is the weaker one, and only when both are is the
    quadrupole moment 1/2 (arXiv:1708.04230, figs. 25 and 29): with only y
    topological the x sectors sit between cells along y, p_y = 1/2, while
    the y sectors do not, p_x = 0"""
    cases = [((0.5, 0.5), 0.5, 0.5, 0.5), ((1.25, 0.25), 0.5, 0., 0.),
             ((0.25, 1.25), 0., 0.5, 0.), ((1.25, 1.25), 0., 0., 0.)]
    for ((gx, gy), py, px, q) in cases:
        h = _bbh(gx, gy)
        for sector in ("+", "-"):
            assert _distance(topology.wannier_sector_polarization(h,
                    loop=0, sector=sector, nk=24), py) < 1e-8
            assert _distance(topology.wannier_sector_polarization(h,
                    loop=1, sector=sector, nk=24), px) < 1e-8
        assert _distance(topology.quadrupole_moment(h, nk=24), q) < 1e-8


def test_sector_polarizations_are_positions_in_the_atomic_gauge():
    """The sector with its Wannier center in (0,1/2) along x holds the
    orbital at (0.2,0.3), whose y coordinate is its polarization, and the
    other one the orbital at (0.7,0.6); this fixes the sign and the origin"""
    h = _atomic_limit()
    expected = {(0, "+"): 0.3, (0, "-"): 0.6, (1, "+"): 0.2, (1, "-"): 0.7}
    for ((loop, sector), p0) in expected.items():
        p = topology.wannier_sector_polarization(h, loop=loop, sector=sector,
                nk=12, gauge="atomic")
        assert abs(p - p0) < 1e-10


@pytest.mark.parametrize("gauge", ["lattice", "atomic"])
@pytest.mark.parametrize("loop", [0, 1])
def test_nested_loop_over_all_states_is_the_wilson_loop(gauge, loop, tmp_path,
                                                       monkeypatch):
    """Keeping every Wannier band, the nested loop along the other direction
    is the ordinary Wilson loop there, whose phase is the sum of the hybrid
    Wannier centers that wannier_centers returns at the same momentum"""
    monkeypatch.chdir(tmp_path)  # wannier_centers writes WANNIER_CENTERS.OUT
    h = _mirror_broken()
    nk = 24
    (s, d) = nestedwilson._occupied_grid(h, nk, None, gauge)
    (p, nu) = nestedwilson._sector_polarizations(s, d, loop,
            lambda nu: np.ones(nu.shape, dtype=bool))
    m = topology.wannier_centers(h, nk=nk+1, nt=nk, full=True, loop=1-loop,
            pump=loop, gauge=gauge)
    ref = np.sum(m[1:], axis=0)/(2.*np.pi)
    assert np.max([_distance(a, b) for (a, b) in zip(p, ref)]) < 1e-10


def test_wilson_loop_eigenvalues_do_not_depend_on_the_base_point(tmp_path,
                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _mirror_broken()
    (s, d) = nestedwilson._occupied_grid(h, 24, None, "lattice")
    (nu, v) = nestedwilson._wannier_bands(s, d[0])
    nu = np.sort(nu, axis=2)
    assert np.max(np.abs(nu - nu[0:1])) < 1e-12
    m = topology.wannier_centers(h, nk=25, nt=24, full=True, loop=0, pump=1)
    ref = np.sort(m[1:].T/(2.*np.pi), axis=1)
    assert np.max(np.abs(ref - nu[0])) < 1e-12


def test_matches_an_independent_implementation():
    """Without mirrors the sector polarizations are not quantized. The
    reference values are from nestedWilsonLib (github.com/kuansenlin/
    nested_and_spin_resolved_Wilson_loop, GPL-3, on PythTB 1.7.2) for the
    same model, orbitals at the origin and a 41x41 grid, with the nested
    phase averaged over the 40 base points along the loop; it agreed with
    this implementation to 9e-6 at every base point, the difference being
    that its links are not made unitary"""
    h = _mirror_broken()
    reference = {(0, "+"): 0.4977461058959559, (0, "-"): 0.4723004688699173,
                 (1, "+"): 0.5058689660326733, (1, "-"): 0.481525502229466}
    for ((loop, sector), p0) in reference.items():
        p = topology.wannier_sector_polarization(h, loop=loop, sector=sector,
                nk=40)
        assert _distance(p, p0) < 1e-4


def test_sector_polarizations_add_up_to_the_polarization():
    """The Berry connection of all the occupied states traces over both
    sectors, so p^+ + p^- is the total polarization, up to the
    discretization of the links (eq. VI.43 of arXiv:1708.04230)"""
    h = _mirror_broken()
    (s, d) = nestedwilson._occupied_grid(h, 40, None, "lattice")
    for loop in (0, 1):
        p = [nestedwilson._average(nestedwilson._sector_polarizations(s, d,
                loop, select)[0]) for select in (nestedwilson._sectors["+"],
                nestedwilson._sectors["-"],
                lambda nu: np.ones(nu.shape, dtype=bool))]
        assert _distance(p[0] + p[1], p[2]) < 1e-3


def test_closing_of_the_wannier_gap_raises():
    """At delta=-1 the plaquettes decouple: the bulk gap stays open, but
    every Wannier center sits at 0, so the two sectors cannot be told apart"""
    h = specialhamiltonian.square_2OTI(delta=-1.)
    assert topology.wannier_gap(h, nk=16) < 1e-12
    with pytest.raises(ValueError, match="Wannier"):
        topology.wannier_sector_polarization(h, nk=16)
    with pytest.raises(ValueError, match="Wannier"):
        topology.quadrupole_moment(h, nk=16)


def test_bad_arguments_raise():
    h = specialhamiltonian.square_2OTI(delta=0.3)
    with pytest.raises(ValueError, match=r"\['\+', '-'\]"):
        topology.wannier_sector_polarization(h, sector="up")
    with pytest.raises(ValueError, match="gauge"):
        topology.wannier_sector_polarization(h, gauge="wrong")
    h1 = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="two-dimensional"):
        topology.wannier_sector_polarization(h1)
