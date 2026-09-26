"""A distance-dependent interaction Vr(r1,r2) must keep whole distance
shells, so that it has the point-group symmetry of the lattice.

The multicell builder behind get_hamiltonian(tij=Vr) keeps a whole lattice
cell as soon as one of its pairs is within its range, so on its own it
returned every pair below 5 plus an uneven fringe beyond it: on the
honeycomb lattice 4 of the 6 pairs at distance 5 and 4 of the 12 at 5.29,
which breaks the three-fold rotation. specialhopping.distance_cut_interaction
keeps every pair up to rcut and none beyond it, and every mean-field, RPA
and BSE entry point that takes Vr builds it that way. A finite (0d) system
has no fringe, so there every pair is kept, as before."""
import numpy as np
import pytest

from pyqula import geometry, islands, specialhopping


def _coulomb(r1, r2):
    return 0.6 / np.sqrt((r1 - r2).dot(r1 - r2) + 0.25)


def _shells(g, v, shift=1):
    """{distance: [values]} of every nonzero pair of a {direction: matrix}
    interaction, reading every shift-th row and column (2 for a spinful
    one, whose four spin blocks carry the same value); the onsite diagonal
    is skipped"""
    out = dict()
    for d, m in v.items():
        R = d[0] * g.a1 + d[1] * g.a2 + d[2] * g.a3
        m = np.array(m)
        n = m.shape[0] // shift
        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(g.r[j] + R - g.r[i])
                if dist < 1e-6: continue
                val = m[shift * i, shift * j]
                if abs(val) > 1e-12:
                    out.setdefault(round(dist, 5), []).append(val)
    return out


def _shell_sizes(g, rmax, ncells=12):
    """Number of pairs of sites at each distance up to rmax, counted by
    brute force over a region of cells far larger than rmax"""
    sizes = dict()
    for i1 in range(-ncells, ncells + 1):
        for i2 in range(-ncells, ncells + 1):
            R = i1 * g.a1 + i2 * g.a2
            for i in range(len(g.r)):
                for j in range(len(g.r)):
                    dist = np.linalg.norm(g.r[j] + R - g.r[i])
                    if 1e-6 < dist < rmax + 1e-6:
                        key = round(dist, 5)
                        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def _assert_whole_shells(g, v, rcut, shift):
    shells = _shells(g, v, shift=shift)
    assert {d: len(x) for d, x in shells.items()} == _shell_sizes(g, rcut)
    for d, vals in shells.items():  # the value depends on the distance alone
        assert np.max(np.abs(np.array(vals) - vals[0])) < 1e-12


def _builders():
    """Every mean-field and RPA builder of a density-density interaction
    that takes Vr, and the spin-spin one that takes the exchange tail Jr,
    as (name, spin-block size, function of (h, rcut))"""
    from pyqula.scftk.spinspin import _build_density_v, _build_v
    from pyqula.chitk.densitychi import _density_v
    return [
        ("VJinteraction", 2,
         lambda h, rcut: _build_density_v(h, Vr=_coulomb, rcut=rcut)),
        ("densitychi", 1,
         lambda h, rcut: _density_v(h, Vr=_coulomb, rcut=rcut)),
        ("exchange Jr", 2,
         lambda h, rcut: _build_v(h, Jr=_coulomb, rcut=rcut)),
    ]


@pytest.mark.parametrize("rcut", [None, 7.5])
@pytest.mark.parametrize("i", range(3))
def test_the_builders_keep_whole_distance_shells(i, rcut):
    name, shift, build = _builders()[i]
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=(shift == 2)).get_multicell().get_dense()
    _assert_whole_shells(g, build(h, rcut), 5.0 if rcut is None else rcut,
                         shift)


def test_a_mean_field_with_vr_has_the_symmetry_of_the_lattice():
    """End to end: the interaction a converged mean field stores on h.V,
    from get_mean_field_hamiltonian(Vr=...), keeps whole distance shells"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    hmf = h.get_mean_field_hamiltonian(Vr=_coulomb, filling=0.5, nk=6,
                                       mf="ferro", maxerror=1e-8)
    _assert_whole_shells(g, hmf.V, 5.0, 2)


def test_vinteraction_with_vr_keeps_whole_distance_shells():
    """The same through the older Vinteraction engine"""
    from pyqula.scftk.densitydensity import Vinteraction
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    scf = Vinteraction(h, Vr=_coulomb, filling=0.5, nk=6, maxerror=1e-8)
    _assert_whole_shells(g, scf.hamiltonian.V, 5.0, 1)


def _island():
    g = islands.get_geometry(name="honeycomb", n=3, nedges=6, rot=0.0)
    g.dimensionality = 0
    return g


def test_a_finite_system_keeps_every_pair():
    """A 0d island has no fringe to trim, so with the default rcut=None every
    pair of sites interacts, however far apart, exactly as the plain
    get_hamiltonian(tij=Vr) builder gives it"""
    g = _island()
    r = np.array(g.r)
    dmax = np.max(np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2))
    assert dmax > 5.0  # there are pairs beyond the periodic default
    v = specialhopping.distance_cut_interaction(g, _coulomb)
    old = g.get_hamiltonian(has_spin=False, is_multicell=True, tij=_coulomb)
    old = old.get_hopping_dict()
    assert list(v.keys()) == [(0, 0, 0)]
    assert np.max(np.abs(v[(0, 0, 0)] - old[(0, 0, 0)])) < 1e-14
    n = len(g.r)
    for i in range(n):
        for j in range(n):
            assert abs(v[(0, 0, 0)][i, j] - _coulomb(r[i], r[j])) < 1e-12
    # and the mean-field builder sees the same, spin-doubled
    from pyqula.scftk.spinspin import _build_density_v
    h = g.get_hamiltonian(has_spin=True)
    vd = _build_density_v(h, Vr=_coulomb)
    assert np.max(np.abs(vd[(0, 0, 0)][::2, ::2] - v[(0, 0, 0)])) < 1e-14


def test_an_explicit_rcut_applies_to_a_finite_system_too():
    g = _island()
    r = np.array(g.r)
    v = specialhopping.distance_cut_interaction(g, _coulomb, rcut=3.0)[(0, 0, 0)]
    dist = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
    assert np.all(v[dist > 3.0 + 1e-6] == 0.)
    assert np.all(np.abs(v[dist <= 3.0]) > 0.)


def test_a_nonpositive_rcut_is_rejected():
    g = geometry.honeycomb_lattice()
    with pytest.raises(ValueError, match="rcut"):
        specialhopping.distance_cut_interaction(g, _coulomb, rcut=-1.)
