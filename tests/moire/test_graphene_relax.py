import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from pyqula import specialgeometry
from pyqula import geometry
from pyqula.graphenetk import gsfe as gsfetk
from pyqula.graphenetk import elastic as elastictk
from pyqula.graphenetk.geometry import GrapheneGeometry
from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian
from pyqula.graphenetk.relax import _layer_groups

# GrapheneGeometry.relax() minimizes the sum of the interlayer GSFE
# adhesion energy (graphenetk/gsfe.py) and intralayer elastic energy
# (graphenetk/elastic.py), following Carr, Massatt, Torrisi, Cazeaux,
# Luskin, Kaxiras, "Relaxation and Domain Formation in Incommensurate 2D
# Heterostructures", arXiv:1805.06972. These tests check the physical
# invariants that distinguish a correct implementation from a broken one
# (rather than any single reference number, which this repo cannot check
# against an external DFT/LAMMPS calculation): AA is the GSFE maximum and
# AB/BA the degenerate minima; the elastic energy genuinely forbids bond
# collapse (a real bug, caught only by explicitly probing for it -- see
# graphenetk/elastic.py's module docstring); relaxed bond lengths and
# band structures stay physical; and the local relaxation amplitude grows
# monotonically as the twist angle shrinks, the one robust, parameter-free
# prediction shared by every paper in this literature.


def _bilayer(m0):
    return specialgeometry.twisted_bilayer(m0=m0)


def test_gsfe_aa_is_max_ab_ba_are_degenerate_minima():
    """AA stacking (zero registry shift) must be the GSFE maximum, and
    the two inequivalent Bernal shifts +(a1-a2)/3 (AB) and -(a1-a2)/3 (BA)
    must be degenerate minima -- the well-established fact that Bernal
    stacking is graphene's energy minimum and AA its maximum, which
    stacking_phases_matrix()'s handedness convention is specifically
    chosen to reproduce (see its docstring)."""
    pm = gsfetk.stacking_phases_matrix()

    def gsfe_at(b):
        v, w = pm @ np.array(b)
        return float(gsfetk.gsfe(v, w))

    e_aa = gsfe_at([0., 0.])
    shift = (gsfetk.MONOLAYER_A1 - gsfetk.MONOLAYER_A2)/3
    e_ab = gsfe_at(shift)
    e_ba = gsfe_at(-shift)

    assert np.isclose(e_ab, e_ba, atol=1e-8), (e_ab, e_ba)
    assert e_aa > e_ab + 1.0, (e_aa, e_ab)

    # AA/AB/BA are stationary points of the periodic GSFE landscape: a
    # small step away from any of them should not lower AA's value below
    # AB/BA's, nor raise AB/BA's above AA's, at a handful of directions.
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        step = 0.05*np.array([np.cos(angle), np.sin(angle)])
        assert gsfe_at(np.array([0., 0.]) + step) < e_aa + 1e-6
        assert gsfe_at(shift + step) > e_ab - 1e-6


def test_cell_elastic_energy_forbids_bond_collapse():
    """Regression test for the exploit found while developing this
    feature (graphenetk/elastic.py's module docstring): forcing one bond
    to shrink to 20% of its rigid length, while letting the other two
    bonds move completely freely to try to hide the resulting strain,
    must still cost a large amount of energy -- not the near-zero value
    that both an earlier least-squares deformation-gradient fit and an
    average-of-exact-pairwise-H fit allowed."""
    theta = np.array([0., 120., 240.])*np.pi/180
    d0 = np.array([[np.cos(t), np.sin(t)] for t in theta])

    def energy(d):
        return float(elastictk.cell_elastic_energy(d0, d.reshape(3, 2)))

    def objective(x):
        d = d0.copy()
        d[0] = 0.2*d0[0]
        d[1] = x[0:2]
        d[2] = x[2:4]
        return energy(d)

    res = minimize(objective, np.concatenate([d0[1], d0[2]]), method="Nelder-Mead")
    assert res.fun > 1e3, res.fun
    assert np.isclose(energy(d0), 0.0, atol=1e-6)


def test_relax_keeps_bond_lengths_physical():
    """The naive discretizations tried while developing this feature let
    relaxation collapse bonds to a small fraction of their rigid length
    at negligible energy cost (see graphenetk/elastic.py) -- this is
    exactly the invariant specialhopping.twisted's own
    `if (r-1.0)<-0.1: raise` assumes always holds, so it is also what
    first surfaced the bug via test_relax_bands_are_finite below."""
    g0 = _bilayer(m0=6)
    g2 = GrapheneGeometry(g0).relax(maxiter=1000)
    r2 = np.array(g2.r)
    for layer_idx in _layer_groups(r2[:, 2]):
        d, _ = cKDTree(r2[layer_idx][:, :2]).query(r2[layer_idx][:, :2], k=2)
        assert d[:, 1].min() > 0.9, d[:, 1].min()


def test_relax_bands_are_finite():
    """A relaxed geometry fed into GrapheneHamiltonian (whose hoppings
    decay with the true 3D interatomic distance, specialhopping.twisted)
    must still produce a finite band structure -- an unphysically
    collapsed bond would instead blow up specialhopping.twisted's hopping
    or trip its internal sanity check."""
    g0 = _bilayer(m0=6)
    g2 = GrapheneGeometry(g0).relax(maxiter=1000)
    h = GrapheneHamiltonian(g2)
    (k, e) = h.get_bands(num_bands=10)
    assert np.all(np.isfinite(np.array(e)))


def test_relax_requires_sublattice():
    """GrapheneGeometry/relax_structure need has_sublattice=True (the
    GSFE registry vector is only meaningful given each atom's A/B label)
    -- a geometry built without it should fail fast with a clear error
    rather than silently mis-relaxing."""
    g = geometry.chain()
    assert not g.has_sublattice
    with pytest.raises(ValueError):
        GrapheneGeometry(g)


def test_relax_amplitude_grows_as_twist_angle_shrinks():
    """The one robust, parameter-free prediction shared by every paper on
    TBG relaxation (Nam & Koshino, arXiv:1706.03908; Carr et al.,
    arXiv:1805.06972): the local relaxation amplitude grows as the twist
    angle shrinks (larger moire supercell => more room for AB/BA domains
    to form => larger local displacements away from the rigid lattice).
    m0 kept small for runtime; the trend already holds well before
    reaching the small-angle domain-wall regime those papers focus on."""
    amplitudes = []
    for m0 in (4, 7, 10):
        g0 = _bilayer(m0=m0)
        g2 = GrapheneGeometry(g0).relax(maxiter=1000)
        u = np.array(g2.r)[:, :2] - np.array(g0.r)[:, :2]
        layer0 = _layer_groups(np.array(g0.r)[:, 2])[0]
        amplitudes.append(np.std(np.linalg.norm(u[layer0], axis=1)))
    assert amplitudes[0] < amplitudes[1] < amplitudes[2], amplitudes
