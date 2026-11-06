# Tests of the band window of the optical conductivity
# (conductivitytk/kubowindow.py): only the bands close to the Fermi level
# are diagonalized and paired. Each test checks the window against the
# calculation with all the bands, or two integrations against each other,
# never against a recorded number.
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula import limits


def _disordered_supercell(n=2, w=0.4, seed=1, has_spin=False):
    """n x n supercell of the gapped honeycomb lattice with a random onsite
    energy of width w, which removes every degeneracy and every symmetry
    of the matrix elements"""
    g = geometry.honeycomb_lattice().get_supercell(n)
    h = g.get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.3)
    rng = np.random.default_rng(seed)
    h.add_onsite(lambda r: w*(rng.random()-0.5))
    return h


def test_window_with_every_band_is_the_full_calculation():
    """A window that holds every band keeps every pair, so it must give the
    full result to rounding, for the energy and the count forms and for
    the grid and the triangle integrations. The velocities are rotated
    another way (the commutator i[H,r] becomes (E_n - E_m) r_nm), which is
    exact for eigenvectors."""
    h = _disordered_supercell()
    ws = np.linspace(0., 4., 9)
    for integration in ["grid", "triangle"]:
        kw = dict(energies=ws, nk=12, T=0.02, delta=0.05,
                integration=integration)
        _, s0 = conductivity.optical_conductivity(h, **kw)
        _, s1 = conductivity.optical_conductivity(h, ewindow=100., **kw)
        _, s2 = conductivity.optical_conductivity(h, nv=8, nc=8, **kw)
        scale = np.max(np.abs(s0))
        assert np.max(np.abs(s1-s0)) < 1e-10*scale
        assert np.max(np.abs(s2-s0)) < 1e-10*scale


def test_absorption_converges_as_the_window_grows():
    """Re sigma at omega below the window is exact up to the Lorentzian
    tails of the pairs left out, so its error inside the window must drop
    as the window grows, and become small, while Im sigma keeps a larger,
    smooth Kramers-Kronig bias from the pairs left out."""
    h = _disordered_supercell(n=3)
    ws = np.linspace(0.2, 3., 15)
    kw = dict(energies=ws, nk=8, T=0.02, delta=0.05, component="xx")
    _, s0 = conductivity.optical_conductivity(h, **kw)
    scale = np.max(np.abs(s0.real))
    errs_re, errs_im = [], []
    for W in [1.2, 1.8, 2.4]:
        _, s = conductivity.optical_conductivity(h, ewindow=W, **kw)
        inside = ws < W - 0.25 # five broadenings below the window
        errs_re.append(np.max(np.abs((s-s0)[inside].real))/scale)
        errs_im.append(np.max(np.abs((s-s0)[inside].imag))/scale)
    assert errs_re[0] > errs_re[1] > errs_re[2]
    assert errs_re[2] < 1e-4
    assert errs_im[0] > errs_re[0] # the reactive part is biased more


def test_grid_triangle_and_qtci_agree_with_the_same_window():
    """The window chooses the bands, the integration does the k-sum, so the
    three integrations must agree with the same window as they do without
    one: qtci with the triangle result on the same mesh, to the
    interpolation tolerance, and a coarse triangle mesh with a fine grid
    at a broadening the grid resolves; the triangle error at nk=64 is
    about 4e-3, down from 1.6e-2 at nk=32, which sets the tolerance."""
    h = _disordered_supercell()
    ws = np.array([0.8, 1.5, 2.5])
    kw = dict(energies=ws, T=0.02, delta=0.1, nv=2, nc=2)
    _, s_tri = conductivity.optical_conductivity(h, nk=32,
            integration="triangle", **kw)
    _, s_q = conductivity.optical_conductivity(h, nk=32,
            integration="qtci", tolerance=1e-8, **kw)
    _, s_tri2 = conductivity.optical_conductivity(h, nk=64,
            integration="triangle", **kw)
    _, s_grid = conductivity.optical_conductivity(h, nk=96,
            integration="grid", **kw)
    scale = np.max(np.abs(s_tri[:,0,0]))
    assert np.max(np.abs(s_q-s_tri)) < 1e-5*scale
    assert np.max(np.abs(s_grid[:,:2,:2]-s_tri2[:,:2,:2])) < 5e-3*scale


def test_window_in_a_metal_keeps_the_bands_fixed_across_k():
    """In a metal the bands inside an energy window change from one
    k-point to the next. The window keeps a fixed range of band indices,
    every band that enters the window somewhere, so the triangle
    interpolation still pairs the same bands at the three vertices, and
    the windowed absorption agrees with the full one inside the window,
    Drude part included."""
    h = _disordered_supercell()
    h.shift_fermi(-0.9) # Fermi level inside the upper bands
    ws = np.linspace(0., 1.2, 7)
    kw = dict(energies=ws, nk=24, T=0.05, delta=0.05, component="xx",
            integration="triangle")
    _, s0 = conductivity.optical_conductivity(h, **kw)
    _, s1 = conductivity.optical_conductivity(h, ewindow=1.6, **kw)
    assert np.max(np.abs((s1-s0).real)) < 1e-3*np.max(np.abs(s0.real))
    # the mesh is really metallic: the number of occupied bands changes
    from pyqula.conductivitytk import kubo
    es = kubo._bands_and_velocities(h, kubo._kmesh(h, 24))[0]
    assert len(np.unique(np.sum(es < 0., axis=1))) > 1


def test_window_does_not_cut_a_degenerate_multiplet():
    """Every band of a spinful copy is a degenerate pair. A window of one
    band below and one above the Fermi level would keep one state of each
    pair, a combination fixed by the eigensolver and not by the physics;
    the range is grown to whole multiplets instead, so it gives the same
    result as two bands on each side."""
    h = _disordered_supercell(has_spin=True)
    kw = dict(energies=np.linspace(0.5, 3., 6), nk=10, T=0.02, delta=0.05,
            integration="triangle")
    _, s1 = conductivity.optical_conductivity(h, nv=1, nc=1, **kw)
    _, s2 = conductivity.optical_conductivity(h, nv=2, nc=2, **kw)
    assert np.max(np.abs(s1-s2)) < 1e-10*np.max(np.abs(s2))


def test_sparse_window_agrees_with_the_dense_one(monkeypatch):
    """Above limits.densedimension the window bands come from a
    shift-invert solver around the Fermi level and the velocity from the
    sparse hoppings. For an insulator it must keep the same bands as the
    dense subset solver and give the same result."""
    h = _disordered_supercell()
    kw = dict(energies=np.linspace(0.5, 3., 6), nk=8, T=0.02, delta=0.05,
            integration="grid")
    _, sd = conductivity.optical_conductivity(h, ewindow=1.5, **kw)
    _, sdc = conductivity.optical_conductivity(h, nv=3, nc=3, **kw)
    monkeypatch.setattr(limits, "densedimension", 4)
    _, ss = conductivity.optical_conductivity(h, ewindow=1.5, **kw)
    _, ssc = conductivity.optical_conductivity(h, nv=3, nc=3, **kw)
    scale = np.max(np.abs(sd))
    assert np.max(np.abs(ss-sd)) < 1e-8*scale
    assert np.max(np.abs(ssc-sdc)) < 1e-8*scale


def test_window_arguments_are_checked():
    h = _disordered_supercell()
    for kw in [dict(ewindow=1., nv=2), dict(ewindow=(0.2, 1.)),
               dict(nv=0, nc=0), dict(ewindow=(1., 2., 3.))]:
        with pytest.raises(ValueError):
            conductivity.optical_conductivity(h, nk=4, **kw)
