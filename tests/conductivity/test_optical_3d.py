"""The Kubo velocities and the optical conductivity of 3D crystals, with the
uniform grid and with the tetrahedron integration.

The invariants: the band velocity is the k-derivative of the band, and the
whole velocity matrix agrees with the independent sparse construction of
kpmkubo.py; a stack of decoupled layers gives the 2D sheet conductance per
layer spacing, exactly on the grid, and to second order with the
tetrahedra, whose projections are the triangles of the 2D layer meshed in
the basis (a1,-a2) (not exactly, since the vertex average of a tetrahedron
counts one projected vertex twice); with an interlayer hopping the tetrahedra
converge at second order to a dense grid; and the f-sum rule holds in 3D.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import multihopping
from pyqula import conductivity
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import kubosimplex


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _haldane(t2=0.1, imbalance=0.2):
    """Spinless Haldane model with a staggered onsite, C=1 at half filling,
    gap from -0.25 to 0.25"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(imbalance)
    return h


def _stack(h2, c=2., tm=0.):
    """3D crystal of copies of the 2D h2 stacked along z with spacing c,
    with the interlayer hopping diag(tm,-tm), a mass that depends on kz"""
    h = h2.copy()
    h.geometry.a3 = np.array([0., 0., c])
    h.geometry.dimensionality = 3
    hd = {(R[0], R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    if tm != 0.:
        M = np.diag([tm, -tm]).astype(np.complex128)
        hd[(0, 0, 1)] = M
        hd[(0, 0, -1)] = M
    h.dimensionality = 3
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _mirror(h2):
    """The same 2D crystal in the lattice basis (a1,-a2)"""
    h = h2.copy()
    h.geometry.a2 = -np.array(h2.geometry.a2)
    hd = {(R[0], -R[1], 0): _dense(m) for R, m in h2.get_hopping_dict().items()}
    h.set_multihopping(multihopping.MultiHopping(hd))
    return h


def _cubic_bipartite():
    """Eight-site cubic cell with a staggered onsite, intracell bonds"""
    h = geometry.cubic_lattice_bipartite().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.3)
    return h


def test_band_velocity_of_the_cubic_lattice_is_the_band_slope():
    """v_z of the simple cubic band at a generic k is dE/dK_z, with
    K_z = 2 pi k_3/|a_3|, by a centered finite difference"""
    h = geometry.cubic_lattice().get_hamiltonian(has_spin=False)
    k = np.array([0.13, 0.31, 0.07])
    es, vs, _, _ = kubo._bands_and_velocities(h, np.array([k]))
    hk = h.get_hk_gen()
    dk = 1e-5
    for a in range(3):
        kp = k.copy(); kp[a] += dk
        km = k.copy(); km[a] -= dk
        ep = np.linalg.eigvalsh(_dense(hk(kp)))[0]
        em = np.linalg.eigvalsh(_dense(hk(km)))[0]
        slope = (ep-em)/(2.*dk)/(2.*np.pi) # dE/dK, the cell is a unit cube
        assert abs(vs[0, a, 0, 0].real-slope) < 1e-8, (a, vs[0, a, 0, 0], slope)
    # and analytically: the first-neighbor hopping is 1 here, so
    # E = 2 sum_a cos(2 pi k_a) and v_z = -2 sin(2 pi k_3)
    assert abs(vs[0, 2, 0, 0].real+2.*np.sin(2.*np.pi*k[2])) < 1e-10


def test_velocity_matrix_agrees_with_the_sparse_kpm_construction():
    """The whole 3D velocity matrix, intracell bonds included, against the
    sparse full-bond-vector construction of kpmkubo.py, at a generic k,
    for the single-k and for the batched code path"""
    from pyqula.conductivitytk import kpmkubo
    h = _cubic_bipartite()
    k = np.array([0.11, 0.27, 0.38])
    hm, orders, hkgen, jac, dr, cellvol, scale = kubo._setup(h)
    hk = kubo._hk(hkgen, k)
    v1 = kubo._velocities(hm, orders, jac, dr, hk, k)
    tms, dirs = kubo._hopping_arrays(hm)
    ph = kubo._bloch_phases(dirs, np.array([k]))
    dhs = [kubo._derivative_batch(tms, dirs, ph, o) for o in orders]
    v2 = kubo._velocities_batch(orders, jac, dr, np.array([hk]), dhs)[0]
    ops = kpmkubo._Operators(h)
    assert abs(cellvol-8.) < 1e-12
    for a in range(3):
        v3 = ops.velocity(k, a).toarray()
        assert np.max(np.abs(v3)) > 0.1
        assert np.allclose(v1[a], v3, atol=1e-12, rtol=0.)
        assert np.allclose(v2[a], v3, atol=1e-12, rtol=0.)


def test_decoupled_layers_give_the_sheet_conductance_per_spacing_on_the_grid():
    """Zero interlayer hopping, grid: every in-plane component of the 3D
    conductivity times the spacing c is the 2D one, exactly, and nothing
    flows along z"""
    h2 = _haldane()
    h3 = _stack(h2, c=2.)
    ws = np.linspace(0., 3., 7)
    kw = dict(energies=ws, nk=12, delta=0.1, T=0.05)
    _, s2 = conductivity.optical_conductivity(h2, **kw)
    _, s3 = conductivity.optical_conductivity(h3, **kw)
    assert np.max(np.abs(s2)) > 0.1
    assert np.allclose(2.*s3, s2, atol=1e-12, rtol=0.)
    assert np.max(np.abs(s3[:, 2, :])) == 0.
    assert np.max(np.abs(s3[:, :, 2])) == 0.


def test_decoupled_layers_with_tetrahedra():
    """Zero interlayer hopping, tetrahedra. The six tetrahedra of a cube
    share its (0,0,0)-(1,1,1) diagonal, so above every triangle of the
    in-plane mesh cut along (0,0)-(1,1) they fill the prism on it. That is
    the 2D mesh of the same layer in the basis (a1,-a2), so the transition
    energies are interpolated exactly as there. The weights are vertex
    averages, and a tetrahedron counts one vertex twice, so the two agree
    to the second order error of the method: below 1e-2 relative at nk=16,
    falling by about 3 from nk=8. In the basis (a1,a2), cut the other
    way, the difference is also of second order"""
    h2 = _haldane()
    hm = _mirror(h2)
    h3 = _stack(h2, c=2.)
    ws = np.linspace(0., 3., 7)
    kw = dict(energies=ws, delta=0.1, T=0.05, integration="triangle")
    err = []
    for nk in (8, 16):
        _, sm = conductivity.optical_conductivity(hm, nk=nk, **kw)
        _, s3 = conductivity.optical_conductivity(h3, nk=nk, **kw)
        sc = np.max(np.abs(sm))
        err.append(np.max(np.abs(2.*s3-sm))/sc)
        assert np.max(np.abs(s3[:, 2, :])) == 0.
    assert err[1] < 1e-2, err
    assert err[0]/err[1] > 2.5, err


def test_tetrahedra_converge_to_the_dense_grid_with_interlayer_hopping():
    """A kz dependent mass couples the layers, and the crystal stays an
    insulator with C=1 at every kz, with a gap from -0.15 to 0.15. At
    delta=0.3 the grid is converged to 1e-4 at nk=48 (7e-5 from nk=60,
    checked against nk=36 here). The tetrahedra at nk=16 and nk=32, in the
    plane and along z, approach it at second order, from 7e-2 to 2e-2
    relative: at this broadening the grid, which converges exponentially,
    is the better method, and the tetrahedra pay off at small delta"""
    h3 = _stack(_haldane(), c=2., tm=0.05)
    ws = np.linspace(0.2, 3., 8)
    kw = dict(energies=ws, delta=0.3, T=0.01, intraband=False)
    _, ref = conductivity.optical_conductivity(h3, nk=48, **kw)
    _, ref2 = conductivity.optical_conductivity(h3, nk=36, **kw)
    for (a, b) in ((0, 0), (0, 1), (2, 2)):
        sc = np.max(np.abs(ref[:, a, b]))
        assert sc > 1e-3, (a, b, sc)
        assert np.max(np.abs(ref[:, a, b]-ref2[:, a, b]))/sc < 2e-3
    err = {}
    for nk in (16, 32):
        _, s = conductivity.optical_conductivity(h3, nk=nk,
                integration="triangle", **kw)
        for (a, b) in ((0, 0), (0, 1), (2, 2)):
            sc = np.max(np.abs(ref[:, a, b]))
            err[(nk, a, b)] = np.max(np.abs(s[:, a, b]-ref[:, a, b]))/sc
    for (a, b) in ((0, 0), (0, 1), (2, 2)):
        assert err[(32, a, b)] < 3e-2, err
        assert err[(16, a, b)]/err[(32, a, b)] > 3.5, err


def _spectral_weight(s, ws, a, delta):
    """int Re sigma_aa(w) dw over the real axis, by the trapezoid rule on
    ws, with the Lorentzian tails beyond |w| = ws[-1] added to first
    order, 2 W delta/L of a total weight pi W"""
    lhs = np.sum(s[1:, a, a].real+s[:-1, a, a].real)*(ws[1]-ws[0])/2.
    return lhs/(1.-2.*delta/(np.pi*ws[-1]))


def test_f_sum_rule_in_3d():
    """int Re sigma_aa(w) dw over the real axis is pi W_aa, with W the
    inverse mass weight of sum_rule_weight, built from the second
    k-derivative of H instead of the velocities, for an insulator in which
    the whole weight is interband. On a finite mesh the two differ by the
    mesh mean of the band curvature, which falls off exponentially with nk:
    in the plane 1e-3 relative at nk=24, along z, where the bands are
    smooth, below 1e-5"""
    h3 = _stack(_haldane(), c=2., tm=0.05)
    nk = 24
    delta = 0.2
    T = 0.002
    ws = np.linspace(-30., 30., 3001)
    _, s = conductivity.optical_conductivity(h3, energies=ws, nk=nk,
            delta=delta, T=T)
    W = conductivity.sum_rule_weight(h3, nk=nk, T=T)
    D = conductivity.drude_weight(h3, nk=nk, T=T)
    for a, tol in ((0, 2e-3), (2, 1e-4)):
        assert W[a, a] > 1e-3
        assert abs(D[a, a]) < 1e-10 # an insulator
        lhs = _spectral_weight(s, ws, a, delta)
        assert abs(lhs-np.pi*W[a, a]) < tol*np.pi*W[a, a], (a, lhs,
                np.pi*W[a, a])


def test_tetrahedra_keep_the_spectral_weight_of_the_mesh():
    """The tetrahedra spread each transition over the energies of its
    tetrahedron but keep its weight, and the vertex averages over the six
    tetrahedra per cube sum to the uniform mean, so the frequency integral
    of Re sigma is the one of the grid on the same mesh"""
    h3 = _stack(_haldane(), c=2., tm=0.05)
    kw = dict(nk=6, delta=0.2, T=0.002, energies=np.linspace(-30., 30., 3001))
    ws = kw["energies"]
    _, sg = conductivity.optical_conductivity(h3, **kw)
    _, st = conductivity.optical_conductivity(h3, integration="triangle", **kw)
    for a in (0, 2):
        g = _spectral_weight(sg, ws, a, 0.2)
        t = _spectral_weight(st, ws, a, 0.2)
        assert g > 1e-3
        assert abs(g-t) < 1e-4*g, (a, g, t)


def test_qtci_refuses_3d():
    h3 = _stack(_haldane())
    with pytest.raises(NotImplementedError, match="3D"):
        conductivity.optical_conductivity(h3, nk=8, integration="qtci")


def test_source_kernel_refuses_tetrahedra_and_the_default_is_direct():
    """The source kernel is written for segments and triangles only"""
    h3 = _stack(_haldane(), tm=0.1)
    ks = kubo._kmesh(h3, 4)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h3, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, 0.01,
            1e-6*scale, True, True)
    args = (3, 4, pairs, dE, W, comps, trans, np.array([0.5, 1.]), 0.1,
            cellvol)
    with pytest.raises(NotImplementedError, match="tetrahedra"):
        kubosimplex._sigma_from_fields(*args, kernel="sources")
    s0 = kubosimplex._sigma_from_fields(*args)
    s1 = kubosimplex._sigma_from_fields(*args, kernel="simplices")
    assert np.max(np.abs(s0-s1)) == 0.
