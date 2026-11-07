# Tests of the triangle and qtci integrations of the optical conductivity
# (conductivitytk/kubosimplex.py). Each one checks the new k-integration
# against an independent path -- the plain grid sum of kubo.py, the Chern
# number of topology.py, the universal conductivity of graphene -- or
# against an exact relation, never against a recorded number.
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula import topology
from pyqula.conductivitytk import kubo


def _haldane(t2=0.2, has_spin=False):
    """Spinless Haldane model at a Fermi level inside its gap, the same
    model as in test_optical_conductivity.py"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    h.shift_fermi(0.3)
    return h


def _gapped_honeycomb(has_spin=False):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_sublattice_imbalance(0.3)
    return h


def test_triangle_agrees_with_the_grid_sum():
    """The triangle integration and the grid sum are two discretizations of
    the same Brillouin-zone integral. At a broadening the grid can resolve,
    a coarse triangle mesh must reproduce a fine grid for the full complex
    tensor, the Hall component included, at frequencies inside the gap,
    across the absorption edge and in the band. The triangle result
    converges at second order in the mesh spacing, and omega=2 sits on a
    van Hove singularity, which sets the tolerance."""
    h = _haldane()
    ws = [0.3, 1.2, 2.0, 3.0]
    _, s_tri = conductivity.optical_conductivity(h, energies=ws, nk=192,
            T=0.01, delta=0.03, integration="triangle")
    _, s_grid = conductivity.optical_conductivity(h, energies=ws, nk=480,
            T=0.01, delta=0.03, integration="grid")
    scale = np.max(np.abs(s_grid[:,0,0]))
    assert np.max(np.abs(s_tri[:,:2,:2]-s_grid[:,:2,:2])) < 5e-3*scale
    assert abs(s_grid[0,0,1]) > 0.1*scale # the Hall part is really tested


def test_triangle_agrees_with_the_grid_sum_in_1d():
    """Same comparison in 1D, where the simplices are segments: a
    dimerized chain, whose interband absorption has square-root van Hove
    singularities at both edges of the band of transitions."""
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.4*np.cos(np.pi*r[0])) # +0.4, -0.4 alternating
    ws = [0.9, 1.5, 2.5]
    _, s_tri = conductivity.optical_conductivity(h, energies=ws, nk=400,
            T=0.01, delta=0.02, integration="triangle", component="xx")
    _, s_grid = conductivity.optical_conductivity(h, energies=ws, nk=20000,
            T=0.01, delta=0.02, integration="grid", component="xx")
    assert np.allclose(s_tri, s_grid, rtol=2e-3)


def test_triangle_converges_below_the_mesh_resolution():
    """The point of the method: at a broadening far below the energy
    spacing of the mesh, the grid sum is noise while the triangle result
    is already converged, because the mesh only has to resolve the smooth
    transition energies and weights. Two triangle meshes must agree, and
    the grid sum on the coarser one must not."""
    h = _gapped_honeycomb()
    ws = [1.0, 1.5, 2.5]
    kw = dict(energies=ws, T=1e-3, delta=1e-3, component="xx")
    _, s128 = conductivity.optical_conductivity(h, nk=128,
            integration="triangle", **kw)
    _, s256 = conductivity.optical_conductivity(h, nk=256,
            integration="triangle", **kw)
    _, g256 = conductivity.optical_conductivity(h, nk=256,
            integration="grid", **kw)
    assert np.allclose(s128.real, s256.real, rtol=5e-3)
    assert np.max(np.abs(g256.real-s256.real)/s256.real) > 2e-2


def test_spinful_copy_doubles_the_conductivity():
    """A spinful copy of a spinless model gives exactly twice its
    conductivity, with both simplex integrations. Every band is then a
    degenerate pair, which goes through the multiplet bookkeeping of the
    pair fields."""
    ws = [0.5, 1.0, 2.0, 3.0]
    for integration in ["triangle", "qtci"]:
        kw = dict(energies=ws, nk=32, T=0.01, delta=1e-2,
                integration=integration, tolerance=1e-8)
        _, s1 = conductivity.optical_conductivity(_haldane(), **kw)
        _, s2 = conductivity.optical_conductivity(_haldane(has_spin=True),
                **kw)
        assert np.allclose(s2, 2.*s1, atol=1e-6*np.max(np.abs(s1)))


def test_pair_weights_do_not_depend_on_the_basis_of_a_multiplet():
    """Inside a degenerate multiplet the eigenvectors are only defined up
    to a unitary rotation, and a single weight v^a_nm v^b_mn changes with
    it. The weights that are interpolated are averaged over the multiplets,
    which makes them independent of that choice. A random rotation inside
    a two-fold degenerate level must leave them unchanged, while the raw
    weights (each band its own block) do change."""
    from pyqula.conductivitytk import kubosimplex
    rng = np.random.default_rng(3)
    nb = 4
    es = np.array([[-1.,-1.,0.5,1.2]]) # bands 0 and 1 are degenerate
    v = rng.normal(size=(3,nb,nb)) + 1j*rng.normal(size=(3,nb,nb))
    v = v + np.conjugate(np.transpose(v,(0,2,1))) # Hermitian velocities
    q,_ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    U = np.eye(nb,dtype=np.complex128)
    U[:2,:2] = q # rotation inside the degenerate level
    vr = np.conjugate(U.T)@v@U
    vs = np.ascontiguousarray(v[None])
    vrs = np.ascontiguousarray(vr[None])
    ratio,_ = kubo._response_weights(es,0.05,1e-8)
    ratio = np.ascontiguousarray(ratio)
    pairs = np.array([(n,m) for n in range(nb) for m in range(n,nb)])
    ca = np.array([0,0,1]); cb = np.array([0,1,1])
    _,W = kubosimplex._pair_fields_jit(es,ratio,vs,pairs,ca,cb,1e-8)
    _,Wr = kubosimplex._pair_fields_jit(es,ratio,vrs,pairs,ca,cb,1e-8)
    assert np.allclose(W, Wr, atol=1e-12)
    _,W0 = kubosimplex._pair_fields_jit(es,ratio,vs,pairs,ca,cb,-1.)
    _,W0r = kubosimplex._pair_fields_jit(es,ratio,vrs,pairs,ca,cb,-1.)
    assert not np.allclose(W0, W0r, atol=1e-3)


def test_triangle_hall_conductivity_of_a_chern_insulator(tmp_path, monkeypatch):
    """sigma_xy(0) = -C/(2 pi), with C from the independent
    Fukui-Hatsugai-Suzuki Chern number, for both signs of the flux. Inside
    the gap the integrand is smooth, and there the linear interpolation
    converges at second order: doubling nk divides the error by four. (The
    grid sum converges exponentially for such a smooth periodic integrand,
    which is why it is the better choice there.)"""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    for t2 in [0.2, -0.2]:
        h = _haldane(t2=t2)
        c = round(topology.chern(h, nk=20))
        assert c != 0
        err = []
        for nk in [30, 60]:
            _, sigma = conductivity.optical_conductivity(h, energies=[0.],
                    nk=nk, T=0.01, delta=1e-3, integration="triangle")
            err.append(abs(2.*np.pi*sigma[0,0,1].real + c))
            assert abs(sigma[0,0,0]) < 1e-3
        assert err[1] < 2e-3
        assert 3. < err[0]/err[1] < 5.


def test_triangle_graphene_universal_conductivity_at_small_broadening():
    """Graphene absorbs pi e^2/(4h) = 1/8 per spin in the Dirac regime. The
    transition energy is a cone there, not smooth at K, and the broadening
    is far below the mesh spacing, so this checks that the linear
    interpolation still converges to the right value. The tight-binding
    dispersion raises the absorption above 1/8 by about 1% at 0.3 t, which
    sets the tolerance."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    _, sigma = conductivity.optical_conductivity(h, energies=[0.2,0.3],
            nk=240, T=0.01, delta=1e-3, integration="triangle")
    universal = 1./8.
    assert np.allclose(sigma[:,0,0].real, universal, rtol=2e-2)
    assert np.allclose(sigma[:,1,1].real, universal, rtol=2e-2)


def test_qtci_reproduces_the_triangle_result_on_the_same_mesh():
    """With integration='qtci' the transition energies and weights are
    reconstructed from a quantics tensor cross interpolation instead of
    being computed on every point; the triangle integral on top is the
    same. On the same 2^R mesh the two must agree to the interpolation
    tolerance, including the Hall component."""
    h = _haldane()
    ws = [0.3, 1.2, 2.0]
    kw = dict(energies=ws, nk=128, T=0.01, delta=1e-2)
    _, s_tri = conductivity.optical_conductivity(h, integration="triangle",
            **kw)
    _, s_q = conductivity.optical_conductivity(h, integration="qtci",
            tolerance=1e-7, **kw)
    scale = np.max(np.abs(s_tri[:,0,0]))
    assert np.max(np.abs(s_q-s_tri)) < 1e-4*scale


def test_unknown_integration_lists_the_accepted_ones():
    h = _haldane()
    with pytest.raises(ValueError) as err:
        conductivity.optical_conductivity(h, nk=4, integration="simpson")
    for name in kubo.get_integration_names():
        assert name in str(err.value)


def test_simplex_integrations_need_a_broadening():
    h = _haldane()
    for integration in ["triangle", "qtci"]:
        with pytest.raises(ValueError):
            conductivity.optical_conductivity(h, nk=8, delta=0.,
                    integration=integration)


def test_qtci_is_two_dimensional_only():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        conductivity.optical_conductivity(h, nk=16, integration="qtci")


def _kernel_pair(h, nk, ws, T, delta):
    """The simplex sum of the same pair fields with the default kernel,
    which builds the sources once and sums them per frequency, and with
    the direct sum over simplices at every frequency"""
    from pyqula.conductivitytk import kubosimplex
    ks = kubo._kmesh(h, nk)
    es, vs, cellvol, scale = kubo._bands_and_velocities(h, ks)
    pairs, dE, W, comps, trans = kubosimplex.pair_fields(es, vs, T,
            1e-6*scale, True, True)
    pairs, dE, W = kubosimplex._prune(pairs, dE, W)
    args = (h.dimensionality, nk, pairs, dE, W, comps, trans, ws, delta,
            cellvol)
    return (kubosimplex._sigma_from_fields(*args),
            kubosimplex._sigma_from_fields(*args, kernel="simplices"))


def test_source_kernel_equals_the_sum_over_simplices():
    """The default kernel rewrites the sum over simplices as a sum over its
    sources (the vertex energies, and the close edges and simplices
    expanded around their middle), built once for all frequencies. It is
    the same integral of the same interpolant, so the two agree to
    rounding, not to a discretization error. The direct sum decides the
    Taylor branch per frequency and is exact to about 1e-13 of the largest
    entry; the source kernel cannot, and its divided differences over
    spreads just above delta lose up to 2e-11 of it here, which sets the
    tolerance. Checked on a gapped and on a spinful Chern insulator, where
    every band is a degenerate pair, at a broadening below and above the
    mesh spacing, and in 1D, where the simplices are segments."""
    ws = np.linspace(0., 7., 41)
    for h in [_gapped_honeycomb(), _haldane(has_spin=True)]:
        for delta in [1e-3, 3e-2]:
            s_new, s_old = _kernel_pair(h, 40, ws, 0.01, delta)
            scale = np.max(np.abs(s_old))
            assert np.max(np.abs(s_new-s_old)) < 1e-10*scale
    assert np.max(np.abs(s_old[:,0,1].real)) > 0.1*scale # Hall part compared
    g = geometry.chain().get_supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.4*np.cos(np.pi*r[0]))
    for delta in [1e-3, 3e-2]:
        s_new, s_old = _kernel_pair(h, 300, ws, 0.01, delta)
        assert np.max(np.abs(s_new-s_old)) < 1e-10*np.max(np.abs(s_old))


def _triangle_quadrature(e, omega, delta, n=64):
    """Average of 1/(omega + i delta - E) over a triangle with E linear
    between the vertex energies e, by Gauss-Legendre quadrature on the
    square of the Duffy map. The energies are measured from omega first,
    which is exact close to resonance, so for an integrand that is smooth
    on the scale of the triangle this is exact to rounding"""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5*(x+1.); w = 0.5*w
    u, v = np.meshgrid(x, x, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    a = u; b = (1.-u)*v # a point of the unit triangle, Jacobian 1-u
    t = e-omega
    x = t[0] + a*(t[1]-t[0]) + b*(t[2]-t[0]) # E - omega
    return 2.*np.sum(wu*wv*(1.-u)/(1j*delta-x))


def test_nearly_flat_triangle_far_from_resonance():
    """A triangle whose transition energies spread by s far from the
    frequency, here energies 0.5, 0.5+0.4s, 0.5+s at omega=0 with
    delta=1e-6. The plain divided difference cancels terms of size
    (|x|/s)^2 there, so a switch to the Taylor series on s compared with
    delta alone lost up to 10% of this simplex. The direct kernel must give
    the quadrature value to rounding, far from resonance and on it. The
    source kernel expands below s = delta only, so at s = delta it takes
    the divided difference, which loses about eps (|x|/s)^2 of the value
    far from resonance. It keeps that vertex form because the loss is
    below 3e-11 of the peak of the triangle, about 1/delta, so here it
    is 1.5e-10 of the far value."""
    from pyqula.conductivitytk import kubosimplex
    delta = 1e-6
    sim = np.array([[0,1,2]], dtype=np.int64)
    W = np.ones((3,1,1), dtype=np.complex128)
    args = (np.zeros(1,dtype=np.int64), np.ones(1,dtype=np.bool_))
    for s in [3e-8, 1e-7, 1e-6]:
        e = np.array([0.5, 0.5+0.4*s, 0.5+s])
        for omega in [0., 0.5+0.5*s]: # far from the triangle, and on it
            ref = _triangle_quadrature(e, omega, delta)
            for name in ["sources", "simplices"]:
                val = kubosimplex._kernels[name](sim, e[:,None], W, *args,
                        np.array([omega]), delta)[0,0]
                tol = 1e-9 if (name=="sources" and s>=delta) else 1e-12
                assert abs(val-ref) < tol*abs(ref)
