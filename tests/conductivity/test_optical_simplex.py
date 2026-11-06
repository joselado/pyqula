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
