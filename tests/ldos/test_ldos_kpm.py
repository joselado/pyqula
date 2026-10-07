import warnings

import numpy as np
import pytest

from pyqula import geometry
from pyqula import gpu
from pyqula import operators
from pyqula.kpmtk import ldosmap
from pyqula.kpmtk.momenttoprofile import generate_profile

# The KPM local DOS of every orbital (kpmtk/ldosmap.py, mode="KPM" of
# h.get_ldos and h.get_multildos) is the Jackson-broadened Chebyshev
# expansion of <e_i|delta(E-H)|e_i>. Its moments are those of the
# eigenstates, mu_n(i) = sum_k |psi_k(i)|^2 T_n(E_k/scale), so a
# diagonalization gives the same map to roundoff, by a route that shares
# nothing with the block recursion; and the recursion truncated to the
# light cone of the expansion must give the map of the whole system.

ES = np.linspace(-2., 2., 9)


def _eigenstate_map(h, energies, scale, npol, ks, op=None):
    """The KPM map from the eigenstates of H(k), (ne,norb)"""
    nm = 2*npol
    basis = ldosmap.kpm_ldos_basis(energies, scale, nm)
    hk = h.get_hk_gen()
    out = 0.
    for k in ks:
        m = hk(k)
        m = m.toarray() if hasattr(m, "toarray") else np.array(m)
        es, vs = np.linalg.eigh(m)
        if op is None: w = np.abs(vs)**2
        else: w = (np.conj(vs)*(op@vs)).real # Re conj(psi_i) (A psi)_i
        T = np.cos(np.arange(nm)[:, None]*np.arccos(es/scale)[None, :])
        out = out + (w@T.T)@basis
    return (out/len(ks)).T


def _rashba_island(n=4, seed=1):
    """A disordered honeycomb island with Rashba coupling and an exchange
    field in a generic direction, so that H is complex and every spin
    component is mixed"""
    rng = np.random.default_rng(seed)
    g = geometry.honeycomb_lattice().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.3)
    h.add_exchange([0.2, 0.1, 0.3])
    h.add_onsite(rng.random(len(g.r)) - 0.5)
    return h


def _hamiltonians():
    rng = np.random.default_rng(2)
    g = geometry.honeycomb_lattice().get_supercell(5)
    g.dimensionality = 0
    spinless = g.get_hamiltonian(has_spin=False, is_sparse=True)
    spinless.add_onsite(rng.random(len(g.r)) - 0.5)
    g = geometry.triangular_lattice().get_supercell(3)
    g.dimensionality = 0
    nambu = g.get_hamiltonian(has_spin=True)
    nambu.add_rashba(0.3)
    nambu.add_exchange([0.2, 0., 0.1])
    nambu.add_swave(0.3)
    g = geometry.honeycomb_lattice().get_supercell(2)
    periodic = g.get_hamiltonian(has_spin=True)
    periodic.add_rashba(0.3)
    periodic.add_haldane(0.1)
    periodic.add_onsite(rng.random(len(g.r)) - 0.5)
    return dict(spinless=spinless, rashba=_rashba_island(), nambu=nambu,
            periodic=periodic)


@pytest.mark.parametrize("name", ["spinless", "rashba", "nambu", "periodic"])
def test_the_map_is_the_expansion_of_the_eigenstates(name):
    h = _hamiltonians()[name]
    ks = [[0., 0., 0.]] if h.dimensionality == 0 else \
            [list(k) for k in np.random.default_rng(3).random((4, 3))]
    ref = np.array([h.full2profile(d) for d in
        _eigenstate_map(h, ES, 8., 40, ks)])
    for (ie, e) in enumerate(ES[::4]):
        d = h.get_ldos(e=e, mode="KPM", npol=40, scale=8., ks=ks, nrep=1,
                write=False)[2]
        assert np.max(np.abs(d - ref[4*ie])) < 1e-12
    assert np.max(ref) > 0.05 # the comparison is not between zeros


def test_the_maps_at_many_energies_come_from_one_expansion():
    h = _rashba_island()
    (_, _, es, maps) = h.get_multildos(energies=ES, mode="KPM", npol=40,
            scale=8., write=False)
    ref = [h.full2profile(d) for d in
            _eigenstate_map(h, ES, 8., 40, [[0., 0., 0.]])]
    assert np.max(np.abs(maps - np.array(ref))) < 1e-12


def test_the_operator_map_is_the_local_matrix_element():
    """sx couples the two spin orbitals of a site, so the pairs it needs
    are not the diagonal ones"""
    h = _rashba_island()
    op = h.get_operator("sx")
    a = op.get_matrix()
    a = a.toarray() if hasattr(a, "toarray") else np.array(a)
    ref = _eigenstate_map(h, ES, 8., 40, [[0., 0., 0.]], op=a)
    (_, _, _, maps) = h.get_multildos(energies=ES, mode="KPM", npol=40,
            scale=8., operator="sx", write=False)
    assert np.max(np.abs(maps - np.array([h.full2profile(d) for d in ref]))) \
            < 1e-12
    assert np.max(np.abs(ref)) > 0.05


def test_each_orbital_integrates_to_one():
    """With E = scale cos(theta) the measure of the expansion cancels, and
    the midpoint rule in theta integrates every Chebyshev term exactly, so
    the integral of the local DOS of an orbital is its zeroth moment, 1"""
    h = _rashba_island()
    scale = 8.
    theta = (np.arange(200) + 0.5)*np.pi/200 # 200 > 2*npol midpoints
    e = scale*np.cos(theta)
    (_, _, _, maps) = h.get_multildos(energies=e, mode="KPM", npol=40,
            scale=scale, write=False)
    w = scale*np.sin(theta)*(theta[1] - theta[0]) # dE
    per_site = np.sum(maps*w[:, None], axis=0)
    assert np.max(np.abs(per_site - 2.)) < 1e-10 # two spin orbitals a site


def test_the_basis_is_the_profile_of_the_package():
    rng = np.random.default_rng(0)
    mus = rng.normal(size=60)*np.exp(-np.arange(60)/20.)
    xs = np.linspace(-0.95, 0.95, 31)
    scale = 3.
    ref = generate_profile(mus, xs).real/scale
    basis = ldosmap.kpm_ldos_basis(xs*scale, scale, len(mus))
    assert np.max(np.abs(mus@basis - ref)) < 1e-12


def _large_island(seed=4):
    """A disordered island much wider than the light cone of npol=12"""
    rng = np.random.default_rng(seed)
    g = geometry.honeycomb_lattice().get_supercell(24)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.3)
    h.add_exchange([0.2, 0.1, 0.3])
    h.add_onsite(rng.random(len(g.r)) - 0.5)
    return h


@pytest.mark.parametrize("operator", [None, "sx"])
def test_the_light_cone_truncation_is_exact(operator):
    """2*npol moments only need the vector of an orbital up to npol-1
    steps, so the recursion on the ball of npol-1 hops around each tile
    gives the map of the whole system; a radius well inside the light
    cone gives another one, which shows that the truncation is on"""
    h = _large_island()
    op = None if operator is None else h.get_operator(operator)
    kw = dict(npol=12, scale=8., operator=op)
    full = ldosmap.ldos_map(h, ES, [[0., 0., 0.]], full=True, **kw)[0]
    cone = ldosmap.ldos_map(h, ES, [[0., 0., 0.]], **kw)[0]
    inside = ldosmap.ldos_map(h, ES, [[0., 0., 0.]], kpm_radius=4, **kw)[0]
    assert np.max(np.abs(cone - full)) < 1e-12
    assert np.max(np.abs(inside - full)) > 1e-4


def test_the_truncation_follows_the_bonds_across_the_cell():
    """A periodic supercell wider than the light cone, on a k-mesh: the
    ball of a site at the edge of the cell continues through the bonds
    that leave it"""
    rng = np.random.default_rng(5)
    g = geometry.honeycomb_lattice().get_supercell(16)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    h.add_haldane(0.1)
    h.add_onsite(rng.random(len(g.r)) - 0.5)
    ks = [list(k) for k in rng.random((3, 3))]
    full = ldosmap.ldos_map(h, ES, ks, npol=10, scale=5., full=True)[0]
    cone = ldosmap.ldos_map(h, ES, ks, npol=10, scale=5.)[0]
    assert np.max(np.abs(cone - full)) < 1e-12


def test_the_device_map_agrees_with_the_cpu_one():
    """With the switch on, the block recursion runs in jax (on its CPU
    backend where there is no card)"""
    h = _large_island()
    cpu = ldosmap.ldos_map(h, ES, [[0., 0., 0.]], npol=12, scale=8.)[0]
    was = gpu.get_gpu()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # no GPU on this machine
            gpu.set_gpu(True)
        dev = ldosmap.ldos_map(h, ES, [[0., 0., 0.]], npol=12, scale=8.,
                kpm_prec="double")[0]
    finally:
        gpu.set_gpu(was)
    assert np.max(np.abs(dev - cpu)) < 1e-12


def test_the_options_are_checked():
    h = _rashba_island()
    with pytest.raises(ValueError, match="'KPM'"):
        h.get_ldos(mode="kpm", write=False)
    with pytest.raises(ValueError, match="'KPM'"):
        h.get_multildos(mode="ed", write=False)
    with pytest.raises(ValueError, match="not both"):
        h.get_ldos(mode="KPM", delta=0.1, npol=50, write=False)
    with pytest.raises(TypeError, match="npl"):
        h.get_multildos(mode="KPM", npl=50, write=False)
    with pytest.raises(TypeError, match="num_bands"):
        h.get_ldos(mode="KPM", npol=20, num_bands=10, write=False)
    with pytest.raises(ValueError, match="whole number"):
        h.get_ldos(mode="KPM", npol=2.5, write=False)
    op = operators.Operator(lambda v, k=None: v) # no matrix
    with pytest.raises(NotImplementedError, match="k-independent"):
        h.get_ldos(mode="KPM", npol=20, operator=op, write=False)
