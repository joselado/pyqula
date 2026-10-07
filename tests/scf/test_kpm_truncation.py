"""The truncated recursion of the KPM mean field (kpm_radius,
kpmtk/truncation.py): every tile of starting orbitals runs on the
Hamiltonian restricted to the sites within kpm_radius hops of it.

What has to hold: a radius that covers the system is the full recursion,
the hops are those between sites (a spin flip or an electron-hole entry on
one site is not a hop, a bond across a periodic cell is), and in a gapped
state the density matrix and the self-consistent mean field converge to
the full ones as the radius grows. A metal at zero temperature has no such
convergence and is not tested.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyqula import geometry
from pyqula.kpmtk import densitymatrix_kpm as D
from pyqula.kpmtk import truncation
from pyqula.scftk import sparsemeanfield


def _gapped_island(n=6):
    """A spinful honeycomb island with Rashba coupling and a sublattice
    imbalance, which opens a gap at half filling"""
    g = geometry.honeycomb_lattice().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.2)
    h.add_sublattice_imbalance(0.6)
    return h


def _hubbard_entries(h):
    v = sparsemeanfield.interaction(h, U=3.*np.ones(len(h.geometry.r)))
    return sparsemeanfield.needed_entries(v, h.intra.shape[0])


def _largest(a, b):
    return max(abs(a[d] - b[d]).max() for d in a)


def test_a_radius_covering_the_island_is_the_full_recursion():
    """With every site inside every region, the truncated recursion is the
    full one, density matrix and trace alike, up to the roundoff of a
    different split into blocks"""
    h = _gapped_island()
    needed = _hubbard_entries(h)
    full, (s1, m1) = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, trace=True)
    cut, (s2, m2) = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, trace=True,
            radius=1000)
    assert s1 == s2
    assert _largest(full, cut) < 1e-12
    assert np.max(np.abs(m1 - m2)) < 1e-12


def test_a_periodic_cell_is_truncated_across_its_boundary():
    """On a supercell with a k-mesh the bonds across the cell are hops of
    the site graph, so a radius covering the cell is the full recursion at
    every k-point"""
    g = geometry.square_lattice().get_supercell(8) # two tiles
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.3)
    h.add_exchange([0.2, 0., 0.3])
    needed = _hubbard_entries(h)
    full = D.get_dm_kpm_sparse(h, needed, nk=3, npol=40)
    cut = D.get_dm_kpm_sparse(h, needed, nk=3, npol=40, radius=8)
    assert _largest(full, cut) < 1e-12
    short = D.get_dm_kpm_sparse(h, needed, nk=3, npol=40, radius=1)
    assert _largest(full, short) > 1e-6 # a shorter radius does truncate


def test_the_hops_are_between_sites():
    """The site graph joins sites, not orbitals: the onsite spin flip of an
    exchange field and the onsite pairing of a Nambu Hamiltonian are not
    hops, so on a chain the ball of radius r holds 2r+1 sites whatever the
    orbitals per site, and a periodic bond is one hop"""
    g = geometry.chain().get_supercell(12)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_exchange([0.3, 0., 0.])
    h.setup_nambu_spinor()
    h.add_swave(0.2)
    adj = truncation.site_graph([csr_matrix(h.intra)], len(g.r))
    for r in range(4):
        assert len(truncation.ball(adj, [6], r)) == 2*r + 1
    gp = geometry.chain().get_supercell(12) # periodic
    hp = gp.get_hamiltonian(is_sparse=True)
    adj = truncation.site_graph([csr_matrix(hp.get_hk_gen()([0.3, 0., 0.]))],
            len(gp.r))
    assert set(truncation.ball(adj, [0], 1)) == {11, 0, 1}


def test_the_density_matrix_converges_with_the_radius_in_a_gapped_state():
    """In a gapped state the density matrix decays exponentially, and so
    does the error of the truncation with the radius: here by more than a
    factor of three for every two hops"""
    h = _gapped_island(8)
    needed = _hubbard_entries(h)
    full = D.get_dm_kpm_sparse(h, needed, nk=1, npol=100)
    err = [_largest(full, D.get_dm_kpm_sparse(h, needed, nk=1, npol=100,
            radius=r)) for r in [2, 4, 6, 8]]
    assert all(b < a/3. for (a, b) in zip(err, err[1:])), err
    assert err[-1] < 1e-3


def test_the_mean_field_converges_with_the_radius():
    """The self-consistent antiferromagnet of a half-filled honeycomb
    island, gapped, through h.get_mean_field_hamiltonian(integration="kpm"):
    the exchange field of the truncated loop approaches the full one as the
    radius grows"""
    g = geometry.honeycomb_lattice().get_supercell(6)
    g.dimensionality = 0
    def field(radius):
        h = g.get_hamiltonian(has_spin=True, is_sparse=True)
        hmf = h.get_mean_field_hamiltonian(U=3., filling=0.5,
                mf="antiferro", integration="kpm", npol=100, mix=0.5,
                maxite=200, kpm_radius=radius, verbose=0, write=False)
        return np.array([hmf.extract(c) for c in ["mx", "my", "mz"]]).T
    full = field(None)
    assert np.mean(np.linalg.norm(full, axis=1)) > 0.5 # an antiferromagnet
    err = [np.max(np.abs(field(r) - full)) for r in [2, 4, 6]]
    assert err[1] < err[0]/2. and err[2] < err[1]/2., err


def test_a_nambu_loop_takes_the_radius_everywhere():
    """A Nambu loop truncates the density matrix, the search of the
    electron-only Fermi level and the energy; a radius covering the island
    gives the loop without truncation, and a radius of one hop moves all
    three"""
    g = geometry.triangular_lattice().get_supercell(6) # several tiles
    g.dimensionality = 0
    from pyqula import meanfield
    def run(radius):
        h = g.get_hamiltonian(has_spin=True, is_sparse=True)
        h.add_exchange([0.1, 0., 0.])
        h.setup_nambu_spinor()
        np.random.seed(1)
        return meanfield.VJinteraction(h, U=-3., filling=0.3, mf="swave",
                integration="kpm", npol=60, maxite=4, kpm_radius=radius,
                verbose=0, write=False)
    a, b, c = run(None), run(100), run(1)
    assert abs(a.total_energy - b.total_energy) < 1e-10
    assert _largest(a.mf, b.mf) < 1e-10
    assert abs(a.hamiltonian.fermi - b.hamiltonian.fermi) < 1e-10
    assert abs(a.total_energy - c.total_energy) > 1e-4
    assert _largest(a.mf, c.mf) > 1e-4
    assert abs(a.hamiltonian.fermi - c.hamiltonian.fermi) > 1e-4


@pytest.mark.parametrize("radius", [-1, 2.5, True, "3"])
def test_a_radius_that_is_not_a_whole_number_of_hops_is_refused(radius):
    with pytest.raises(ValueError, match="whole number of hops"):
        truncation.check_radius(radius)


def test_the_device_runs_every_tile_in_one_batch(monkeypatch):
    """With the switch on, the tiles go to the device kernel as members of
    a batch, all of one shape per kind, so that it compiles once whatever
    the sizes of the regions, also when they are handed over in several
    groups (forced here to one tile per group and one member per call),
    and give the values of the CPU (on jax's CPU backend, where there is
    no card)"""
    import warnings
    from pyqula import gpu
    from pyqula.kpmtk import pairmomentsjax
    h = _gapped_island()
    needed = _hubbard_entries(h)
    cpu = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, radius=3,
            kpm_prec="double")
    shapes = dict()
    kernel = pairmomentsjax._contracted_each
    def recorded(*args):
        shapes.setdefault(args[9], []).append(tuple(np.shape(a)
            for a in args if hasattr(a, "shape")))
        return kernel(*args)
    monkeypatch.setattr(pairmomentsjax, "_contracted_each", recorded)
    monkeypatch.setattr(pairmomentsjax, "_MAX_BLOCK", 1)
    monkeypatch.setattr(truncation, "_TILES_PER_CALL", 1)
    was = gpu.get_gpu()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # no GPU on this machine
            gpu.set_gpu(True)
        dev = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, radius=3,
                kpm_prec="double")
    finally:
        gpu.set_gpu(was)
    assert _largest(cpu, dev) < 1e-12
    assert sum(len(x) for x in shapes.values()) > 2 # the tiles of the island
    assert all(len(set(x)) == 1 for x in shapes.values()), shapes


def test_the_tiles_in_groups_give_the_same_values(monkeypatch):
    """The restricted matrices are built for one group of tiles at a time,
    to bound the memory; the size of the groups is bookkeeping only, and so
    is running the tiles one after the other, with their rows split among
    the threads, which the CPU does when one tile per thread does not fit
    in memory"""
    from pyqula.kpmtk import pairmomentsnumba
    h = _gapped_island()
    needed = _hubbard_entries(h)
    whole = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, radius=3)
    monkeypatch.setattr(truncation, "_TILES_PER_CALL", 1)
    pieces = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, radius=3)
    assert _largest(whole, pieces) < 1e-13
    monkeypatch.setattr(pairmomentsnumba, "_MAX_BLOCK", 1)
    monkeypatch.setattr(pairmomentsnumba, "_SERIAL_BLOCK", 0)
    one_by_one = D.get_dm_kpm_sparse(h, needed, nk=1, npol=60, radius=3)
    assert _largest(whole, one_by_one) < 1e-12


def test_the_regions_are_found_once_per_site_graph(monkeypatch):
    """A mean-field loop asks for the same regions every iteration, which
    are kept from the previous call while the site graph, the pairs and
    the radius stay; a new radius finds them again"""
    h = _gapped_island()
    needed = _hubbard_entries(h)
    calls = []
    find = truncation._find_regions
    def counted(*args):
        calls.append(1)
        return find(*args)
    monkeypatch.setattr(truncation, "_find_regions", counted)
    monkeypatch.setattr(truncation, "_REGIONS", dict())
    a = D.get_dm_kpm_sparse(h, needed, nk=1, npol=40, radius=3)
    h2 = h.copy()
    h2.add_onsite(0.1) # the values change, the site graph does not
    D.get_dm_kpm_sparse(h2, needed, nk=1, npol=40, radius=3)
    assert len(calls) == 1
    D.get_dm_kpm_sparse(h, needed, nk=1, npol=40, radius=4)
    assert len(calls) == 2
    b = D.get_dm_kpm_sparse(h, needed, nk=1, npol=40, radius=3)
    assert len(calls) == 2 and _largest(a, b) == 0.
