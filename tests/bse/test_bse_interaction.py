"""The BSE kernel is built from the interaction the mean field was
converged with, which the SCF stores as h.V in a halved convention. Getting
that factor wrong scales the whole kernel, so it is checked here against
the SCF's own mean-field routine rather than against the algebra."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.interaction import bare_interaction, density_interaction
from pyqula.scftk.densitydensity import get_mf_normal


def _hubbard_dimer(U):
    g = geometry.chain().supercell(2)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    return h.get_mean_field_hamiltonian(U=U, filling=0.5, mf="random",
                                        maxerror=1e-8)


def test_bare_interaction_is_twice_the_stored_one():
    """h.V holds W/2, not W: for a Hubbard model the stored value is U/2
    on the up-down entries, and the bare interaction is U."""
    U = 1.7
    h = _hubbard_dimer(U)
    W = bare_interaction(h)
    assert abs(W[(0, 0, 0)][0, 1] - U) < 1e-10
    assert abs(h.V[(0, 0, 0)][0, 1] - U / 2.) < 1e-10


def test_bare_interaction_reproduces_the_scf_mean_field():
    """The real check on the factor of two: the interaction returned here,
    put into the textbook Hartree-Fock expressions, must give exactly the
    mean field the SCF's own get_mf_normal builds from h.V."""
    h = _hubbard_dimer(1.7)
    Wm = bare_interaction(h)[(0, 0, 0)]
    n = Wm.shape[0]
    np.random.seed(0)
    a = np.random.random((n, n)) + 1j * np.random.random((n, n))
    dm = {(0, 0, 0): a + a.conj().T}  # a generic Hermitian density matrix
    mf = get_mf_normal(h.V, dm)[(0, 0, 0)]
    D = dm[(0, 0, 0)]
    ref = -Wm * D.T                       # Fock:    -W_ij dm_ji
    ref = ref + np.diag(Wm @ np.diag(D))  # Hartree: +sum_j W_ij n_j
    assert np.max(np.abs(mf - ref)) < 1e-10


def test_density_interaction_matches_the_scf_convention():
    """density_interaction must produce the same interaction Vinteraction
    would have converged with, up to that same factor of two."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h = h.get_multicell().get_dense()
    U, V1 = 1.3, 0.7
    W = density_interaction(h, U=U, V1=V1)
    from pyqula.scftk.densitydensity import Vinteraction
    # rebuild the SCF's own v without running the SCF loop
    from pyqula import specialhopping
    nd = h.geometry.neighbor_distances()
    mg = specialhopping.distance_hopping_matrix([V1 / 2.], nd[0:1])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
                                    mgenerator=mg)
    for d, m in hv.get_hopping_dict().items():
        ns = m.shape[0]
        for i in range(ns):
            for j in range(ns):
                assert abs(W[d][2 * i, 2 * j] - 2. * m[i, j]) < 1e-10
    assert abs(W[(0, 0, 0)][0, 1] - U) < 1e-10


def test_spinless_hubbard_is_rejected():
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        density_interaction(h, U=1.0)


def test_missing_interaction_is_reported():
    """A Hamiltonian that never went through an SCF has no h.V, and the
    error must say so instead of failing somewhere in the kernel build."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="no interaction"):
        bare_interaction(h)


def _coulomb(r1, r2):
    return 0.6 / np.sqrt((r1 - r2).dot(r1 - r2) + 0.25)


def _pairs_by_distance(g, W, shift):
    """Distance of every nonzero pair of a {direction: matrix} interaction,
    with its value, reading every shift-th row and column (2 for a spinful
    one, where the four spin blocks of a pair carry the same value)"""
    out = []
    for d, m in W.items():
        R = d[0] * g.a1 + d[1] * g.a2 + d[2] * g.a3
        n = m.shape[0] // shift
        for i in range(n):
            for j in range(n):
                v = m[shift * i, shift * j]
                if abs(v) > 1e-12:
                    out.append((np.linalg.norm(g.r[j] + R - g.r[i]), v))
    return out


def _shell_sizes(g, rmax, ncells=12):
    """Number of pairs of sites at each distance up to rmax, counted over a
    region of cells far larger than rmax"""
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


@pytest.mark.parametrize("has_spin", [False, True])
@pytest.mark.parametrize("rcut", [5.0, 7.5])
def test_the_vr_tail_keeps_whole_distance_shells(has_spin, rcut):
    """A long-range Vr is kept for every pair up to rcut and for no pair
    beyond it, with every distance shell complete. The multicell builder
    alone kept an uneven fringe past its range (4 of the 6 pairs at
    distance 5 on the honeycomb lattice), which broke the three-fold
    rotation of the lattice."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=has_spin)
    W = density_interaction(h, Vr=_coulomb, rcut=rcut)
    pairs = _pairs_by_distance(g, W, 2 if has_spin else 1)
    kept = dict()
    for dist, v in pairs:
        assert dist < rcut + 1e-6
        # the value depends on the distance alone
        assert abs(v - _coulomb(np.zeros(3), np.array([dist, 0., 0.]))) < 1e-10
        kept[round(dist, 5)] = kept.get(round(dist, 5), 0) + 1
    assert kept == _shell_sizes(g, rcut)


def test_the_screened_nearest_neighbors_are_equivalent():
    """The three nearest-neighbor bonds of the honeycomb lattice are
    related by rotation, so the screened interaction must be the same on
    all of them; the uneven Vr fringe made them 0.506 and 0.486."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_sublattice_imbalance(1.0)
    V = density_interaction(h, U=1.0, Vr=_coulomb)
    W = h.get_screened_interaction(V=V, nk=8).get_dict()
    nn = [v for dist, v in _pairs_by_distance(g, W, 2) if abs(dist - 1.) < 1e-3]
    assert len(nn) == 6
    assert np.max(np.abs(np.array(nn) - nn[0])) < 1e-8


def test_a_nonpositive_rcut_is_rejected():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="rcut"):
        density_interaction(h, Vr=_coulomb, rcut=0.)
