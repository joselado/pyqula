"""The real-space hoppings of get_wannier_hamiltonian sit on the cells of the
Wigner-Seitz cell of the Born-von Karman supercell, each divided by its
degeneracy, which is Wannier90's construction (hamiltonian_wigner_seitz).

These replace a plain box of nk cells per direction, taken from
np.fft.fftfreq. For an even nk that box holds the cell R=-nk/2 but not
R=+nk/2, so the hopping to one of them had no Hermitian partner and the
Wannier Hamiltonian was not Hermitian away from the mesh: an anti-Hermitian
part of 2e-2 for the gapped honeycomb valence band on an 8x8 mesh, and of
2.6e-2 for the BdG chain of the Wannierization notebook 04 on a mesh of 24.
On the mesh itself nothing changes, the mesh is reproduced exactly with
either set of cells.

What moves is the interpolation between the mesh points. The Wigner-Seitz
cells are the ones closest to the origin, a hexagon on the honeycomb lattice
rather than a parallelogram, so the largest error of the gapped honeycomb
valence band away from the mesh goes from 1.1e-2 to 3.2e-3 at nk=8, and from
2.0e-3 to 4.7e-4 at nk=11, where the box was already Hermitian. The BdG chain
of notebook 04 does not move (4.7e-2 at nk=24 either way): its error comes
from Wannier functions made wide by a 0.13 anticrossing with the next band,
and goes down with the mesh, to 1.8e-4 at nk=192. The Wannier functions
themselves (wannier_functions) are the same as before.
"""
import itertools

import numpy as np
import pytest

from pyqula import geometry
from pyqula.wanniertk import wannierize
from pyqula.wanniertk.wannierpy._engine.ws_vectors import wigner_seitz_vectors


def _gapped_honeycomb():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8])
    return h


def _bdg_chain():
    """The BdG chain of jupyter-notebooks/functionalities/wannierization/
    04_dimensionality_and_bdg.ipynb: a Rashba chain with exchange and s-wave
    pairing."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0., 0., 0.3])
    h.shift_fermi(1.0)
    h.add_swave(0.2)
    return h


def _offmesh_kpoints(dim, n=40):
    ks = np.zeros((n, 3))
    ks[:, :dim] = np.random.default_rng(1).random((n, dim))
    return ks


def _max_antihermitian(hw, ks):
    f = hw.get_hk_gen()
    return max(np.max(np.abs(f(k) - np.conj(f(k)).T)) for k in ks)


@pytest.mark.parametrize("nk", [8, 9])
def test_wannier_hamiltonian_is_hermitian_away_from_the_mesh(nk):
    """H_wan(k) must be Hermitian at every k, not only on the mesh, for an
    even nk (where the box of cells used before failed, by 2e-2 at nk=8)
    and an odd one (where it was already Hermitian)."""
    h = _gapped_honeycomb()
    hw = h.get_wannier_hamiltonian(bands=[0, 0], nk=nk, cutoff=0.0)
    assert _max_antihermitian(hw, _offmesh_kpoints(2)) < 1e-12


@pytest.mark.parametrize("nk", [8, 9])
def test_wannier_hamiltonian_reproduces_the_mesh_exactly(nk):
    """Every class of cells equivalent modulo the supercell carries a total
    weight of one, so the valence band is reproduced to machine precision
    at every mesh k-point, for an even and an odd nk."""
    h = _gapped_honeycomb()
    hw = h.get_wannier_hamiltonian(bands=[0, 0], nk=nk, cutoff=0.0)
    f0, fw = h.get_hk_gen(), hw.get_hk_gen()
    for k1, k2 in itertools.product(range(nk), repeat=2):
        k = np.array([k1 / nk, k2 / nk, 0.])
        assert abs(np.linalg.eigvalsh(fw(k))[0] - np.linalg.eigvalsh(f0(k))[0]) < 1e-10


def test_interpolation_between_mesh_points_is_symmetric():
    """The Wigner-Seitz cell of the honeycomb supercell is a hexagon, so the
    Wannier band keeps the symmetry of the valence band between the mesh
    points too: the error at k, -k, and k rotated by 60 degrees (in reduced
    coordinates (k1,k2) -> (k1+k2,-k1)) is the same. With the box of cells
    used before it was not, the box being a parallelogram."""
    h = _gapped_honeycomb()
    hw = h.get_wannier_hamiltonian(bands=[0, 0], nk=8, cutoff=0.0)
    f0, fw = h.get_hk_gen(), hw.get_hk_gen()

    def error(k1, k2):
        k = np.array([k1, k2, 0.])
        return np.linalg.eigvalsh(fw(k))[0] - np.linalg.eigvalsh(f0(k))[0]

    for k1, k2 in np.random.default_rng(2).random((5, 2)):
        images = [error(k1, k2), error(-k1, -k2), error(k1 + k2, -k1),
                  error(k2, k1)]
        assert np.max(np.abs(np.array(images) - images[0])) < 1e-10
        assert abs(images[0]) > 1e-8 # a point off the mesh, not a trivial check


def test_bdg_chain_is_hermitian_and_electron_hole_symmetric_away_from_the_mesh():
    """The BdG chain of the Wannierization notebook 04, on the 24-point mesh
    it ran on, where the box of cells gave an anti-Hermitian part of 2.6e-2. The
    electron-hole symmetry of every hopping, C h_R^* C^-1 = -h_R, still
    holds with the degeneracy weights, since ndegen(R) is real."""
    h = _bdg_chain()
    hw = h.get_wannier_hamiltonian(bands=[1, 2], nk=24, num_iter=1000)
    assert _max_antihermitian(hw, _offmesh_kpoints(1)) < 1e-12
    C = hw.wannier_particle_hole_operator
    Cinv = np.linalg.inv(C)
    for m in hw.get_multihopping().get_dict().values():
        assert np.max(np.abs(C @ np.conj(m) @ Cinv + m)) < 1e-8


def _check_cells(irvec, ndegen, mp_grid):
    """Inversion symmetric, with the same degeneracy for R and -R, and one
    unit of total weight per class of cells modulo the supercell."""
    cells = {tuple(r): n for r, n in zip(irvec, ndegen)}
    for r, n in cells.items():
        assert cells.get(tuple(-x for x in r)) == n
    weight = {}
    for r, n in cells.items():
        key = tuple(x % m for x, m in zip(r, mp_grid))
        weight[key] = weight.get(key, 0.) + 1. / n
    assert len(weight) == np.prod(mp_grid)
    assert all(abs(w - 1.) < 1e-12 for w in weight.values())


@pytest.mark.parametrize("lattice,nk", [
    ("honeycomb_lattice", (8, 8)), ("honeycomb_lattice", (11, 11)),
    ("honeycomb_lattice", (12, 4)), ("kagome_lattice", (12, 12)),
    ("square_lattice", (8, 8)), ("chain", (24,)), ("chain", (23,)),
    ("cubic_lattice", (4, 4, 4))])
def test_wigner_seitz_cells_are_those_of_wannier90(lattice, nk):
    """The cells and degeneracies are the ones of Wannier90's own search
    (the bundled port, wannierpy/_engine/ws_vectors.py) wherever that
    search succeeds."""
    h = getattr(geometry, lattice)().get_hamiltonian(has_spin=False)
    mp_grid = wannierize._mp_grid(h, list(nk))
    real_lattice = wannierize._real_lattice(h)
    irvec, ndegen = wannierize._wigner_seitz_cells(mp_grid, real_lattice, h.dimensionality)
    irvec90, ndegen90, _ = wigner_seitz_vectors(mp_grid, real_lattice)
    assert ({tuple(r): n for r, n in zip(irvec, ndegen)} ==
            {tuple(r): n for r, n in zip(irvec90, ndegen90)})
    _check_cells(irvec, ndegen, mp_grid)


def test_wigner_seitz_cells_of_a_skewed_anisotropic_supercell():
    """A mesh much denser along one lattice vector than the other of a
    non-orthogonal lattice gives a long, skewed supercell, whose
    Wigner-Seitz cell Wannier90's search (two supercells in every
    direction) does not cover; the cells are still complete here."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    real_lattice = wannierize._real_lattice(h)
    for nk in [(12, 2), (1, 16), (40, 1)]:
        mp_grid = wannierize._mp_grid(h, list(nk))
        irvec, ndegen = wannierize._wigner_seitz_cells(mp_grid, real_lattice, 2)
        _check_cells(irvec, ndegen, mp_grid)


def test_split_clusters_are_compared_on_the_returned_hoppings():
    """auto_split_clusters picks between the joint and the split
    Wannierization with _offmesh_reproduction_error, which builds the
    hoppings the same way as the returned Hamiltonian: the result is
    Hermitian away from the mesh and exact on it, whichever is kept."""
    g = geometry.ladder() # two bands with a gap between them at every k
    h = g.get_hamiltonian(has_spin=False)
    hw = h.get_wannier_hamiltonian(bands=[0, 1], nk=8, cutoff=0.0,
                                   auto_split_clusters=True)
    assert _max_antihermitian(hw, _offmesh_kpoints(1)) < 1e-12
    f0, fw = h.get_hk_gen(), hw.get_hk_gen()
    for k1 in range(8):
        k = np.array([k1 / 8., 0., 0.])
        assert np.max(np.abs(np.linalg.eigvalsh(fw(k)) - np.linalg.eigvalsh(f0(k)))) < 1e-10
