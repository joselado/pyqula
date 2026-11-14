"""The density matrix of the simplex integration (dmtk.simplexdm.get_dm)
is the contraction of the occupation weights of every state with its
projector and the Bloch phase of each lattice vector,
  dm[d]_ij = sum_(k,n) e^{2 pi i k.d} conj(u_n,i(k)) w_n(k) u_n,j(k).
The code builds the weighted projector of each k-point once and contracts
it with the phases of every direction in one product, and it diagonalizes
the mesh in batches whose size depends on the matrix size. Here the same
sum is done directly in numpy, with numpy's own diagonalization, on a metal
with many directions, also with batches that do not divide the mesh."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.dmtk import simplexdm
from pyqula.dostk.simplexdos import uniform_mesh, _degeneracy_tol

NK = 12


def _metal():
    h = geometry.honeycomb_lattice().get_supercell(2).get_hamiltonian(
            has_spin=False, tij=[1., 0.3])
    h.add_onsite(0.4) # a metal
    return h


def _direct(h, ds, T):
    ks = uniform_mesh(h.dimensionality, NK)
    hk = h.get_hk_gen()
    es, U = np.linalg.eigh(np.array([hk(k) for k in ks]))
    w = simplexdm.simplex_weights(h.dimensionality, NK, es, mu=0., T=T,
            bloechl=True)
    simplexdm._multiplet_average_real(es, w, _degeneracy_tol(es))
    out = {}
    for d in ds:
        ph = np.exp(2j*np.pi*(ks@np.array(d, dtype=float)))
        out[d] = np.einsum("k,kin,kn,kjn->ij", ph, U.conj(), w, U)
    return out


@pytest.mark.parametrize("batch", [None, 37])
@pytest.mark.parametrize("T", [0., 1e-2])
@pytest.mark.parametrize("directions", ["onsite", "one bond", "many"])
def test_the_density_matrix_is_the_direct_contraction(batch, T, directions,
        monkeypatch):
    """with one direction, where the phase goes into the weights (the
    onsite one, and one bond alone, whose phase is not 1), and
    with many, where the projector of each k-point is built once"""
    if batch is not None: # batches that do not divide the 144 k-points
        init = simplexdm._Mesh.__init__
        def small(self, h, nk, batch=batch): init(self, h, nk, batch=batch)
        monkeypatch.setattr(simplexdm._Mesh, "__init__", small)
    h = _metal()
    ds = [(0, 0, 0)]+[tuple(d) for d in h.get_dict() if tuple(d) != (0, 0, 0)]
    ds = ds+[(2, -1, 0)] # one more, with no hopping
    if directions == "onsite": ds = [(0, 0, 0)]
    if directions == "one bond": ds = [(1, 0, 0)]
    simplexdm._last_search = None
    dm = simplexdm.get_dm(h, ds, nk=NK, T=T)
    ref = _direct(h, ds, T)
    for d in ds:
        assert np.max(np.abs(dm[d]-ref[d])) < 1e-12, d
    if directions == "one bond": # a bond the phase matters for
        assert np.max(np.abs(ref[(1, 0, 0)])) > 1e-2
        return
    n = h.intra.shape[0]
    assert abs(np.trace(dm[(0, 0, 0)]).real-np.sum(ref[(0, 0, 0)].diagonal()
        ).real) < 1e-12
    assert 0.1*n < np.trace(dm[(0, 0, 0)]).real < 0.9*n # a metal


def test_the_batch_holds_a_fixed_number_of_matrix_entries():
    """the batch of the mesh shrinks with the matrix size, so that its
    arrays stay near 2^22 complex numbers each"""
    for m, n in ((1, 2), (5, 50)):
        h = geometry.square_lattice().get_supercell(m).get_hamiltonian(
                has_spin=True)
        mesh = simplexdm._Mesh(h, 128)
        assert mesh.n == n
        assert mesh.batch == min(2**14, 2**22//(n*n))
        assert mesh.batch*n*n <= 2**22
