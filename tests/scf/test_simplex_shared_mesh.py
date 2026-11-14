"""The Fermi search and the density matrix of one triangle mean-field
iteration share their mesh (dmtk.simplexdm._last_search). The loop finds
the Fermi level of h and asks for the density matrix of h shifted by it,
which has the same eigenvectors and the band energies shifted by a
constant. A search that keeps its mesh hands its band energies, its
eigenvectors and its simplices to the next density matrix, which is the
same to rounding as one computed from scratch, and only for a Hamiltonian
that is the searched one plus a constant times the identity, on the same
mesh. The record is taken by the next density matrix in any case, so
nothing is held past one iteration."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.dmtk import simplexdm
from pyqula.dostk import simplexdos
from pyqula.htk import eigenvectors
from pyqula.scftk.densitydensity import Vinteraction

NK = 24
FILLING = 0.3


def _metal(t2=0.2):
    """a two-band metal with first and second neighbors"""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False, tij=[1., t2])
    h.add_sublattice_imbalance(0.3)
    return h


def _ds(h):
    return [(0, 0, 0)]+[tuple(d) for d in h.get_dict() if tuple(d) != (0, 0, 0)]


def _fresh(h, T):
    """the density matrix with no record of a search"""
    simplexdm._last_search = None
    return simplexdm.get_dm(h, _ds(h), nk=NK, T=T, counted=True)


def _close(a, b, tol=1e-12):
    return max(np.max(np.abs(a[d]-b[d])) for d in a) <= tol


@pytest.mark.parametrize("T", [0., 1e-3])
@pytest.mark.parametrize("fits", [True, False])
def test_the_density_matrix_reuses_the_search_of_a_shifted_hamiltonian(T,
        fits, monkeypatch):
    """With the eigenvectors kept the density matrix needs no
    diagonalization, and without them (they do not fit in memory) one pass
    where it needed two, and it is the density matrix of a fresh call"""
    if not fits: monkeypatch.setattr(simplexdm, "max_memory_gb", 0.)
    h = _metal()
    mu = simplexdm.get_fermi4filling(h, FILLING, nk=NK, T=T, keep_mesh=True)
    assert simplexdm._last_search is not None
    h.shift_fermi(-mu)
    dm, ndiag = simplexdm.get_dm(h, _ds(h), nk=NK, T=T, counted=True)
    assert simplexdm._last_search is None # taken
    ref, nref = _fresh(h, T)
    assert ndiag == (0 if fits else NK**2)
    assert nref == (NK**2 if fits else 2*NK**2)
    assert _close(dm, ref)
    # and the charge is the requested one
    assert abs(np.trace(dm[(0, 0, 0)]).real-FILLING*2) < 1e-11


def test_another_hamiltonian_or_mesh_starts_from_scratch():
    h = _metal()
    simplexdm.get_fermi4filling(h, FILLING, nk=NK, keep_mesh=True)
    other = h.copy()
    other.add_sublattice_imbalance(0.1) # not a constant shift
    dm, ndiag = simplexdm.get_dm(other, _ds(other), nk=NK, counted=True)
    assert ndiag == NK**2 and simplexdm._last_search is None
    assert _close(dm, _fresh(other, 0.)[0], tol=0.)
    # nor is a change of the hoppings alone, with the same onsite matrix
    simplexdm.get_fermi4filling(h, FILLING, nk=NK, keep_mesh=True)
    hop = _metal(t2=0.25)
    assert np.array_equal(hop.intra, h.intra)
    dm, ndiag = simplexdm.get_dm(hop, _ds(hop), nk=NK, counted=True)
    assert ndiag == NK**2 and simplexdm._last_search is None
    assert _close(dm, _fresh(hop, 0.)[0], tol=0.)
    simplexdm.get_fermi4filling(h, FILLING, nk=NK, keep_mesh=True)
    dm, ndiag = simplexdm.get_dm(h, _ds(h), nk=NK+2, counted=True)
    assert ndiag == (NK+2)**2 and simplexdm._last_search is None
    # a search that is not asked to keep its mesh keeps nothing
    simplexdm.get_fermi4filling(h, FILLING, nk=NK)
    assert simplexdm._last_search is None


def test_a_search_that_cannot_recognize_its_hamiltonian_keeps_nothing(
        monkeypatch):
    """The next density matrix recognizes the searched Hamiltonian on its
    Bloch matrices. Without them the search stores nothing and computes no
    eigenvectors, and finds the same level"""
    h = _metal()
    ref = simplexdm.get_fermi4filling(h, FILLING, nk=NK)
    called = []
    monkeypatch.setattr(simplexdm._Mesh, "bloch_data", lambda self: None)
    pd = eigenvectors.parallel_diagonalization
    def diag(m, *a, **k): # the real one, counted
        called.append(len(m))
        return pd(m, *a, **k)
    monkeypatch.setattr(eigenvectors, "parallel_diagonalization", diag)
    mu = simplexdm.get_fermi4filling(h, FILLING, nk=NK, keep_mesh=True)
    assert simplexdm._last_search is None and not called
    assert abs(mu-ref) < 1e-12


def test_the_simplices_passed_must_belong_to_the_mesh():
    es = np.zeros((NK**2, 2))
    with pytest.raises(ValueError, match="do not belong"):
        simplexdm.simplex_weights(2, NK, es,
                simplex_mesh=simplexdos.simplices(2, NK+1))
    with pytest.raises(ValueError, match="do not belong"):
        simplexdm.fermi_from_energies(2, NK, es+np.arange(2), 0.3,
                simplex_mesh=simplexdos.simplices(3, 4))


def test_a_mean_field_iteration_diagonalizes_the_mesh_once(monkeypatch):
    """One plain iteration of a triangle mean field (maxite=0) diagonalizes
    the mesh once and builds its simplices once"""
    count = {"diag": 0, "sim": 0}
    pd, pv, si = (eigenvectors.parallel_diagonalization,
            eigenvectors.peigvalsh, simplexdos.simplices)
    def diag(m, *a, **k):
        count["diag"] += len(m)
        return pd(m, *a, **k)
    def vals(m, *a, **k):
        count["diag"] += len(m)
        return pv(m, *a, **k)
    def sim(*a, **k):
        count["sim"] += 1
        return si(*a, **k)
    monkeypatch.setattr(eigenvectors, "parallel_diagonalization", diag)
    monkeypatch.setattr(eigenvectors, "peigvalsh", vals)
    monkeypatch.setattr(simplexdos, "simplices", sim)
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    Vinteraction(h, U=6., filling=0.4, nk=NK, integration="triangle",
            mf="ferro", maxite=0, load_mf=False, write=False, verbose=0)
    # one iteration of the loop (a search and a density matrix) and the
    # band energy at the end, where the iteration diagonalized and built
    # the simplices twice
    assert count["diag"] == 2*NK**2
    assert count["sim"] == 2


def test_the_qtci_search_hands_its_simplices_to_one_density_matrix(
        monkeypatch):
    """The qtci search of the loop (keep_mesh=True) keeps the simplices of
    its 2^R mesh next to its eigenpairs; the density matrix of the shifted
    h takes them, so they are built once, and drops them, so they are not
    held past the iteration"""
    from pyqula.qtcitk import densitymatrix_qtci as dq
    count = [0]
    build = simplexdos.simplices
    def counted(*a, **k):
        count[0] += 1
        return build(*a, **k)
    monkeypatch.setattr(simplexdos, "simplices", counted)
    monkeypatch.setattr(dq, "_last_mesh", None)
    h = geometry.square_lattice().get_hamiltonian(has_spin=False)
    h.add_onsite(0.3)
    mu = dq.get_fermi4filling_qtci(h, FILLING, nk=16, T=1e-3, keep_mesh=True)
    h.shift_fermi(-mu)
    needed = {((0, 0, 0), 0, 0), ((1, 0, 0), 0, 0)}
    dm, ndiag, ranks = dq.dm_entries_qtci(h, needed, nk=16, T=1e-3,
            counted=True)
    assert ndiag == 0 and count[0] == 1
    assert dq._last_mesh["simplex_mesh"] is None
    # and the charge is the requested one, to the interpolation tolerance
    assert abs(dm[(0, 0, 0)][0, 0].real-FILLING) < 1e-6
    # a density matrix of another Hamiltonian (other hoppings, the same
    # onsite matrix) does not take the simplices of a search, starts from
    # scratch, and the simplices are dropped all the same
    dq.get_fermi4filling_qtci(h, FILLING, nk=16, T=1e-3, keep_mesh=True)
    assert dq._last_mesh["simplex_mesh"] is not None
    other = geometry.square_lattice().get_hamiltonian(has_spin=False,
            tij=[1., 0.3])
    dm, ndiag, ranks = dq.dm_entries_qtci(other, needed, nk=16, T=1e-3,
            counted=True)
    assert ndiag > 0
    assert dq._last_mesh["simplex_mesh"] is None
