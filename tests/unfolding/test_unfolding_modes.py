"""The unfolding operator through every mode of the routines it is advertised
for. The projector itself is pinned in test_unfolding_projector.py; the
tests here check that each consumer hands it the right kpoint and reports
the same spectral weight whichever mode it computes it in."""

import numpy as np
import pytest
from scipy.linalg import eigh

from pyqula import geometry
from pyqula.unfolding import get_supercell_map, bloch_phase_matrix


def defective_supercell(setup="spinless"):
    """A non-diagonal honeycomb supercell with an onsite defect, so that no
    two states are degenerate at a generic kpoint"""
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell([[2, 1, 0], [0, 1, 0], [0, 0, 1]], store_primal=True)
    h = g.get_hamiltonian(has_spin=(setup != "spinless"))
    h.add_onsite(lambda r: 0.7 * (np.sum((r - g.r[0])**2) < 1e-2))
    if setup == "nambu":
        h.setup_nambu_spinor()
    return h


def unfold_matrix(h, k):
    """O(k) = P^dagger P, built explicitly"""
    g0 = h.geometry.primal_geometry
    nf = h.intra.shape[0] // len(h.geometry.r)
    M, rep, prim = get_supercell_map(h.geometry, g0)
    P = bloch_phase_matrix(len(g0.r) * nf, rep, prim, nf, M)(k)
    P = P.conjugate().toarray()
    return P.conj().T @ P


def unfolded_weight(h, k, e, delta, nstates=None):
    """sum_n <n|O(k)|n> delta/((e-E_n)^2+delta^2) by brute force, over all
    the states or over the nstates closest to e"""
    E, V = eigh(h.get_hk_gen()(k))
    w = np.real(np.einsum("in,ij,jn->n", V.conj(), unfold_matrix(h, k), V))
    keep = np.argsort(np.abs(E - e))[0:nstates]
    return np.sum(w[keep] * delta / ((e - E[keep])**2 + delta**2))


@pytest.mark.parametrize("mode", ["eigen", "full"])
def test_unfolded_fermi_surface_does_not_depend_on_the_mode(mode, tmp_path,
                                                            monkeypatch):
    """mode='full' used to return twice, and mode='eigen' 1/pi times, the
    weight that the same routine returns without an operator"""
    monkeypatch.chdir(tmp_path)
    h = defective_supercell()
    e, delta = 0.3, 0.2
    kx, ky, d = h.get_fermi_surface(nk=4, e=e, delta=delta, mode=mode,
                                    operator="unfold", reciprocal=False,
                                    write=False)
    ref = [unfolded_weight(h, np.array([x, y, 0.]), e, delta)
           for (x, y) in zip(kx, ky)]
    assert np.allclose(d, ref)


def test_unfolded_fermi_surface_lowest_mode(tmp_path, monkeypatch):
    """mode='lowest' keeps the num_waves states closest to the energy, so
    it is the brute-force sum over those. The kpoints are shifted off the
    high-symmetry ones, where ARPACK is not reliable for degenerate states"""
    monkeypatch.chdir(tmp_path)
    h = defective_supercell(setup="spinful")
    e, delta, nw = 0.3, 0.2, 6
    k0 = np.array([0.137, 0.291])
    kx, ky, d = h.get_fermi_surface(nk=3, e=e, delta=delta, mode="lowest",
                                    num_waves=nw, operator="unfold", k0=k0,
                                    reciprocal=False, write=False)
    ref = [unfolded_weight(h, np.array([x + k0[0], y + k0[1], 0.]), e, delta,
                           nstates=nw) for (x, y) in zip(kx, ky)]
    assert np.allclose(d, ref, rtol=1e-4)


def test_unfolded_multi_fermi_surface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = defective_supercell()
    es, delta = [-0.4, 0.3], 0.2
    out = h.get_multi_fermi_surface(nk=4, energies=es, delta=delta,
                                    operator="unfold", reciprocal=False,
                                    write=False)
    for (e, (kx, ky, d)) in zip(es, out):
        ref = [unfolded_weight(h, np.array([x, y, 0.]), e, delta)
               for (x, y) in zip(kx, ky)]
        assert np.allclose(d, ref)


@pytest.mark.parametrize("setup", ["spinless", "spinful", "nambu"])
def test_unfolded_kdos_is_the_same_from_eigenvectors_and_green_function(
        setup, tmp_path, monkeypatch):
    """mode='ED' weights each eigenstate by <n|O|n>, mode='green' takes
    Tr[O G]; for a Hermitian Hamiltonian the two are the same number"""
    monkeypatch.chdir(tmp_path)
    h = defective_supercell(setup)
    kpath = np.array([[0.37, 0.81, 0.], [0.12, -0.4, 0.]])
    energies, delta = np.linspace(-2., 2., 7), 0.2
    ed = h.get_kdos_bands(kpath=kpath, operator="unfold", mode="ED",
                          delta=delta, energies=energies)[2]
    gr = h.get_kdos_bands(kpath=kpath, operator="unfold", mode="green",
                          delta=delta, energies=energies)[2]
    assert np.allclose(ed, gr)
    ref = [unfolded_weight(h, k, e, delta) / np.pi
           for k in kpath for e in energies]
    assert np.allclose(ed, ref)


def test_response_qpi_refuses_what_it_would_drop(tmp_path, monkeypatch):
    """mode='response' takes neither an operator nor nunfold, and used to
    return the plain supercell QPI when given either"""
    monkeypatch.chdir(tmp_path)
    g = geometry.square_lattice().get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        h.get_qpi(mode="response", operator="unfold", nk=4, energies=[0.])
    with pytest.raises(NotImplementedError):
        h.get_qpi(mode="response", nunfold=2, nk=4, energies=[0.])


# three supercells of the primal cell's shape and not: n x n, the rotated
# sqrt(3) x sqrt(3), and one whose two vectors have different lengths
SUPERCELLS = [("triangular_lattice", [[3, 0, 0], [0, 3, 0], [0, 0, 1]]),
              ("triangular_lattice", np.sqrt(3)),
              ("honeycomb_lattice", [[2, 1, 0], [0, 1, 0], [0, 0, 1]])]


def clean_supercell(lattice, M):
    """A supercell and its primal cell. The primal cell is the one the
    supercell stores, and not the geometry it was built from: a float size
    returns a cell rotated so that its first vector lies along x, and the
    primal geometry it stores is rotated with it, so that the two share a
    Cartesian frame and a Fermi surface drawn in it"""
    g0 = getattr(geometry, lattice)()
    g = g0.get_supercell(M, store_primal=True)
    nrep = int(round(abs(np.linalg.det(np.array(g.supercell_matrix,
                                                   dtype=float)))))
    return g.primal_geometry.get_hamiltonian(has_spin=False), \
        g.get_hamiltonian(has_spin=False), nrep


@pytest.mark.parametrize("lattice,M", SUPERCELLS)
@pytest.mark.parametrize("mode", ["eigen", "full"])
def test_unfolded_fermi_surface_on_the_primal_mesh_is_the_primal_one(
        lattice, M, mode, tmp_path, monkeypatch):
    """A clean supercell unfolds onto its primal cell exactly: the states
    at M@k0 that carry the weight are the primal ones at k0, each with
    weight N_rep. The mesh drawn through the supercell's own get_k2K is a
    different set of momenta, sheared for the third cell"""
    monkeypatch.chdir(tmp_path)
    h0, h, nrep = clean_supercell(lattice, M)
    kw = dict(nk=5, e=0.3, delta=0.2, write=False, mode=mode, nsuper=1,
              k0=np.array([0.05, 0.11]))
    kx0, ky0, d0 = h0.get_fermi_surface(**kw)
    kx, ky, d = h.get_fermi_surface(operator="unfold", primal_mesh=True, **kw)
    assert np.allclose(kx, kx0) and np.allclose(ky, ky0)
    assert np.allclose(d / nrep, d0, atol=1e-10)


@pytest.mark.parametrize("lattice,M", SUPERCELLS)
def test_unfolded_multi_fermi_surface_on_the_primal_mesh(lattice, M,
                                                         tmp_path,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    h0, h, nrep = clean_supercell(lattice, M)
    kw = dict(nk=5, energies=[-0.8, 0.3], delta=0.2, write=False)
    out0 = h0.get_multi_fermi_surface(**kw)
    out = h.get_multi_fermi_surface(operator="unfold", primal_mesh=True, **kw)
    for ((kx0, ky0, d0), (kx, ky, d)) in zip(out0, out):
        assert np.allclose(kx, kx0) and np.allclose(ky, ky0)
        assert np.allclose(d / nrep, d0, atol=1e-10)


def test_primal_mesh_needs_the_primal_geometry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.triangular_lattice().get_supercell(2).get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_fermi_surface(nk=3, primal_mesh=True, write=False)


def qpi_output():
    """DOS.OUT and the MULTIQPI maps that get_qpi wrote, in energy order"""
    import glob
    dos = np.genfromtxt("DOS.OUT")
    files = glob.glob("MULTIQPI/MULTIQPI_*_.OUT")
    files = sorted(files, key=lambda f: float(f.split("_")[-2]))
    return dos, [np.genfromtxt(f) for f in files]


def test_poor_man_qpi_of_an_n_by_n_supercell_is_unchanged(tmp_path,
                                                         monkeypatch):
    """The workflow of examples/2d/multiqpi_unfold, an n x n supercell with
    nunfold=n: sampling the primal zone and mapping it with M is the mesh
    the old sample-the-supercell-and-divide-by-n built, so the numbers
    recorded from that implementation still come out"""
    monkeypatch.chdir(tmp_path)
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 3. * (np.sum((r - g.r[0])**2) < 1e-2))
    h.get_qpi(mode="pm", operator="unfold", nunfold=2, nsuper=2, nk=8,
              energies=[-0.5, 0.3], delta=0.3)
    dos, maps = qpi_output()
    assert np.allclose(dos[:, 1], [218.85925759, 109.17408297], rtol=1e-8)
    assert np.allclose([np.sum(m[:, 2]) for m in maps],
                       [747191.3805584055, 185498.87560630124], rtol=1e-8)


def test_poor_man_qpi_unfolds_a_sqrt3_supercell(tmp_path, monkeypatch):
    """The sqrt(3) x sqrt(3) cell is rotated with respect to the primal
    one, which the old divide-by-nunfold could not unfold. A clean one has
    to give the primal QPI: the Fermi surface N_rep times the primal one,
    the QPI, quadratic in it, N_rep**2 times"""
    monkeypatch.chdir(tmp_path)
    h0, h, nrep = clean_supercell("triangular_lattice", np.sqrt(3))
    kw = dict(mode="pm", nsuper=2, nk=8, energies=[-0.5, 0.3], delta=0.3)
    h0.get_qpi(**kw)
    dos0, maps0 = qpi_output()
    h.get_qpi(operator="unfold", nunfold=np.sqrt(3), **kw)
    dos, maps = qpi_output()
    assert np.allclose(dos[:, 1], nrep * dos0[:, 1], rtol=1e-8)
    for (m, m0) in zip(maps, maps0):
        assert np.allclose(m[:, 0:2], m0[:, 0:2])  # the same q-points
        assert np.allclose(m[:, 2], nrep**2 * m0[:, 2], rtol=1e-6,
                           atol=1e-6 * np.max(np.abs(m0[:, 2])))


def test_poor_man_qpi_refuses_a_size_that_is_not_the_supercell(tmp_path,
                                                              monkeypatch):
    """nunfold is read as get_supercell reads a size, nunfold**2 cells"""
    monkeypatch.chdir(tmp_path)
    g0 = geometry.triangular_lattice()
    for (M, nunfold) in [(3, 2), (np.sqrt(3), 2)]:
        h = g0.get_supercell(M, store_primal=True).get_hamiltonian(
                has_spin=False)
        with pytest.raises(ValueError):
            h.get_qpi(mode="pm", operator="unfold", nunfold=nunfold, nk=4,
                      energies=[0.])
    # a fresh lattice: store_primal stores the primal copy on the geometry
    # the supercell is built from, so every later supercell of g0 has it
    g1 = geometry.triangular_lattice()
    h = g1.get_supercell(2).get_hamiltonian(has_spin=False)  # no primal
    with pytest.raises(ValueError):
        h.get_qpi(mode="pm", nunfold=2, nk=4, energies=[0.])
