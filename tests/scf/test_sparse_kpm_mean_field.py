import tracemalloc

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from pyqula import geometry, meanfield
from pyqula.kpmtk import pairmomentsjax, pairmomentsnumba
from pyqula.kpmtk.densitymatrix_kpm import get_band_energy_kpm
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm
from testutils import temporary_attr


def _dense(m):
    return m.toarray() if issparse(m) else np.asarray(m)


def _model(case, sparse):
    """The same model built as a dense and as a sparse Hamiltonian"""
    if case == "Rashba Hubbard island": # a non-collinear state
        g = geometry.square_lattice().get_supercell(4)
        g.dimensionality = 0
        h = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
        h.add_rashba(0.3)
        return h, dict(U=2.0, filling=0.5, nk=1), "ferro"
    if case == "honeycomb with V1":
        h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True,
                is_sparse=sparse)
        return h, dict(U=2.5, V1=0.5, filling=0.5, nk=4), "ferro"
    if case == "Nambu island": # pairing, Rashba and an in-plane field
        g = geometry.triangular_lattice().get_supercell(3)
        g.dimensionality = 0
        h = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
        h.add_rashba(0.2)
        h.add_exchange([0.1, 0., 0.])
        h.setup_nambu_spinor()
        return h, dict(U=-2.0, filling=0.4, nk=1), "swave"


def _run(case, sparse, **kw):
    h, model, mode = _model(case, sparse)
    np.random.seed(7)
    mf = meanfield.guess(_model(case, False)[0], mode=mode) # the same guess
    if not isinstance(mf, dict): mf = {(0, 0, 0): mf}
    mf = {k: (csr_matrix(m) if sparse else _dense(m)) for (k, m) in mf.items()}
    return Vinteraction_kpm(h, mf=mf, npol=60, mix=0.5, T=1e-2, verbose=0,
            write=False, **{**model, **kw}) # kw overrides the model


@pytest.mark.parametrize("case", ["Rashba Hubbard island",
    "honeycomb with V1", "Nambu island"])
def test_sparse_engine_follows_the_dense_one(case):
    """A sparse Hamiltonian goes through the sparse KPM engine
    (scftk/sparsemeanfield.py), which holds the interaction, the density
    matrix and the mean field as sparse matrices; it computes the same
    density matrix from the same moments, so the mean field after several
    iterations is the dense engine's to roundoff, and the Hamiltonian stays
    sparse throughout"""
    dense = _run(case, False, maxite=6)
    sparse = _run(case, True, maxite=6)
    assert issparse(sparse.hamiltonian.intra)
    for key in dense.mf:
        err = np.max(np.abs(_dense(dense.mf[key]) - _dense(sparse.mf[key])))
        assert err < 1e-10, (case, key, err)


def test_sparse_interaction_is_the_dense_one():
    """The interaction built with a KD-tree (V1, V2, V3 on their shells, U
    onsite, Vr up to rcut) is the one the dense builder makes by evaluating
    every pair, on a lattice whose shells sit within the two cells the
    dense builder looks at"""
    kw = dict(U=1.0, V1=0.4, V2=0.2, V3=0.1, Vr=lambda r1, r2:
            0.3*np.exp(-np.linalg.norm(r1 - r2)), rcut=2.5, maxite=0)
    v0 = _run("honeycomb with V1", False, **kw).v
    v1 = _run("honeycomb with V1", True, **kw).v
    assert sorted(v0) == sorted(v1)
    for key in v0:
        assert np.max(np.abs(_dense(v0[key]) - _dense(v1[key]))) < 1e-14


def test_vr_on_an_island_needs_rcut():
    """Vr with rcut=None on a finite system means every pair of sites,
    N^2 bonds, which the sparse engine refuses and names rcut"""
    g = geometry.square_lattice().get_supercell(3)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    with pytest.raises(ValueError, match="rcut"):
        Vinteraction_kpm(h, Vr=lambda r1, r2: 0.1, filling=0.5, maxite=0,
                write=False)


@pytest.mark.parametrize("nambu", [False, True])
def test_kpm_band_energy_converges_to_the_exact_one(nambu):
    """The total energy of the sparse engine takes the band energy from the
    KPM density matrix on the hoppings, Tr(H rho), and with Nambu adds the
    trace of the electron block and halves, as spectrum.total_energy does;
    it converges to the sum of the occupied eigenvalues as npol grows"""
    g = geometry.triangular_lattice().get_supercell(3)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.2)
    h.add_exchange([0.3, 0., 0.2])
    h.add_onsite(0.4)
    if nambu: h.add_swave(0.3)
    ed = h.get_total_energy(nk=1)
    err = [abs(get_band_energy_kpm(h, nk=1, npol=npol, T=1e-7) - ed)/abs(ed)
            for npol in [100, 400]]
    assert err[1] < 1e-4 and err[1] < err[0]/10.


def _iteration_peak(L):
    g = geometry.square_lattice().get_supercell(L)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.2)
    h.add_exchange([0.1, 0., 0.])
    h.setup_nambu_spinor()
    tracemalloc.start()
    try:
        h.get_mean_field_hamiltonian_kpm(U=-2.0, filling=0.4, mf="swave",
                npol=10, maxite=0, verbose=0, write=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_one_iteration_takes_memory_linear_in_the_sites():
    """With the working buffers of the recursion and of the trace made
    small, so that they are not what is measured, the memory of a Nambu
    mean-field iteration grows as the number of sites: the interaction,
    the density matrix, the mean field and the Hamiltonian are all sparse.
    An n x n matrix anywhere would grow it sixteen-fold here"""
    with temporary_attr(pairmomentsnumba, "_MAX_BLOCK", 2**16), \
            temporary_attr(pairmomentsnumba, "_MIN_COLUMNS", 8), \
            temporary_attr(pairmomentsjax, "_MAX_BLOCK", 2**16):
        small, large = _iteration_peak(10), _iteration_peak(20)
    assert large/small < 6., large/small


def _vj_model(sparse, nambu):
    g = geometry.honeycomb_lattice().get_supercell(2) if not nambu else \
            geometry.triangular_lattice().get_supercell(2)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
    h.add_rashba(0.2)
    if nambu:
        h.add_exchange([0.1, 0., 0.])
        h.setup_nambu_spinor()
    return h


def _same_guess(nambu, sparse):
    np.random.seed(3)
    mf = meanfield.guess(_vj_model(False, nambu), mode="swave" if nambu
            else "randomXY")
    if not isinstance(mf, dict): mf = {(0, 0, 0): mf}
    return {k: (csr_matrix(m) if sparse else _dense(m)) for (k, m) in mf.items()}


def test_sparse_exchange_mean_field_follows_the_dense_one():
    """h.get_mean_field_hamiltonian(integration="kpm"), VJinteraction, takes
    the sparse engine for a sparse Hamiltonian too: the exchange channels x
    and y are decoupled in a rotated spin frame, which needs the density
    matrix on whole 2x2 spin blocks, and the mean field is the dense KPM
    path's to roundoff"""
    from pyqula.scftk.spinspin import VJinteraction
    kw = dict(U=1.0, J1=0.6, J1z=0.2, filling=0.5, nk=1, integration="kpm",
            npol=60, maxite=6, mix=0.5, T=1e-2)
    dense = VJinteraction(_vj_model(False, False), mf=_same_guess(False, False), **kw)
    sparse = VJinteraction(_vj_model(True, False), mf=_same_guess(False, True), **kw)
    assert issparse(sparse.hamiltonian.intra)
    for key in dense.mf:
        assert np.max(np.abs(_dense(dense.mf[key]) - _dense(sparse.mf[key]))) < 1e-10


def test_sparse_exchange_mean_field_takes_nambu():
    """The dense KPM path of VJinteraction refuses a Nambu Hamiltonian, and
    the sparse one takes it: with only U it is the density-density sparse
    loop to roundoff, and with an exchange J at a fixed chemical potential
    it converges to exact diagonalization as npol grows"""
    from pyqula.scftk.spinspin import VJinteraction
    kw = dict(U=-4.0, nk=1, mix=0.5, T=5e-2)
    a = VJinteraction(_vj_model(True, True), mf=_same_guess(True, True),
            filling=0.4, integration="kpm", npol=60, maxite=6, **kw)
    b = Vinteraction_kpm(_vj_model(True, True), mf=_same_guess(True, True),
            filling=0.4, npol=60, maxite=6, verbose=0, write=False, **kw)
    for key in b.mf:
        assert np.max(np.abs(_dense(a.mf[key]) - _dense(b.mf[key]))) < 1e-10
    kw.update(J1=0.4, mu=-1.0, maxerror=1e-8, maxite=400)
    ed = VJinteraction(_vj_model(False, True), mf=_same_guess(True, False),
            integration="ed", **kw)
    from pyqula.superconductivity import get_eh_sector
    pairing = get_eh_sector(_dense(ed.mf[(0, 0, 0)]), i=0, j=1)
    assert np.max(np.abs(pairing)) > 0.1 # the comparison sees the pairing
    err = []
    for npol in [150, 600]:
        kp = VJinteraction(_vj_model(True, True), mf=_same_guess(True, True),
                integration="kpm", npol=npol, **kw)
        assert kp.converged
        err.append(max(np.max(np.abs(_dense(ed.mf[k]) - _dense(kp.mf[k])))
            for k in ed.mf))
    assert err[1] < 1e-3 and err[1] < err[0]/5.


def test_multihopping_dot_of_sparse_and_dense_matrices():
    """MultiHopping.dot, behind every Hermiticity check and norm, takes the
    inner product entry by entry without making a sparse matrix dense; with
    a sparse matrix on either side, or on both, it is the dense number, and
    a direction present on one side only contributes nothing"""
    from pyqula.multihopping import MultiHopping
    rng = np.random.default_rng(2)
    a = rng.random((6, 6)) + 1j*rng.random((6, 6))
    a[a.real < 0.6] = 0. # a sparse pattern
    b = rng.random((6, 6)) + 1j*rng.random((6, 6))
    ref = np.sum(np.conj(a)*b)
    extra = {(1, 0, 0): rng.random((6, 6))} # on one side only
    for x, y in [(csr_matrix(a), b), (a, csr_matrix(b)),
            (csr_matrix(a), csr_matrix(b)), (np.matrix(a), csr_matrix(b))]:
        out = MultiHopping({(0, 0, 0): x, **extra}).dot(
                MultiHopping({(0, 0, 0): y}))
        assert abs(out - ref) < 1e-12


@pytest.mark.parametrize("vj,nambu", [(False, False), (True, False),
    (True, True)])
def test_random_guess_of_the_sparse_engine(vj, nambu):
    """With no mf given, the sparse engine starts from random phases on the
    pairs of sites the interaction couples, in the Hilbert space of h; the
    guess is Hermitian, so the loop runs, and it is sparse, so the
    Hamiltonian stays sparse"""
    from pyqula.multihopping import MultiHopping
    from pyqula.scftk.spinspin import VJinteraction
    g = geometry.honeycomb_lattice().get_supercell(2)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    h.add_rashba(0.2)
    if nambu: h.setup_nambu_spinor()
    np.random.seed(1)
    if vj:
        scf = VJinteraction(h, U=-2.0 if nambu else 2.0, J1=0.3,
                filling=0.4, nk=1, integration="kpm", npol=40, maxite=2)
    else:
        scf = Vinteraction_kpm(h, U=2.0, V1=0.3, filling=0.4, nk=1,
                npol=40, maxite=2, load_mf=False, write=False, verbose=0)
    assert issparse(scf.hamiltonian.intra)
    assert MultiHopping(scf.hamiltonian.get_dict()).is_hermitian()


def test_the_change_of_a_sparse_mean_field_is_measured_per_stored_entry():
    """The convergence check averages a sparse mean field over the entries
    it holds, so that the same change on every site reads the same at any
    size; averaged over all N^2 entries it shrank as 1/N, and at 10^5
    sites a loop stopped after its first iteration. A dense mean field is
    averaged over every entry, as before"""
    from scipy.sparse import identity
    from pyqula.scftk.densitydensity import diff_mf
    def change(n, sparse):
        a = identity(n, format="csr")*0.1
        b = a.copy()
        b.setdiag(0.2)
        if not sparse: a, b = a.toarray(), b.toarray()
        return diff_mf({(0, 0, 0): a}, {(0, 0, 0): b})
    assert abs(change(10, True) - 0.1) < 1e-12
    assert abs(change(10000, True) - 0.1) < 1e-12
    assert abs(change(10, False) - 0.01) < 1e-12
