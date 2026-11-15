"""A sparse Hamiltonian runs the sparse mean-field loop, where the
interaction, the density matrix and the mean field are csr matrices. A
dense starting mean field, from an MF.pkl that a dense run left in the
directory or from an mf= dictionary, used to stay dense: the mixing of a
dense and a sparse matrix is dense, so the loop computed on dense matrices
from then on, and the convergence check averaged the change over all n^2
entries instead of the stored ones, which weakened maxerror by about
nnz/n^2. The starting mean field is now made csr, so the loop is the one
it runs from the same guess given as csr."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from pyqula import geometry, inout, meanfield
from pyqula.scftk import densitydensity
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm


def _island(sparse):
    g = geometry.square_lattice().get_supercell(3)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=sparse)
    h.add_rashba(0.3)
    return h


def _dense_guess():
    np.random.seed(5)
    mf = meanfield.guess(_island(False), mode="ferro")
    if not isinstance(mf, dict): mf = {(0, 0, 0): mf}
    return {d: np.array(m.toarray() if issparse(m) else m,
        dtype=np.complex128) for (d, m) in mf.items()}


def _spy(monkeypatch):
    """Record, at every convergence check, whether the two mean fields it
    compares are both sparse"""
    calls = []
    diff_mf = densitydensity.diff_mf
    def spy(a, b):
        calls.append(all(issparse(m) for m in list(a.values())
            + list(b.values())))
        return diff_mf(a, b)
    monkeypatch.setattr(densitydensity, "diff_mf", spy)
    return calls


def _kpm(mf, **kw):
    """A few iterations at a fixed chemical potential, so that the residual
    is the change of the mean field alone"""
    return Vinteraction_kpm(_island(True), U=2.0, mu=0.0, nk=1, npol=60,
            mix=0.5, T=1e-2, maxite=3, verbose=0, write=False, mf=mf, **kw)


@pytest.mark.parametrize("source", ["mf", "MF.pkl"])
def test_a_dense_start_keeps_the_density_density_loop_sparse(source,
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    dense = _dense_guess()
    reference = _kpm({d: csr_matrix(m) for (d, m) in dense.items()})
    calls = _spy(monkeypatch)
    if source == "MF.pkl":
        inout.save(dense, "MF.pkl") # as a dense run leaves it
        scf = _kpm(None, load_mf=True)
    else: scf = _kpm(dense)
    assert len(calls) > 0 and all(calls)
    assert issparse(scf.hamiltonian.intra)
    assert abs(scf.residual - reference.residual) < 1e-10*reference.residual


def test_a_dense_start_keeps_the_exchange_loop_sparse(monkeypatch, tmp_path):
    """The same for the sparse engine of VJinteraction, which made the
    Hamiltonian of every iteration sparse again but kept a dense starting
    mean field dense"""
    from pyqula.scftk.spinspin import VJinteraction
    monkeypatch.chdir(tmp_path)
    dense = _dense_guess()
    def vj(mf):
        return VJinteraction(_island(True), U=2.0, J1=0.3, mu=0.0, nk=1,
                integration="kpm", npol=60, mix=0.5, T=1e-2, maxite=3,
                mf=mf)
    reference = vj({d: csr_matrix(m) for (d, m) in dense.items()})
    calls = _spy(monkeypatch)
    scf = vj(dense)
    assert len(calls) > 0 and all(calls)
    assert abs(scf.residual - reference.residual) < 1e-10*reference.residual
