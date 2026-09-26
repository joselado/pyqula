"""Every mode of the Fermi surface reports the same weight, and passing the
identity as an operator changes nothing. The operator branches of 'eigen',
'lowest' and fermi_surface_generator used to be 1/pi times the plain one,
and 'full' twice it."""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.fermisurface import fermi_surface_generator


def hamiltonian():
    """Spinful, with no degeneracies left at a generic kpoint"""
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0.1, 0.2, 0.3])
    h.add_onsite(lambda r: 0.4 * (np.sum((r - g.r[0])**2) < 1e-2))
    return h


@pytest.mark.parametrize("mode", ["eigen", "full", "lowest"])
def test_identity_operator_leaves_the_fermi_surface_unchanged(mode, tmp_path,
                                                              monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = hamiltonian()
    kw = dict(nk=3, e=0.2, delta=0.2, mode=mode, write=False,
              k0=np.array([0.137, 0.291]))
    if mode == "lowest": kw["num_waves"] = 8
    iden = np.identity(h.intra.shape[0], dtype=np.complex128)
    d0 = h.get_fermi_surface(operator=None, **kw)[2]
    d1 = h.get_fermi_surface(operator=iden, **kw)[2]
    assert np.allclose(d0, d1, rtol=1e-4)


def test_fermi_surface_modes_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = hamiltonian()
    kw = dict(nk=3, e=0.2, delta=0.2, write=False, operator="sz")
    d_eigen = h.get_fermi_surface(mode="eigen", **kw)[2]
    d_full = h.get_fermi_surface(mode="full", **kw)[2]
    assert np.allclose(d_eigen, d_full)


def test_generator_identity_operator_leaves_the_weight_unchanged(tmp_path,
                                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = hamiltonian()
    iden = np.identity(h.intra.shape[0], dtype=np.complex128)
    kw = dict(energies=[-0.3, 0.2], nk=3, delta=0.2)
    d0 = fermi_surface_generator(h, operator=None, **kw)[2]
    d1 = fermi_surface_generator(h, operator=iden, **kw)[2]
    assert np.allclose(d0, d1)


@pytest.mark.parametrize("operator", [None, "identity"])
def test_lowest_mode_with_as_many_states_as_arpack_cannot_return(
        operator, tmp_path, monkeypatch):
    """num_waves=N-1 falls back to every state, with or without an
    operator, and then it is mode='eigen'"""
    monkeypatch.chdir(tmp_path)
    h = hamiltonian()
    n = h.intra.shape[0]
    op = None if operator is None else np.identity(n, dtype=np.complex128)
    kw = dict(nk=3, e=0.2, delta=0.2, write=False, operator=op)
    d_lowest = h.get_fermi_surface(mode="lowest", num_waves=n - 1, **kw)[2]
    d_eigen = h.get_fermi_surface(mode="eigen", **kw)[2]
    assert np.allclose(d_lowest, d_eigen)


def test_bands_accept_one_less_than_the_dimension(tmp_path, monkeypatch):
    """ARPACK cannot return N-1 eigenpairs of a complex N x N matrix, so
    num_bands=N-1 has to fall back to a full diagonalization as larger
    values do, rather than raise inside scipy"""
    monkeypatch.chdir(tmp_path)
    h = hamiltonian()
    n = h.intra.shape[0]
    out = h.get_bands(kpath=[[0.1, 0.2, 0.]], num_bands=n - 1,
                      operator="sz", write=False)
    assert len(out[1]) == n
