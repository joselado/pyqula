"""The BSE refuses what it cannot do before it does any work, and does not
return a quietly different answer from the one asked for.

Each case here used to be one of two things: an error raised only after
the expensive part had run (the diagonalization of the mesh, the screened
interaction), or a result that silently differed from the request (the
number of exciton bands, the kernel of a complex interaction).
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain
from pyqula.bsetk.interaction import density_interaction, bare_interaction
from pyqula.bsetk.pairbasis import PairBasis


def test_an_unknown_gauge_is_refused_before_the_mesh_is_diagonalized(
        monkeypatch):
    from pyqula import algebra
    h = gapped_ionic_chain()
    def eigh(*args, **kwargs):
        raise AssertionError("the mesh was diagonalized before the check")
    monkeypatch.setattr(algebra, "eigh", eigh)
    with pytest.raises(ValueError, match="'none', 'phase', 'projection'"):
        PairBasis(h, nk=8, gauge="smooth")


def test_the_iterative_solver_refuses_screening_before_computing_it(
        monkeypatch):
    from pyqula.bsetk import screening
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0, V1=0.5)
    def screened(*args, **kwargs):
        raise AssertionError("the screened interaction was computed")
    monkeypatch.setattr(screening, "screened_interaction", screened)
    with pytest.raises(ValueError, match="solver='iterative'"):
        h.get_bse(V=W, nk=16, tda=True, solver="iterative", screening="rpa")


def test_exciton_bands_return_the_requested_number_with_the_iterative_solver():
    """n is the number of excitons per q-point; with solver='iterative'
    it used to truncate the default neig=4 instead of setting it"""
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0, V1=0.5)
    path = [[0., 0., 0.], [0.25, 0., 0.]]
    qs, es = h.get_exciton_bands(V=W, nk=16, tda=True, solver="iterative",
                                 qpath=path, n=6)
    assert len(es) == 6*len(path)
    qs0, es0 = h.get_exciton_bands(V=W, nk=16, tda=True, qpath=path, n=6)
    assert np.allclose(np.sort(es.real), np.sort(es0.real), atol=1e-8)


def test_a_complex_density_interaction_is_refused():
    """W(-Q) is taken as conj(W(Q)) throughout the kernel, which holds for
    the real W(d) of every density-density interaction and would be
    silently wrong at finite Q for a complex one"""
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0, V1=0.5)
    W[(1, 0, 0)] = W[(1, 0, 0)]*(1. + 0.1j)
    with pytest.raises(ValueError, match="real"):
        bare_interaction(h, V=W)
