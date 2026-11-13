import numpy as np
import pytest

from pyqula import geometry
from pyqula import densitymatrix as dmm


def _flat_site():
    """One spinful site with H=0: both levels sit exactly at the Fermi
    energy, so the occupation 1/(1+exp(e/delta)) is 0/0 at delta=0."""
    g = geometry.chain()
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    n = h.intra.shape[0]
    pairs = {(0,0,0): (np.arange(n), np.arange(n))}
    return h, pairs


def _routes(h, pairs, delta):
    """The diagonal of the onsite density matrix from every direct route"""
    out = dict()
    out["accumulate"] = np.diag(dmm.full_dm_accumulate(h, nk=1, delta=delta))
    out["simultaneous"] = np.diag(dmm.full_dm_simultaneous(h, nk=1,
            delta=delta))
    dm = dmm.full_dm_accumulate_sparse(h, pairs, nk=1, delta=delta)
    out["sparse"] = np.diag(dm[(0,0,0)])
    dm, fermi = dmm.full_dm_accumulate_sparse_with_fermi(h, pairs, 0.5,
            nk=1, delta=delta)
    out["sparse_with_fermi"] = np.diag(dm[(0,0,0)])
    return out


def test_zero_smearing_is_the_T0_limit_in_every_route():
    """A zero smearing is the T=0 limit, as full_dm reads it: every direct
    route gives the density matrix of a vanishing smearing, and none
    returns a NaN for a level exactly at the Fermi energy."""
    h, pairs = _flat_site()
    zero = _routes(h, pairs, 0.)
    small = _routes(h, pairs, 1e-15)
    ref = np.diag(dmm.full_dm(h, nk=1, T=0.))
    for name in zero:
        assert np.all(np.isfinite(zero[name])), (name, zero[name])
        assert np.allclose(zero[name], small[name], atol=1e-12), name
        assert np.allclose(zero[name], ref, atol=1e-12), name
    # the two degenerate levels share the one electron of half filling
    assert np.allclose(zero["sparse_with_fermi"].real, 0.5, atol=1e-12)
