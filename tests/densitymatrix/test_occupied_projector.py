"""densitymatrix.occupied_projector honours its smearing.

It used to accept delta and never pass it on, so every width returned the
same sharp projector. The default call is what topologytk/realspace.py
makes, and it has to stay exactly as it was."""
import numpy as np
import pytest

from pyqula import densitymatrix
from pyqula.dmtk.fulldm import full_dm_python


def _matrix():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    return A + A.conj().T


def _fermi_dirac_projector_transposed(m, occ):
    e, v = np.linalg.eigh(m)
    return ((v*occ(e)) @ v.conj().T).T


def test_default_is_unchanged():
    m = _matrix()
    es, vs = np.linalg.eigh(m)
    ref = full_dm_python(es, np.array(vs.T))
    assert np.max(np.abs(densitymatrix.occupied_projector(m) - ref)) < 1e-12


@pytest.mark.parametrize("delta", [0.5, 5.0])
def test_delta_is_a_fermi_dirac_width(delta):
    m = _matrix()
    ref = _fermi_dirac_projector_transposed(m, lambda e: 1/(1+np.exp(e/delta)))
    P = densitymatrix.occupied_projector(m, delta=delta)
    assert np.max(np.abs(P - ref)) < 1e-12


def test_zero_delta_is_the_sharp_projector():
    m = _matrix()
    ref = _fermi_dirac_projector_transposed(m, lambda e: (e < 0).astype(float))
    P = densitymatrix.occupied_projector(m, delta=0.)
    assert np.max(np.abs(P - ref)) < 1e-12
    assert np.max(np.abs(P.T @ P.T - P.T)) < 1e-12 # P.T is the projector
