"""kdos.kdos_bands took frand -- the KPM random-vector generator -- and
never forwarded it, so the KDOS was the unprojected one whatever was
passed.

This file used to pin a total of 13043.3144 from a call that passed an
edge-localized frand in the default (ED) mode, and argued that the
number being bit-identical across two unseeded runs showed the quantity
was robust to the random realization. It showed the opposite: no
realization was ever drawn. That value is exactly what the same call
returns with no frand at all, so it never tested the argument. It is
replaced here by checks that the generator is used and that using it
changes the answer in the way a projected KDOS has to.
"""

import numpy as np
import pytest

from pyqula import geometry, kdos


def _ribbon():
    g = geometry.honeycomb_zigzag_ribbon(15)
    h = g.get_hamiltonian()
    h.add_haldane(0.1)  # opens a gap, with states left inside it at the edge
    return h


def _projector(h, sites):
    """A KPM random-vector generator drawing only from the given orbitals"""
    n = h.intra.shape[0]
    w = np.zeros(n) ; w[sites] = 1.0
    return lambda: (-0.5 + np.random.random(n))*w


def test_frand_is_actually_called(tmp_path, monkeypatch):
    """The direct check: count the invocations of the callable that was
    passed. It used to be zero, in both modes."""
    monkeypatch.chdir(tmp_path)
    h = _ribbon()
    calls = []
    f0 = _projector(h, slice(0, 10))
    def frand():
        calls.append(1)
        return f0()
    kdos.kdos_bands(h, use_kpm=True, frand=frand, nk=4, delta=0.3, ntries=4,
                    energies=np.linspace(-1., 1., 11))
    assert len(calls) > 0


def test_frand_is_refused_where_it_cannot_be_honoured(tmp_path, monkeypatch):
    """frand only means something to the KPM branch: mode='ED'
    diagonalizes, it does not sample. Accepting it there and dropping it
    is what let a whole example and this very test believe in a
    projection that was never applied."""
    monkeypatch.chdir(tmp_path)
    h = _ribbon()
    with pytest.raises(ValueError):
        kdos.kdos_bands(h, frand=_projector(h, slice(0, 10)), nk=4,
                        energies=np.linspace(-1., 1., 11))
    kdos.kdos_bands(h, nk=4, energies=np.linspace(-1., 1., 11))  # without it


def test_edge_projection_sees_the_in_gap_states(tmp_path, monkeypatch):
    """What the projection is for. A Haldane-gapped zigzag ribbon carries
    states inside the gap at its edges and none in its interior, so
    sampling the KPM vectors on the edge orbitals must put weight at E=0
    where sampling them in the middle of the ribbon must not. Without
    frand forwarded, both calls returned the same unprojected KDOS."""
    monkeypatch.chdir(tmp_path)
    np.random.seed(1)
    h = _ribbon()
    n = h.intra.shape[0]
    energies = np.linspace(-1., 1., 21)
    def run(sites):
        (k, e, d) = kdos.kdos_bands(h, use_kpm=True, nk=9, delta=0.3,
                        ntries=8, frand=_projector(h, sites),
                        energies=energies)
        return np.mean(d[np.abs(e) < 0.15])  # weight inside the gap
    edge = run(slice(0, 10))
    bulk = run(slice(n//2-5, n//2+5))
    assert edge > 10.*bulk
