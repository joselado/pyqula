"""Accuracy characterization of topology.chern(integration="qtci").

The qtci path is accurate for a SMOOTH Berry curvature and unreliable for a
sharply peaked one -- the opposite of what its docstring used to claim, which
recommended it "near a small gap". These tests pin the good regime so a
regression there is caught, and RECORD the bad one so nobody has to
rediscover it.

Deliberately NOT asserted: that the sharp case converges. It does not, and
writing an assertion that passes on the current numbers would pin
anti-convergence as intended behaviour. The sharp-case test below only checks
that the answer still rounds to the right integer -- which is all
test_haldane_chern.py ever checked, and exactly why this went unnoticed.

Measured (spinful Haldane, exact C=2), error in the returned Chern number:

    smooth   (t2=0.3):  nk=10 4.8e-5   nk=20 5.1e-5   nk=40 9.1e-6   converges
    trivial  (C=0):     nk=10 4.8e-7   nk=20 3.1e-8   nk=40 8.4e-9   converges
    sharp    (t2=0.05): nk=10 6.5e-3   nk=20 1.5e-2   nk=40 6.6e-2   WORSE

Isolation runs behind that (not repeated here, they are slow): tolerance is
flat from 1e-4 to 1e-8 on the sharp case, so the TCI rank tolerance is not the
limit; mode="Green", which has no Wilson plaquette and hence no dk, degrades
identically, so the dk=1/(2nk) coupling is not the cause either. What both
modes share is a Gauss-Kronrod order that grows with nk, so refining the grid
shrinks the fraction of nodes near the curvature peak.

The mesh path is not merely more accurate on the sharp case but exactly
quantized (~1e-15 at every nk): Fukui-Hatsugai-Suzuki counts vortices in link
variables instead of quadraturing a field.
"""
import numpy as np
import pytest

from pyqula import geometry, topology
from testutils import temporary_attr


def _haldane(t2, mass=0.0, has_spin=True):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=has_spin)
    h.add_haldane(t2)
    if mass != 0.0:
        h.add_sublattice_imbalance(mass)
    return h


def test_mesh_path_is_exactly_quantized_even_for_a_small_gap():
    """The reference the qtci path is measured against. Fukui-Hatsugai-Suzuki
    is manifestly quantized, so this is exact rather than merely accurate --
    which is why it should be preferred whenever the gap is small."""
    h = _haldane(0.05)  # sharp: small gap, strongly peaked curvature
    with temporary_attr(topology.parallel, "cores", 1):
        for nk in (10, 20):
            c = topology.chern(h, nk=nk)
            assert abs(c - 2.0) < 1e-9, (nk, c)


@pytest.mark.parametrize("t2,mass,exact", [
    (0.3, 0.0, 2),    # smooth curvature, topological
    (0.05, 1.5, 0),   # smooth curvature, trivial
])
def test_qtci_is_accurate_for_smooth_curvature(t2, mass, exact):
    """The regime qtci is genuinely good in: error well below 1e-3, from far
    fewer integrand evaluations than a dense mesh. Regression guard."""
    h = _haldane(t2, mass=mass)
    with temporary_attr(topology.parallel, "cores", 1):
        c = topology.chern(h, integration="qtci", nk=20)
    assert abs(c - exact) < 1e-3, (t2, mass, c, exact)


@pytest.mark.slow
def test_qtci_sharp_gap_only_rounds_correctly_and_does_not_converge():
    """Records the limitation. The qtci error on a small-gap model is ~1e-2
    and does NOT shrink with nk, so only the rounded integer is trustworthy.

    The assertion here is deliberately weak (rounds correctly) because that is
    the honest guarantee. Do not tighten it into a convergence assertion
    without first fixing the underlying quadrature -- and do not "fix" a
    failure here by loosening the smooth-case test above.
    """
    h = _haldane(0.05)
    with temporary_attr(topology.parallel, "cores", 1):
        c10 = topology.chern(h, integration="qtci", nk=10)
        c40 = topology.chern(h, integration="qtci", nk=40)
    assert round(c10) == 2 and round(c40) == 2, (c10, c40)
    # the point of the test: refining nk does not improve matters here
    assert abs(c40 - 2.0) > abs(c10 - 2.0) / 10.0, (
        f"qtci sharp-gap error improved unexpectedly (nk=10 {abs(c10-2):.2e}, "
        f"nk=40 {abs(c40-2):.2e}) -- if the quadrature was fixed, update this "
        f"test and chern_qtci's docstring together")
