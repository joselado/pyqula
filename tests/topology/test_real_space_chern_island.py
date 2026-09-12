"""topologytk.realspace.real_space_chern on a Haldane honeycomb island.

WHY NOT THE SUM OF THE MARKER. real_space_chern returns the diagonal of a
commutator (realspace.py: C = A@B - B@A with A = PXP, B = PYP), so summing
it is taking the trace of a commutator, which is zero for ANY square
matrices. The previous version of this file asserted exactly that -- that
np.sum(c) matched a recorded 1.07e-14 -- which is an algebraic identity and
not a property of the Hamiltonian: it holds for a trivial island and a
topological one alike, at every t2, so it could not have caught a sign
flip, a normalization error, or the marker being zero everywhere.

The discriminating quantity is the marker's value on a BULK site, which the
local Chern marker is constructed to make equal to the Chern number of the
corresponding periodic crystal (Bianco & Resta, PRB 84, 241106(R) (2011)).
That is what is asserted here, against h.get_chern on the periodic Haldane
lattice -- a genuinely independent code path (Fukui-Hatsugai-Suzuki Wilson
loops on a k-mesh) applied to a different Hamiltonian. The finite island
(n=6) reaches about 0.944 rather than 1, the usual few-percent deficit of a
small bulk region, so the band is generous; what it must not do is come out
near zero or with the wrong sign.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import islands
from pyqula import topology


def _bulk_marker(t2, workdir):
    """Mean local Chern marker over the deep bulk of the island: the sites
    within the inner third (in r^2) of the island radius, i.e. away from
    the edge where the marker is not expected to be quantized."""
    import os
    old = os.getcwd()
    os.chdir(workdir) # real_space_chern writes REAL_SPACE_CHERN.OUT
    try:
        g = islands.get_geometry(name="honeycomb", n=6, nedges=4, rot=0.0,
                                 clean=False)
        h = g.get_hamiltonian(has_spin=False)
        if t2 != 0.: h.add_haldane(t2)
        (r, c) = topology.real_space_chern(h)
    finally:
        os.chdir(old)
    d2 = np.array([ri.dot(ri) for ri in np.array(r)])
    core = d2 < np.max(d2)/9.
    assert core.sum() > 10 # the bulk region must not be empty
    return np.mean(np.array(c)[core])


def _periodic_chern(t2):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    return h.get_chern(nk=14)


@pytest.mark.slow
def test_bulk_marker_reproduces_the_periodic_chern_number(tmp_path):
    """Sign and magnitude, against the k-space Chern number of the same
    model on an infinite lattice."""
    marker = _bulk_marker(0.1, tmp_path)
    reference = _periodic_chern(0.1)
    assert np.isclose(reference, 1.0, atol=1e-6) # the model is C=+1
    assert np.isclose(marker, reference, atol=0.1), (marker, reference)


@pytest.mark.slow
def test_bulk_marker_vanishes_without_the_haldane_flux(tmp_path):
    """The same island with t2=0 is a gapless, topologically trivial
    graphene flake: the marker must collapse. This is the check the old
    trace-of-a-commutator assertion could not make -- it gave the identical
    answer for both islands."""
    marker = _bulk_marker(0.0, tmp_path)
    assert np.abs(marker) < 0.05, marker


@pytest.mark.slow
def test_bulk_marker_reverses_with_the_haldane_flux(tmp_path):
    """Reversing the Haldane flux is time reversal, which reverses the
    Chern number; the marker must follow, exactly."""
    plus = _bulk_marker(0.1, tmp_path)
    minus = _bulk_marker(-0.1, tmp_path)
    assert np.isclose(minus, -plus, atol=1e-6), (plus, minus)
