import numpy as np

from pyqula import geometry
from pyqula import topology
from testutils import temporary_attr


def _symmetric_kpath(n=15):
    """k-path that visits every point together with its opposite, so the
    parity of a k-resolved quantity can be read off pair by pair."""
    ks = []
    for t in np.linspace(0.05, 0.6, n):
        ks.append([t, t / 2., 0.])
        ks.append([-t, -t / 2., 0.])
    return np.array(ks)


def _berry(mass, kpath):
    """Plain and valley-projected Berry curvature of a gapped honeycomb."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(mass)
    op = h.get_operator("valley", projector=True)
    with temporary_attr(topology.parallel, "cores", 1):
        (x1, y1) = topology.write_berry(h, kpath=kpath)
        (x, y) = topology.write_berry(h, kpath=kpath, operator=op)
    return np.array(y1), np.array(y)


def test_berry_curvature_is_odd_in_k_and_its_valley_projection_is_even(tmp_path,
                                                                       monkeypatch):
    """A sublattice imbalance gaps graphene without breaking time reversal,
    so the Berry curvature is odd in k -- the two valleys carry opposite
    curvature and the total Chern number is zero. Projecting onto the valley
    operator flips the sign of one valley's contribution, making the
    projected curvature *even* in k and leaving a finite valley Chern
    number whose sign follows the sign of the mass.

    The old assertion was sum(y1) over the k-path, which is zero for any
    time-reversal-symmetric Hamiltonian on a symmetric path -- and in fact
    for any mass, because the curvature cancels pointwise between k and -k.
    Asserting the cancellation pointwise, and the parity of the projected
    curvature, is what that sum was only a shadow of.

    The magnitude bound is analytic too: a massive Dirac cone peaks at
    |Omega| ~ v_F^2/(2 m^2), which is of order unity for v_F = 3t/2 and
    m = 0.6, and would fall by four orders of magnitude for a mass a
    hundred times larger."""
    monkeypatch.chdir(tmp_path)  # write_berry writes BERRY_CURVATURE.OUT to cwd
    mass = 0.6
    kpath = _symmetric_kpath()
    (y1, y) = _berry(mass, kpath)

    # plain curvature: odd in k, hence zero Chern number
    assert np.allclose(y1[0::2], -y1[1::2], atol=1e-6 * np.max(np.abs(y1)))
    # valley-projected curvature: even in k, and finite
    assert np.allclose(y[0::2], y[1::2], atol=1e-6 * np.max(np.abs(y)))
    assert np.max(np.abs(y)) > 1.0
    assert np.sum(y) > 1.0

    # reversing the mass reverses the valley Chern number
    (y1m, ym) = _berry(-mass, kpath)
    assert np.allclose(ym, -y, atol=1e-6 * np.max(np.abs(y)))
