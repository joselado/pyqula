import numpy as np

from pyqula import geometry
from pyqula import topology
from testutils import temporary_attr


def _symmetric_kpath(n=15):
    """k-path that visits every point together with its opposite."""
    ks = []
    for t in np.linspace(0.05, 0.6, n):
        ks.append([t, t / 2., 0.])
        ks.append([-t, -t / 2., 0.])
    return np.array(ks)


def _berry(mass, kpath):
    """Plain and (valley*sz)-projected Berry curvature of an
    antiferromagnetic honeycomb lattice."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(mass)
    op = h.get_operator("valley") * h.get_operator("sz")
    with temporary_attr(topology.parallel, "cores", 1):
        (x1, y1) = topology.write_berry(h, kpath=kpath)
        (x, y) = topology.write_berry(h, kpath=kpath, operator=op)
    return np.array(y1), np.array(y)


def test_antiferromagnet_berry_curvature_cancels_between_the_spin_sectors(
        tmp_path, monkeypatch):
    """An antiferromagnet on the honeycomb lattice is a gapped Dirac system
    whose two spin sectors see opposite masses, so their Berry curvatures
    cancel *pointwise*: the total curvature is zero at every k, not merely
    on average over the path. Projecting onto valley*sz -- which flips the
    sign of one valley and of one spin -- adds the two sectors instead of
    subtracting them, leaving a finite spin-valley curvature whose sign
    follows the sign of the staggered moment.

    The old assertion was sum(y1) over the k-path. That is zero because the
    summand itself is zero everywhere, for any staggered moment and any
    path, so it never saw the antiferromagnet.

    The magnitude bound on the projected curvature is the massive-Dirac
    peak |Omega| ~ v_F^2/(2 m^2): of order unity for v_F = 3t/2 and
    m = 0.6, four orders of magnitude smaller for a moment a hundred times
    larger."""
    monkeypatch.chdir(tmp_path)  # write_berry writes BERRY_CURVATURE.OUT to cwd
    mass = 0.6
    kpath = _symmetric_kpath()
    (y1, y) = _berry(mass, kpath)

    assert np.max(np.abs(y1)) < 1e-8 * np.max(np.abs(y))  # cancels pointwise
    assert np.allclose(y[0::2], y[1::2], atol=1e-6 * np.max(np.abs(y)))  # even
    assert np.max(np.abs(y)) > 1.0
    assert np.sum(y) > 1.0

    # reversing the staggered moment reverses the spin-valley curvature
    (y1m, ym) = _berry(-mass, kpath)
    assert np.max(np.abs(y1m)) < 1e-8 * np.max(np.abs(y))
    assert np.allclose(ym, -y, atol=1e-6 * np.max(np.abs(y)))
