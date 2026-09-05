import numpy as np

from pyqula import geometry, ldos

ES = np.linspace(-1., 1., 60)
R0 = [2., 0., 0.]


def _chain(has_spin):
    g = geometry.chain().supercell(8)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=has_spin)


def _integral(h, nn):
    (ee, d) = ldos.ldosr_generator(h, es=ES, nn=nn, rs=0.5)(R0)
    return np.trapezoid(d, ee)


def test_spinless_accumulates_over_every_neighbour():
    """The spinless branch assigned `yout = yi*ws[i]` where the spinful and
    Nambu branches accumulate, so the continuum-space LDOS at a point was
    only the last neighbour's contribution. A spinful Hamiltonian on the
    same geometry counts each site twice, so it is an independent
    reference: it must be exactly twice the spinless value."""
    spinless = _integral(_chain(False), nn=4)
    spinful = _integral(_chain(True), nn=4)
    assert abs(spinful - 2. * spinless) < 1e-8


def test_spinless_weights_are_normalized():
    """The weights are normalized to 1, so the weighted sum over the
    neighbours cannot be smaller than the smallest single contribution.
    With the overwrite the nn=4 result was 8.6x too small, and it also
    changed discontinuously with nn."""
    h = _chain(False)
    one = _integral(h, nn=1)
    four = _integral(h, nn=4)
    # a broader neighbourhood re-weights, it does not collapse the result
    assert four > 0.5 * one
