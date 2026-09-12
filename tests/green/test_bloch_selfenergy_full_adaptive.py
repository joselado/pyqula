import numpy as np

from pyqula import geometry, algebra
from pyqula.greentk.selfenergy import bloch_selfenergy

# mode="full_adaptive" approximates the bulk Green's function of a periodic
# Hamiltonian by adaptively integrating inv(E+i*delta-H(k)) over the
# Brillouin zone. Its definition is therefore its own oracle: the same
# integral evaluated on a dense uniform mesh. No reference number is
# pinned; what is asserted is that the caller's `error` actually controls
# the answer and that the answer converges to the definition as `error` is
# tightened. The 2D branch used to pass a hardcoded eps=0.1 to the
# integrator, so the returned Green's function was bit-identical for every
# `error` and ~50% off on some entries.

ENERGY, DELTA = 0.3, 0.1


def brute_force_bulk_green(h, nk=400):
    """Uniform-mesh Brillouin-zone average of inv(E+i*delta-H(k)), i.e.
    the quantity full_adaptive is an adaptive quadrature of."""
    hk = h.get_hk_gen()
    n = h.intra.shape[0]
    e = np.identity(n)*(ENERGY + 1j*DELTA)
    ks = np.linspace(0., 1., nk, endpoint=False)
    acc = np.zeros((n, n), dtype=np.complex128)
    for kx in ks:
        for ky in ks:
            acc = acc + algebra.inv(e - hk([kx, ky]))
    return acc/nk**2


def test_full_adaptive_2d_honours_error():
    """Tightening `error` must both change the answer and move it closer
    to the Brillouin-zone integral it approximates."""
    h = geometry.square_lattice().get_hamiltonian()
    ref = brute_force_bulk_green(h)
    gs = {}
    for error in [1e-2, 1e-4]:
        g, _ = bloch_selfenergy(h, energy=ENERGY, delta=DELTA,
                                mode="full_adaptive", error=error)
        gs[error] = np.array(g)
    loose = np.max(np.abs(gs[1e-2]-ref))
    tight = np.max(np.abs(gs[1e-4]-ref))
    # the argument is live at all
    assert np.max(np.abs(gs[1e-2]-gs[1e-4])) > 0.
    # and tightening it helps by more than a rounding error
    assert tight < loose/10.
    # the tight answer really is the BZ integral
    assert tight < 1e-3*max(1., np.max(np.abs(ref)))


def test_full_adaptive_1d_honours_error():
    """The 1D branch of the same mode already threaded `error` through;
    pin that, so the two branches cannot drift apart again."""
    h = geometry.chain().get_hamiltonian()
    hk = h.get_hk_gen()
    n = h.intra.shape[0]
    e = np.identity(n)*(ENERGY + 1j*DELTA)
    nk = 4000
    ref = sum(algebra.inv(e - hk([k, 0., 0.]))
              for k in np.linspace(0., 1., nk, endpoint=False))/nk
    g, _ = bloch_selfenergy(h, energy=ENERGY, delta=DELTA,
                            mode="full_adaptive", error=1e-6)
    assert np.max(np.abs(np.array(g)-ref)) < 1e-4
