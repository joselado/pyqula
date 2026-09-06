import numpy as np

from pyqula.greentk.rg import (green_renormalization_python,
                               green_renormalization_jit,
                               green_renormalization_jit_batch)

# green_renormalization_python documents `nite` (a deliberately truncated
# decimation) and `error` (the convergence threshold) as caller-supplied.
# The numba twins took only (intra,inter,energy,delta,**kwargs) and
# recomputed both from delta, so a caller asking for a truncated result
# got a fully converged one instead: on an O(1) surface Green's function
# the two backends differed by ~1 at nite=2.


def _lead(n=4, seed=0, coupling=0.3):
    rng = np.random.RandomState(seed)
    a = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
    intra = a + np.conjugate(a).T  # Hermitian onsite
    inter = (rng.random_sample((n, n)) +
             1j * rng.random_sample((n, n))) * coupling
    return intra, inter


def _agree(a, b):
    return np.max(np.abs(np.array(a) - np.array(b)))


def test_backends_agree_on_a_truncated_decimation():
    # a strongly coupled lead, so that a few iterations really are far
    # from the converged answer and the comparison below has teeth
    intra, inter = _lead(coupling=1.0)
    energy, delta = 0.3, 1e-2
    for nite in [0, 1, 2]:
        gp = green_renormalization_python(intra, inter, energy=energy,
                                          delta=delta, nite=nite)
        gj = green_renormalization_jit(intra, inter, energy=energy,
                                       delta=delta, nite=nite)
        gb = green_renormalization_jit_batch(intra, inter, np.array([energy]),
                                             delta=delta, nite=nite)
        assert _agree(gp, gj) < 1e-12
        assert _agree(gp, (gb[0][0], gb[1][0])) < 1e-12
    # a truncation must really be unconverged, i.e. differ from the
    # converged answer -- otherwise the check above is vacuous
    conv = green_renormalization_python(intra, inter, energy=energy,
                                        delta=delta)
    trunc = green_renormalization_python(intra, inter, energy=energy,
                                         delta=delta, nite=1)
    assert _agree(conv, trunc) > 1e-2


def test_backends_agree_on_a_loose_convergence_threshold():
    intra, inter = _lead(seed=1)
    energy, delta = 0.1, 1e-2
    for error in [1e-2, 1e-4]:
        gp = green_renormalization_python(intra, inter, energy=energy,
                                          delta=delta, error=error)
        gj = green_renormalization_jit(intra, inter, energy=energy,
                                       delta=delta, error=error)
        assert _agree(gp, gj) < 1e-12


def test_defaults_are_unchanged():
    """Omitting both arguments must give what the backends gave before."""
    intra, inter = _lead(seed=2)
    energy, delta = 0.4, 1e-2
    gp = green_renormalization_python(intra, inter, energy=energy, delta=delta)
    gj = green_renormalization_jit(intra, inter, energy=energy, delta=delta)
    gb = green_renormalization_jit_batch(intra, inter, np.array([energy]),
                                         delta=delta)
    assert _agree(gp, gj) < 1e-10
    assert _agree(gp, (gb[0][0], gb[1][0])) < 1e-10
