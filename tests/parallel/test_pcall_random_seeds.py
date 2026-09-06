"""Forked pool workers inherit the parent's numpy random state, so every
worker used to draw the *same* random numbers: a stochastic quantity
averaged through pcall did not average independently when cores>1, and the
serial and parallel backends disagreed."""
import numpy as np

from pyqula import parallel


def _draw(i):
    return float(np.random.random())


def test_every_task_draws_its_own_random_numbers():
    """cores=4 used to give 4 distinct values out of 8, one per worker."""
    try:
        for n in (1, 2, 4):
            parallel.set_cores(n)
            out = parallel.pcall(_draw, range(8))
            assert len(set(out)) == 8, (n, out)
    finally:
        parallel.set_cores(1)


def test_the_backends_agree_and_the_run_is_reproducible():
    """The seeds are derived per task index, not per worker, so the answer
    does not depend on how tasks are distributed over the pool -- and given
    the parent's seed it is the same answer every run."""
    ref = None
    try:
        for n in (1, 2, 4):
            parallel.set_cores(n)
            for _ in range(2):
                np.random.seed(42)
                out = parallel.pcall(_draw, range(8))
                if ref is None: ref = out
                assert out == ref, (n, out, ref)
    finally:
        parallel.set_cores(1)


def test_pcall_costs_the_parent_exactly_one_draw():
    """The tasks reseed the global RNG; the parent's own stream must come
    back where a single draw would have left it, whatever the tasks did."""
    np.random.seed(7)
    parallel.pcall(_draw, range(5))
    after = np.random.random()
    np.random.seed(7)
    np.random.randint(0, 2**32 - 1)  # the one draw pcall makes
    assert after == np.random.random()


def _random_matrix():
    import scipy.sparse as sp
    rs = np.random.RandomState(1)  # independent of the stream under test
    m = sp.random(200, 200, density=0.02, format="csc", random_state=rs)
    return (m + m.getH())/20.


_M = _random_matrix()


def _kpm_moment(i):
    """A stochastic estimator of a fixed number: KPM's random_trace draws
    random vectors, so repeated calls scatter around the true moment."""
    from pyqula import kpm
    return kpm.random_trace(_M, ntries=4, n=20)[3].real


def test_a_stochastic_estimator_averages_independently():
    """KPM's random_trace is the motivating case: eight independent
    estimates of the same moment, spread over the pool."""
    fun = _kpm_moment
    ref = None
    try:
        for n in (1, 4):
            parallel.set_cores(n)
            np.random.seed(3)
            out = np.array(parallel.pcall(fun, range(8)))
            assert len(set(np.round(out, 12))) == 8, (n, out)
            assert np.std(out) > 1e-6  # genuinely stochastic, not a constant
            if ref is None: ref = out
            assert np.allclose(out, ref)
    finally:
        parallel.set_cores(1)
