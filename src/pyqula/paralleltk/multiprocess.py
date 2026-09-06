from __future__ import print_function
import multiprocess as mp
import numpy as np
import sys

# ---------- global state ----------
_pool = None          # the shared process pool
_is_worker = False    # set to True in child processes via initializer
_num_cores = 1        # number of cores currently configured

# ---------- worker initializer ----------
def _init_worker():
    """Called once when each worker process starts."""
    global _is_worker
    _is_worker = True
    from .. import parallel
    parallel.set_num_threads() # never let numba/BLAS oversubscribe inside a worker

# ---------- public API ----------
def set_cores(n=1):
    """Set the number of processes to use.  n='max' uses all available CPUs."""
    global _pool, _num_cores

    if _pool is not None: # close the existing pool, if any
        _pool.close()
        _pool.join()
        _pool = None

    n = mp.cpu_count() if n == "max" else int(n)
    _num_cores = n

    if n > 1:
        try:
            _pool = mp.Pool(processes=n, initializer=_init_worker)
        except Exception as e:
            print(f"Failed to create process pool with {n} processes, "
                  f"falling back to serial execution: {e}", file=sys.stderr)
            _pool = None
            _num_cores = 1

# ---------- per-task random seeds ----------
# Forked workers inherit the parent's numpy random state, so without this
# every worker drew the *same* random numbers: averaging a stochastic
# quantity (KPM's random_trace, say) over k-points did not reduce its
# variance the way it does serially, and the two backends disagreed.
#
# Reseeding each worker from the OS entropy pool would fix that but make
# every parallel result irreproducible run to run. Instead the seeds are
# derived from the parent's own stream, one per *task index* rather than per
# worker: the run is reproducible given the parent's seed, and cores=1 and
# cores>1 produce identical results, since a task's seed does not depend on
# which worker happens to pick it up.

def _task_seeds(n):
    """One seed per task, derived from a single draw off the caller's stream.

    Drawing (rather than reading the state and putting it back) is what keeps
    two pcalls apart: repeated stochastic estimates have to see different
    random numbers or they do not average. The cost is one draw, the same as
    any other routine that uses randomness -- and because the draw is the only
    thing pcall takes, seeding numpy makes the whole sweep reproducible."""
    base = int(np.random.randint(0, 2**32 - 1))
    return [int(s.generate_state(1)[0])
            for s in np.random.SeedSequence(base).spawn(n)]


class _SeededCall(object):
    """Call `fun` on a task with the global numpy RNG seeded for that task."""
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, task):
        (arg, seed) = task
        np.random.seed(seed)
        return self.fun(arg)


def pcall(fun, args):
    """
    Call `fun` on each element of `args`, in parallel over the pool
    configured by `set_cores` if there is one.
    """
    tasks = list(zip(args, _task_seeds(len(args))))
    f = _SeededCall(fun)
    # `_is_worker`: already inside a worker, do not spawn a nested pool
    if _pool is None or _is_worker:
        state = np.random.get_state() # the tasks reseed it, put it back after
        try:
            return [f(t) for t in tasks]
        finally:
            np.random.set_state(state)
    return _pool.map(f, tasks)

# clean up the pool when the interpreter exits
import atexit
atexit.register(lambda: _pool.close() if _pool else None)
