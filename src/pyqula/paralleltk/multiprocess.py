from __future__ import print_function
import multiprocess as mp
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
    parallel.set_num_threads() # never let numba oversubscribe inside a worker

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

def pcall(fun, args):
    """
    Call `fun` on each element of `args`, in parallel over the pool
    configured by `set_cores` if there is one.
    """
    if _is_worker: # already inside a worker, do not spawn a nested pool
        return [fun(a) for a in args]
    if _pool is not None:
        return _pool.map(fun, args)
    return [fun(a) for a in args]

# clean up the pool when the interpreter exits
import atexit
atexit.register(lambda: _pool.close() if _pool else None)
