from __future__ import print_function
import multiprocess as mp
import sys

# ---------- global state ----------
_pool = None          # the shared process pool
_is_worker = False    # set to True in child processes via initializer
_num_cores = 1        # number of cores currently configured

# ---------- worker initializer ----------
def _init_worker():
    """Called when each worker process starts; sets the _is_worker flag."""
    global _is_worker
    _is_worker = True

# ---------- public API ----------
def set_cores(n=1):
    """Set the number of cores to use.  n='max' uses all available CPUs."""
    global _pool, _num_cores

    # close existing pool if any
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None

    if n == "max":
        n = mp.cpu_count()
    else:
        n = int(n)

    _num_cores = n

    if n > 1:
        try:
            _pool = mp.Pool(processes=n, initializer=_init_worker)
        except Exception as e:
            print(f"Failed to create pool with {n} processes: {e}", file=sys.stderr)
            _pool = None
            _num_cores = 1

def pcall(fun, args):
    """
    Call `fun` on each element of `args` in parallel if a pool exists and
    we are not already inside a worker process.
    """
    global _pool, _is_worker

    # If we are already in a worker, run serially to avoid recursive pooling.
    if _is_worker:
        return [fun(a) for a in args]

    # If we have a working pool, use it.
    if _pool is not None:
        try:
            return _pool.map(fun, args)
        except Exception as e:
            print(f"Parallel execution failed: {e}", file=sys.stderr)
            # fall through to serial

    # Fallback: serial execution
    return [fun(a) for a in args]

# optional: clean up pool when the interpreter exits
import atexit
atexit.register(lambda: _pool.close() if _pool else None)
