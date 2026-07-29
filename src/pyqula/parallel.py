# routines to call a function in parallel, across processes
import numba

# numba's default threading layer (tbb, where available) does not survive
# fork(): a @jit(parallel=True) call in the main process initializes tbb's
# thread pool, and forking a new process afterward (paralleltk/multiprocess.py
# uses multiprocess.Pool, which forks on Linux) deadlocks -- the child
# inherits tbb's internal state but not its threads. 'workqueue' is numba's
# own fork-safe threading layer; set it before any parallel=True function
# anywhere in the package can run (this module is imported ahead of them).
numba.config.THREADING_LAYER = 'workqueue'

from .paralleltk import multiprocess as _backend

enabled = True # master switch: set to False to force the whole package to
               # run strictly serially (no process pool, numba/BLAS threads
               # clamped to 1), e.g. for debugging, profiling, or
               # reproducibility. Use set_enabled() to change this after
               # import so the change takes effect immediately.

numba_cores = None # numba threads per process ("None" = numba's own default)
blas_cores = None  # BLAS/LAPACK threads in the *main* process ("None" = leave
                    # as-is); workers are always clamped to 1 regardless, so
                    # a raised main-process value here is only safe to set
                    # when cores==1 (no worker pool competing for the same
                    # physical cores)

def _clamp_blas_threads(n):
    """Best-effort: cap the BLAS/LAPACK thread count via threadpoolctl."""
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(n)
    except ImportError:
        pass

def set_num_threads():
    """Set the number of numba/BLAS threads for the current process."""
    if _backend._is_worker or not enabled: # never oversubscribe / disabled
        numba.set_num_threads(1)
        _clamp_blas_threads(1)
    else: # main process
        if numba_cores is not None: numba.set_num_threads(numba_cores)
        if blas_cores is not None: _clamp_blas_threads(blas_cores)

cores = 1 # number of processes currently in use

def set_cores(n):
    """Set the number of processes used by pcall."""
    global cores
    if not enabled: n = 1 # disabled: never spin up a process pool
    _backend.set_cores(n)
    cores = _backend._num_cores # may fall back to 1 if the pool failed

def set_enabled(flag):
    """Globally enable/disable all parallelism (process pool + numba/BLAS
    threads). Disabling tears down any live process pool and clamps thread
    counts to 1 immediately; re-enabling only lifts the restriction -- call
    set_cores() again to actually spin up a pool."""
    global enabled
    enabled = bool(flag)
    set_cores(cores) # re-applies the n=1 clamp above if just disabled
    set_num_threads()

def pcall(f,xs,**kwargs):
    """Call f on every element of xs, in parallel if cores>1."""
    return _backend.pcall(f,list(xs))
