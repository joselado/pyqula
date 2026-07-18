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
    if _backend._is_worker: # inside a worker: never oversubscribe
        numba.set_num_threads(1)
        _clamp_blas_threads(1)
    else: # main process
        if numba_cores is not None: numba.set_num_threads(numba_cores)
        if blas_cores is not None: _clamp_blas_threads(blas_cores)

cores = 1 # number of processes currently in use

def set_cores(n):
    """Set the number of processes used by pcall."""
    global cores
    _backend.set_cores(n)
    cores = _backend._num_cores # may fall back to 1 if the pool failed

def pcall(f,xs,**kwargs):
    """Call f on every element of xs, in parallel if cores>1."""
    return _backend.pcall(f,list(xs))
