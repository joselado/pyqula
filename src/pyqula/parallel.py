# routines to call a function in parallel, across processes
import numba

from .paralleltk import multiprocess as _backend

numba_cores = None # numba threads per process ("None" = numba's own default)

def set_num_threads():
    """Set the number of numba threads for the current process."""
    if _backend._is_worker: # inside a worker: never oversubscribe
        numba.set_num_threads(1)
    elif numba_cores is not None: # main process
        numba.set_num_threads(numba_cores)

cores = 1 # number of processes currently in use

def set_cores(n):
    """Set the number of processes used by pcall."""
    global cores
    _backend.set_cores(n)
    cores = _backend._num_cores # may fall back to 1 if the pool failed

def pcall(f,xs,**kwargs):
    """Call f on every element of xs, in parallel if cores>1."""
    return _backend.pcall(f,list(xs))
