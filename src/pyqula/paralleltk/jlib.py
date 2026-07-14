import sys
import atexit

# Attempt to import joblib
try:
    import joblib
    _joblib_available = True
except ImportError:
    _joblib_available = False
    print("joblib not installed. Install with: pip install joblib", file=sys.stderr)

# Global state
_parallel = None          # single Parallel instance
_in_parallel = False      # flag to prevent nested parallelism
_cores = -1               # -1 = all cores (default)

# ---------- core management ----------
def set_cores(n=1):
    """Set the number of parallel jobs.  n='max' uses all cores."""
    global _cores
    if n == "max":
        _cores = -1
    else:
        _cores = int(n)

# ---------- parallel pool lifecycle ----------
def _get_parallel():
    """Return a reusable Parallel instance, creating it if needed."""
    global _parallel
    if _parallel is None and _joblib_available:
        # Use the 'loky' backend – more robust than default multiprocessing.
        _parallel = joblib.Parallel(n_jobs=_cores, backend='loky', verbose=0)
        # Register cleanup at exit.
        atexit.register(_cleanup_parallel)
    return _parallel

def _cleanup_parallel():
    """Forcefully terminate the pool to avoid resource_tracker errors."""
    global _parallel
    if _parallel is not None:
        try:
            # For loky backend, this is the clean way.
            if hasattr(_parallel, '_terminate_backend'):
                _parallel._terminate_backend()
            # Fallback: try to close the underlying pool.
            if hasattr(_parallel, '_pool'):
                _parallel._pool.close()
                _parallel._pool.join()
        except Exception:
            pass
        _parallel = None

# ---------- main parallel call ----------
def pcall(fun, args):
    """Call `fun` on every element of `args` in parallel (or serial if nested)."""
    global _in_parallel

    # If joblib is missing, run serially.
    if not _joblib_available:
        return [fun(a) for a in args]

    # Prevent nested parallelism (workers calling pcall again).
    if _in_parallel:
        return [fun(a) for a in args]

    # Get the pool (creates it if necessary).
    parallel = _get_parallel()
    if parallel is None:
        # Fallback if pool creation failed.
        return [fun(a) for a in args]

    # Execute in parallel.
    _in_parallel = True
    try:
        results = parallel(joblib.delayed(fun)(a) for a in args)
    except Exception as e:
        print(f"Parallel execution failed: {e}", file=sys.stderr)
        results = [fun(a) for a in args]
    finally:
        _in_parallel = False

    return results
