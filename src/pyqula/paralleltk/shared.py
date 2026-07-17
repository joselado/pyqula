# broadcast a large payload to the worker pool once via shared memory,
# instead of re-pickling it on every parallel.pcall dispatch.
#
# parallel.pcall dispatches a function per call; if that function's closure
# captures a large object (e.g. a Hamiltonian's k-dependent generator, which
# closes over every nonzero hopping matrix), multiprocess.Pool.map re-pickles
# it once per dispatched chunk. For large enough Hamiltonians that IPC cost
# dominates the actual per-k diagonalization it was meant to parallelize (see
# densitymatrix.full_dm_accumulate, the case this was built for). pcall_shared
# avoids that by writing the payload into a shared-memory block once, and
# shipping only a small {name, size} handle through the ordinary pcall
# channel; each worker attaches to it and builds its local function once,
# then reuses that for every task it receives in this call.

from multiprocessing import shared_memory
import dill

from . import multiprocess as _backend
from .. import parallel

_worker_cache = {} # shm name -> built f; worker-local, holds at most one entry


def pcall_shared(make_f, payload, xs):
    """
    Equivalent to parallel.pcall(make_f(payload), xs), except `payload`
    is shipped to the worker pool once instead of once per dispatched chunk.

    - `payload` is the large object (e.g. a Hamiltonian's k-Hamiltonian
      generator); it must be picklable with dill.
    - `make_f` builds the actual per-task function from the payload. It is
      called once per worker, so it must itself be small: anything it
      captures besides `payload` still goes through the ordinary pcall
      channel and gets repickled per dispatch as usual.
    """
    if len(xs) == 0: return []
    if parallel.cores == 1 or _backend._is_worker:
        f = make_f(payload)
        return [f(x) for x in xs]

    data = dill.dumps(payload)
    try:
        shm = shared_memory.SharedMemory(create=True, size=max(len(data), 1))
    except OSError:
        # /dev/shm unavailable or too small for this payload (common in
        # containers with a small default shm mount) -- fall back to a
        # plain pcall dispatch rather than failing outright.
        return parallel.pcall(lambda x: make_f(payload)(x), xs)
    try:
        shm.buf[:len(data)] = data
        handle = (shm.name, len(data))
        return parallel.pcall(lambda x: _run(handle, make_f, x), xs)
    finally:
        shm.close()
        shm.unlink()


def _run(handle, make_f, x):
    """Runs inside a worker: attach to the shared payload (once per worker
    per pcall_shared call, cached across every chunk that worker receives)
    and apply the built function to this task's argument."""
    name, size = handle
    if name not in _worker_cache:
        shm = shared_memory.SharedMemory(name=name)
        try:
            payload = dill.loads(bytes(shm.buf[:size]))
        finally:
            shm.close()
        _worker_cache.clear() # only this call's payload is worth keeping
        _worker_cache[name] = make_f(payload)
    return _worker_cache[name](x)
