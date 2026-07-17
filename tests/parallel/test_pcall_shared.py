from pyqula import parallel


def test_pcall_shared_matches_plain_pcall():
    """pcall_shared must return the same results as calling make_f(payload)
    directly, regardless of how many worker processes are used."""
    payload = {"factor": 3, "offset": 1} # stand-in for a large captured object

    def make_f(p):
        def f(x): return x * p["factor"] + p["offset"]
        return f

    xs = list(range(12))
    serial = [make_f(payload)(x) for x in xs]
    try:
        for n in (1, 2, 4):
            parallel.set_cores(n)
            assert parallel.pcall_shared(make_f, payload, xs) == serial
    finally:
        parallel.set_cores(1)


def test_pcall_shared_builds_payload_at_most_once_per_worker():
    """The whole point of pcall_shared over plain pcall: make_f should run
    once per worker per call (to unpack the shared payload), not once per
    task. Verified with a cross-process counter, not just output equality."""
    import multiprocess as mp
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    def make_f(offset):
        with lock:
            counter.value += 1
        def f(x): return x + offset
        return f

    xs = list(range(40)) # many more tasks than workers
    try:
        parallel.set_cores(4)
        out = parallel.pcall_shared(make_f, 10, xs)
    finally:
        parallel.set_cores(1)

    assert out == [x + 10 for x in xs]
    assert counter.value <= 4, f"make_f ran {counter.value} times for 4 workers"


def test_pcall_shared_empty_input():
    parallel.set_cores(1)
    assert parallel.pcall_shared(lambda p: (lambda x: x), None, []) == []


def test_pcall_shared_falls_back_when_shared_memory_unavailable():
    """If /dev/shm is missing or too small (common in small containers),
    pcall_shared must still return correct results instead of raising."""
    from pyqula.paralleltk import shared as shared_mod

    def _raise(*args, **kwargs):
        raise OSError("no space left on device")

    payload = {"factor": 3, "offset": 1}
    def make_f(p):
        def f(x): return x * p["factor"] + p["offset"]
        return f
    xs = list(range(12))
    serial = [make_f(payload)(x) for x in xs]

    original = shared_mod.shared_memory.SharedMemory
    shared_mod.shared_memory.SharedMemory = _raise
    try:
        parallel.set_cores(4)
        assert parallel.pcall_shared(make_f, payload, xs) == serial
    finally:
        shared_mod.shared_memory.SharedMemory = original
        parallel.set_cores(1)
