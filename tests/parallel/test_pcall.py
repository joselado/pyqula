from pyqula import parallel


class _Multiplier:
    """Toy stand-in for a Hamiltonian-like object whose bound methods get
    parallelized over inputs, mirroring the real call-site shape used
    throughout the library (e.g. `lambda x: self.dos(energy=x, **kwargs)`)."""
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x, offset=0):
        return x * self.factor + offset


def _row_of_products(pair):
    """Itself calls pcall, mirroring code such as algebra.eigh's
    block-diagonal split that pcalls inside a function that is already
    being pcalled over k-points."""
    i, xs = pair
    return parallel.pcall(lambda x: x * i, xs)


def test_pcall_matches_serial_for_a_bound_method_closure():
    """pcall's results must not depend on how many processes were used."""
    obj = _Multiplier(3)
    kwargs = {"offset": 1}
    xs = list(range(12))
    fun = lambda x: obj.scale(x, **kwargs)
    serial = [fun(x) for x in xs]
    try:
        for n in (1, 2, 4):
            parallel.set_cores(n)
            assert parallel.pcall(fun, xs) == serial
    finally:
        parallel.set_cores(1)


def test_nested_pcall_runs_serially_inside_a_worker():
    """A pcall'd function that itself calls pcall must not deadlock or
    spawn a nested pool; the inner call should just run serially."""
    rows = [(i, list(range(5))) for i in range(6)]
    expected = [[i * x for x in xs] for i, xs in rows]
    try:
        parallel.set_cores(3)
        out = parallel.pcall(_row_of_products, rows)
    finally:
        parallel.set_cores(1)
    assert out == expected
