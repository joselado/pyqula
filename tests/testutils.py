import contextlib

import numpy as np

SCF_MAXERROR = 1e-8  # convergence threshold shared by the SCF invariance tests


@contextlib.contextmanager
def temporary_attr(module, name, value):
    """Temporarily set `module.name = value`, restoring the original value
    afterward even if the wrapped block raises."""
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def random_hermitian_hamiltonian(geometry_factory, supercell=None):
    """Build a Hamiltonian from `geometry_factory()` with a random Hermitian
    intra-cell block, as used by the mode-consistency tests."""
    g = geometry_factory()
    if supercell is not None:
        g = g.get_supercell(supercell)
    h = g.get_hamiltonian()
    m = np.random.random(h.intra.shape) + 1j * np.random.random(h.intra.shape)
    h.intra = m + np.conjugate(m).T
    return h


def assert_all_consistent(outs, tol, label):
    """Assert every array in `outs` agrees with their mean to within `tol`."""
    outs = np.array(outs)
    mout = np.mean(outs, axis=0)
    diffs = [np.sum(np.abs(o - mout)) for o in outs]
    assert np.max(diffs) < tol, f"{label} disagree: {diffs}"
