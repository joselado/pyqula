import numpy as np

from pyqula import geometry


def test_get_dos_kpm_accepts_frand():
    """Regression check for kpm.pdos: it used to build its own random
    generator without popping a caller-provided `frand` out of kwargs
    first, so h.get_dos(use_kpm=True, frand=...) always crashed with
    "got multiple values for keyword argument 'frand'" (and, since P was
    None in this path, the caller's frand was being silently discarded
    even before the crash was fixed)."""
    g = geometry.chain()
    g = g.supercell(40)
    g.dimensionality = 0
    h = g.get_hamiltonian(is_sparse=True, has_spin=False)

    calls = []
    def frand():
        calls.append(1)
        return (np.random.random(len(h.geometry.r)) - 0.5)

    (x, y) = h.get_dos(use_kpm=True, scale=4.0, frand=frand,
            delta=0.1, energies=np.linspace(-1.0, 1.0, 20), ntries=2)
    assert len(calls) > 0  # the caller-provided generator was actually used
    assert np.all(np.isfinite(y))
