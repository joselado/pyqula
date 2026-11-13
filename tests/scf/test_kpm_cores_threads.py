"""cores= of the KPM routes is the number of numba threads of the Chebyshev
recursion during the call. It used to call parallel.set_cores every
iteration, which re-forked a process pool that received no task and left
the package-wide core count behind (bug_audit_6.md #10)"""
import numba
import numpy as np
from pyqula import geometry, parallel, algebra
from pyqula.kpmtk import densitymatrix_kpm as dmk


def test_cores_is_the_thread_count_of_the_recursion(monkeypatch):
    seen = []
    original = dmk._pair_values_of
    def spy(h, radius):
        f = original(h, radius)
        def g(*args, **kwargs):
            seen.append(numba.get_num_threads()) # threads at the recursion
            return f(*args, **kwargs)
        return g
    monkeypatch.setattr(dmk, "_pair_values_of", spy)
    threads0, cores0 = numba.get_num_threads(), parallel.cores
    n = 1 if threads0 != 1 else min(2, numba.config.NUMBA_NUM_THREADS)
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    v = h.geometry.get_hamiltonian(has_spin=False).get_hopping_dict()
    dm = dmk.get_dm_kpm(h, v, nk=4, npol=100, cores=n)
    fermi = dmk.get_fermi4filling_kpm(h, 0.3, nk=4, npol=100, cores=n)
    assert seen and set(seen) == {n}
    assert numba.get_num_threads() == threads0 # restored after the call
    assert parallel.cores == cores0 # no package-wide change
    seen.clear()
    dm0 = dmk.get_dm_kpm(h, v, nk=4, npol=100)
    fermi0 = dmk.get_fermi4filling_kpm(h, 0.3, nk=4, npol=100)
    assert set(seen) == {threads0}
    assert abs(fermi - fermi0) < 1e-12
    for d in dm0:
        assert np.max(np.abs(algebra.todense(dm[d] - dm0[d]))) < 1e-12
