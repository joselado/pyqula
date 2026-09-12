import numpy as np
import numba

from pyqula import geometry
from pyqula.filling import eigenvalues as filling_eigenvalues
from pyqula.spectrum import total_energy
from pyqula.fermisurface import fermi_surface_generator
from pyqula.htk.eigenvectors import get_eigenvectors, hk_matrix_batch
from pyqula.ldos import ldosmap
from pyqula.kdos import kdos_bands


def max_eigenpair_residual(h, es, vs, ks):
    """max_i || H(k_i) v_i - e_i v_i ||, the oracle for eigenvector
    content (same one as tests/parallel/test_get_eigenvectors_dense.py,
    where it is also shown to reject an arbitrary orthonormal set). It
    is invariant under the phase of each eigenvector, which the
    completeness contraction it replaced was too -- but so was every
    orthonormal set, which is why that one could not fail."""
    hks = hk_matrix_batch(h.get_hk_gen(), ks)  # H(k) for every state's k
    resid = np.einsum("iab,ib->ia", hks, vs) - es[:, None]*vs
    return np.max(np.abs(resid))


def test_eigenvalue_only_functions_independent_of_thread_count():
    """filling.eigenvalues, spectrum.total_energy, and
    fermisurface.fermi_surface_generator all batch their diagonalization
    through htk.eigenvectors.peigvalsh (eigenvalues only); none of their
    results should depend on how many numba threads did the work --
    i.e. serial (1 thread) and parallel (2, 4 threads) execution must
    agree exactly."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    default_threads = numba.get_num_threads()
    try:
        cases = {
            "filling.eigenvalues":
                lambda: np.sort(filling_eigenvalues(h, nk=6)),
            "spectrum.total_energy":
                lambda: total_energy(h, nk=6, mode="mesh"),
            "fermi_surface_generator":
                lambda: fermi_surface_generator(h, energies=[0.0, 0.2], nk=5, delta=0.1)[2],
        }
        for name, f in cases.items():
            results = []
            for n in (1, 2, 4): # serial, then increasingly parallel
                numba.set_num_threads(n)
                results.append(f())
            for r in results[1:]:
                assert np.allclose(r, results[0]), \
                    f"{name}: result depends on numba thread count"
    finally:
        numba.set_num_threads(default_threads)


def test_eigenvector_based_functions_independent_of_thread_count(tmp_path, monkeypatch):
    """htk.eigenvectors.get_eigenvectors, ldos.ldosmap, and
    kdos.kdos_bands all batch full diagonalization (eigenvectors, not
    just eigenvalues) through parallel_diagonalization; same check as
    above, serial vs. parallel execution."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    default_threads = numba.get_num_threads()
    try:
        # get_eigenvectors: the eigenvalues must agree exactly, and the
        # eigenvectors are compared through the eigenpair residual
        # ||H(k)v - e v|| -- individual eigenvector phases aren't
        # guaranteed stable across independently-threaded
        # diagonalizations of the same matrix, and the completeness
        # contraction conj(vs).T@vs that used to be used here is
        # satisfied by any orthonormal set (see
        # tests/parallel/test_get_eigenvectors_dense.py).
        runs = []
        for n in (1, 2, 4):
            numba.set_num_threads(n)
            runs.append(get_eigenvectors(h, nk=5, kpoints=True))
        for (es, vs, ks) in runs[1:]:
            assert np.allclose(es, runs[0][0]), \
                "get_eigenvectors: eigenvalues depend on numba thread count"
            assert np.allclose(ks, runs[0][2]), \
                "get_eigenvectors: kpoints depend on numba thread count"
        for (es, vs, ks) in runs:
            assert max_eigenpair_residual(h, es, vs, ks) < 1e-10, \
                "get_eigenvectors: eigenvectors depend on numba thread count"

        # ldosmap draws its own random kpoints internally; pin them so
        # only the thread count varies between runs (see
        # tests/ldos/test_ldosmap.py for why naive seeding isn't enough).
        energies = np.linspace(-1, 1, 10)
        rng = np.random.RandomState(0)
        ks = [rng.random(3) for _ in range(6)]
        ds_all = []
        for n in (1, 2, 4):
            it = iter(ks)
            monkeypatch.setattr(np.random, "random", lambda *a, **kw: next(it))
            numba.set_num_threads(n)
            _, ds = ldosmap(h, energies=energies, delta=0.1, nk=6)
            ds_all.append(ds)
        for d in ds_all[1:]:
            assert np.allclose(d, ds_all[0]), \
                "ldosmap: result depends on numba thread count"

        # kdos_bands, mode="ED"
        kpath = h.geometry.get_kpath(None, nk=6)
        kenergies = np.linspace(-3, 3, 30)
        out_all = []
        for n in (1, 2, 4):
            numba.set_num_threads(n)
            out = kdos_bands(h, kpath=kpath, energies=kenergies, delta=0.1, mode="ED")
            out_all.append(out[2])
        for o in out_all[1:]:
            assert np.allclose(o, out_all[0]), \
                "kdos_bands: result depends on numba thread count"
    finally:
        numba.set_num_threads(default_threads)
