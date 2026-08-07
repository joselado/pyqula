"""Dense (scipy.linalg.eigh via algebra.eigvalsh) vs sparse (ARPACK
shift-invert via scipy.sparse.linalg.eigsh) diagonalization of the same
Hamiltonian matrix -- the same dispatch bandstructure.get_bands_nd makes
based on `num_bands`/`limits.densedimension` (currently 10000). Unlike the
other cases, neither backend here is numba/jax-jitted, so cold and warm
timings should come out roughly equal; that equality is itself a sanity
check on the harness.

`size` is the Hamiltonian matrix dimension (2 sites per unit cell x n
cells). Reference method for the agreement column is "dense": the sparse
solver targets the NUM_BANDS eigenvalues closest to SIGMA via shift-invert,
compared against the same eigenvalues picked out of the full dense
spectrum.
"""
import numpy as np
import scipy.sparse.linalg as slg
from scipy.sparse import csc_matrix

from pyqula import algebra, geometry

from benchmarks.harness import time_cold_warm

CASE_NAME = "diag_dense_sparse"
SIZES_QUICK = [20, 80, 200]
SIZES_FULL = [20, 80, 200, 500, 1000]

NUM_BANDS = 6
SIGMA = 0.1234  # off-symmetry point: avoids landing exactly on an
                # eigenvalue of this chain Hamiltonian, which would make
                # ARPACK's shift-invert LU factorization exactly singular


def _matrix(n):
    h = geometry.chain(n).get_hamiltonian()
    hk = h.get_hk_gen()([0.0, 0.0, 0.0])
    return np.array(hk)


def run(sizes):
    records = []
    for n in sizes:
        hk = _matrix(n)
        dim = hk.shape[0]
        nb = min(NUM_BANDS, dim - 2)

        t_cold, t_warm, eigs_dense = time_cold_warm(lambda: algebra.eigvalsh(hk))
        eigs_dense = np.sort(eigs_dense)
        ref = np.sort(eigs_dense[np.argsort(np.abs(eigs_dense - SIGMA))][:nb])
        records.append(dict(
            case=CASE_NAME, method="dense", size=dim,
            t_cold=t_cold, t_warm=t_warm, value=float(np.mean(ref)),
            reldiff=0.0, meta=dict(n_cells=n, num_bands=nb),
        ))

        hk_sparse = csc_matrix(hk)

        def call_sparse():
            return slg.eigsh(hk_sparse, k=nb, which="LM", sigma=SIGMA)[0]

        t_cold, t_warm, eigs_sparse = time_cold_warm(call_sparse)
        eigs_sparse = np.sort(eigs_sparse)
        reldiff = float(np.linalg.norm(eigs_sparse - ref) / (np.linalg.norm(ref) + 1e-300))
        records.append(dict(
            case=CASE_NAME, method="sparse", size=dim,
            t_cold=t_cold, t_warm=t_warm, value=float(np.mean(eigs_sparse)),
            reldiff=reldiff, meta=dict(n_cells=n, num_bands=nb),
        ))
    return records
