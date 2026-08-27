"""The Lindhard kernel behind the RPA spin response: numba (CPU) against
jax (`chi_cpugpu="GPU"`, which falls back to jax's CPU backend on a machine
without a device).

This is `chitk/chiAB.py::chiAB_matrix` versus `chitk/chijax.py`'s gathered
GEMM formulation, driven through `chiAB_q` exactly as `get_spinchi_full`
drives it, so the record includes the per-k-point eigendecompositions and
the operator tensors, not just the contraction.

`size` is N, the number of sites in the unit cell. The kernel costs
`36 N^4 nw` complex FMA per k-point, so the sweep is deliberately short at
the top end -- see `future_development/gpu_rpa_spin_response.md` for the
measured N^4 scaling and for what these numbers are and are not evidence
of. In particular: **a jax number taken on a machine with no GPU is a
jax-CPU number**, and says nothing about the device. `machine_info()`
records `jax_devices`; read it before quoting any ratio from this case.

`nk` is kept small on purpose. A 100-site cell has an already-folded
Brillouin zone, so `chiAB`'s `nk=60` default would mean thousands of
k-points that no real calculation at that size does, and the benchmark
would measure the k-loop rather than the kernel.

On a hybrid (performance + efficiency core) CPU, run this pinned --
`taskset -c 0,1 python -m benchmarks.run_all --case rpa_spin_response` --
or the same kernel lands on E cores half the time and the baseline is not
a number any later speedup can be divided by.
"""
import numpy as np

from pyqula import geometry
from pyqula.chitk.chiAB import chiAB_q
from pyqula.chitk.rpa import build_ops_projectors
from pyqula.chitk.spinchi import _full_spin_operators

from benchmarks.harness import time_cold_warm

CASE_NAME = "rpa_spin_response"
SIZES_QUICK = [4, 8, 12]
SIZES_FULL = [4, 8, 12, 16, 24, 32, 48, 64]
# the top of the full sweep is bounded by the *numba* side: at nk=4/nw=40 the
# CPU kernel takes ~10 s at N=32 and ~2.5 min at N=64 (N^4), while the device
# side stays under a second

METHODS = (("numba", "CPU"), ("jax", "GPU"))

NK = 4          # a folded BZ: few k-points, see the module docstring
NW = 40         # frequencies
DELTA = 0.05
Q = [0.2, 0., 0.]


def _system(n):
    """A ferromagnetic chain supercell with n sites in the unit cell.

    An explicit exchange field rather than a self-consistent one: the
    kernel's cost does not depend on the mean field being converged, and an
    SCF at every size would time the solver instead of the response.
    """
    h = geometry.chain().supercell(n).get_hamiltonian()
    h.add_exchange([0., 0., 0.3])
    return h


def _quantity(chis):
    """Im Tr chi summed over the frequency grid -- one scalar that touches
    every frequency and every diagonal element, so the two backends cannot
    agree on it by accident."""
    return float(np.sum([np.trace(c).imag for c in chis]))


def run(sizes):
    records = []
    for n in sizes:
        h = _system(n)
        ops = _full_spin_operators(h)
        pAs, pBs = build_ops_projectors(h, ops)
        pAs, pBs = np.array(pAs), np.array(pBs)
        energies = np.linspace(0.01, 1.0, NW)
        ref = None
        batch = []
        for method, backend in METHODS:
            def call(backend=backend):
                return chiAB_q(h, pAs=pAs, pBs=pBs, q=Q, nk=NK,
                               energies=energies, delta=DELTA,
                               chi_cpugpu=backend)
            t_cold, t_warm, (_, chis) = time_cold_warm(call)
            value = _quantity(chis)
            if method == "numba":
                ref = value
            batch.append(dict(
                case=CASE_NAME, method=method, size=n,
                t_cold=t_cold, t_warm=t_warm, value=value,
                meta=dict(nk=NK, nw=NW, delta=DELTA, q=Q,
                          dim=h.intra.shape[0], nops=len(pAs)),
            ))
        for rec in batch:
            rec["reldiff"] = abs(rec["value"] - ref) / (abs(ref) + 1e-300)
        records.extend(batch)
    return records
