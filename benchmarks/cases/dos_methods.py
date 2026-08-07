"""Flagship case: dos.get_dos_general(mode="ED"|"Green"|"KPM") -- one public
entry point, three genuinely different algorithms for the density of
states of the same Hamiltonian (exact diagonalization on a k-mesh, the
Green's-function trace, and Chebyshev/KPM expansion).

`size` is the k-mesh density (`nk`) passed to each mode. Note this axis
mostly drives cost for ED and KPM; get_dos_general's "Green" mode ignores
`nk` for the no-operator path (bloch_selfenergy manages its own adaptive
k-refinement internally) -- that near-flat Green timing across `nk` is
itself the expected, honest result, not a bug in this case.

Reference method for the agreement column is "ED".
"""
import numpy as np

from pyqula import geometry
from pyqula.dos import get_dos_general

from benchmarks.harness import time_cold_warm

CASE_NAME = "dos_methods"
SIZES_QUICK = [20, 60]
SIZES_FULL = [20, 60, 150, 300]

ENERGIES = np.linspace(-3.0, 3.0, 100)
MODES = ("ED", "Green", "KPM")


def _hamiltonian():
    return geometry.honeycomb_lattice().get_hamiltonian()


def run(sizes):
    h = _hamiltonian()
    records = []
    for nk in sizes:
        values = {}
        batch = []
        for mode in MODES:
            kwargs = dict(energies=ENERGIES, nk=nk)
            if mode == "Green":
                kwargs["delta"] = 0.1

            def call(mode=mode, kwargs=kwargs):
                return get_dos_general(h, mode=mode, **kwargs)

            t_cold, t_warm, (e, ys) = time_cold_warm(call)
            ys = np.array(ys)
            values[mode] = ys
            de = ENERGIES[1] - ENERGIES[0]
            integral = float(np.sum((ys[:-1] + ys[1:]) / 2.0) * de)
            batch.append(dict(
                case=CASE_NAME, method=mode, size=nk,
                t_cold=t_cold, t_warm=t_warm, value=integral,
                meta=dict(n_energies=len(ENERGIES)),
            ))
        ref = values["ED"]
        ref_norm = np.linalg.norm(ref) + 1e-300
        for rec, mode in zip(batch, MODES):
            rec["reldiff"] = float(np.linalg.norm(values[mode] - ref) / ref_norm)
        records.extend(batch)
    return records
