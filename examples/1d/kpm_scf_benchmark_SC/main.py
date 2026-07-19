# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Benchmark: exact-diagonalization SCF (get_mean_field_hamiltonian) versus
# the KPM/sparse SCF (get_mean_field_hamiltonian_kpm) for a 1d BdG/Nambu
# (superconducting) mean-field Hamiltonian, mirroring
# examples/1d/scf_SC_critical_T (onsite attractive Hubbard chain, singlet
# swave pairing, at half filling -- unlike the dilute-filling 2d swave case
# in kpm_scf_benchmark_SC, this stays well-resolved at a moderate npol
# since the Fermi level sits away from the band edge). As in
# kpm_scf_benchmark_SC, get_dm_kpm tracks exactly which entries are needed
# through the Nambu reordering for BdG Hamiltonians (required_elements_eh
# in kpmtk/densitymatrix_kpm.py) rather than evaluating the whole dense
# block.
import time
import numpy as np
from pyqula import geometry

common = dict(maxerror=1e-6, mix=0.5, maxite=300, verbose=0,
              load_mf=False)


def chain():
    h = geometry.chain().get_hamiltonian()
    h.turn_nambu()
    return h


def gap(nk, npol=None, seed=None, **kwargs):
    if seed is not None: np.random.seed(seed)
    h = chain()
    if npol is None:
        h = h.get_mean_field_hamiltonian(nk=nk, **common, **kwargs)
    else:
        h = h.get_mean_field_hamiltonian_kpm(nk=nk, npol=npol, **common,
                **kwargs)
    return h.get_gap()/2.


# Onsite attractive Hubbard U, half filling, T=0 (as in the T=0 point of
# examples/1d/scf_SC_critical_T's temperature sweep)
t0 = time.time()
g_ed = gap(nk=20, U=-.6, T=0.0, mf="random", seed=3)
t_ed = time.time()-t0

t0 = time.time()
g_kpm = gap(nk=20, npol=400, U=-.6, T=0.0, mf="random", seed=3)
t_kpm = time.time()-t0

print(f"[1d-SC:U] ED  gap/2={g_ed: .6f}  time={t_ed:6.2f}s")
print(f"[1d-SC:U] KPM gap/2={g_kpm: .6f}  time={t_kpm:6.2f}s")
print(f"[1d-SC:U] |delta gap/2|={abs(g_ed-g_kpm):.2e}")
