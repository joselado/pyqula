# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Benchmark: exact-diagonalization SCF (get_mean_field_hamiltonian) versus
# the KPM/sparse SCF (get_mean_field_hamiltonian_kpm) for BdG/Nambu
# (superconducting) mean-field Hamiltonians, mirroring
# examples/readme_examples/scf_SC (onsite attractive U, singlet swave) and
# examples/readme_examples/scf_SC_triplet (attractive V1 + ferromagnetism,
# unconventional/triplet pairing).
#
# For BdG Hamiltonians (h.has_eh), get_dm_kpm (kpmtk/densitymatrix_kpm.py)
# tracks the required elements through the Nambu reordering
# (required_elements_eh: which entries get_mf's electron/anomalous-sector
# extraction and get_dc_energy actually read, mapped into h.intra's native
# per-site-interleaved index convention -- see sctk/reorder.py), so it
# only evaluates those, not the whole dense block. Note dilute fillings
# (the swave case below uses filling=0.05) put the Fermi level very close
# to the band edge, needing a much larger npol than the non-SC benchmarks
# to resolve the gap accurately -- a standard KPM/Chebyshev resolution
# effect, not specific to has_eh.
import time
import numpy as np
from pyqula import geometry

common = dict(maxerror=1e-4, mix=0.5, maxite=150, verbose=0,
              load_mf=False, return_total_energy=True)


def run_case(label, build_h, nk=8, npol=600, seed=None, **kwargs):
    if seed is not None: np.random.seed(seed)
    h_ed = build_h()
    t0 = time.time()
    hed, eed = h_ed.get_mean_field_hamiltonian(nk=nk, **common, **kwargs)
    t_ed = time.time()-t0

    if seed is not None: np.random.seed(seed)
    h_kpm = build_h()
    t0 = time.time()
    hkpm, ekpm = h_kpm.get_mean_field_hamiltonian_kpm(nk=nk, npol=npol,
            **common, **kwargs)
    t_kpm = time.time()-t0

    print(f"[2d-SC:{label}] ED  gap={hed.get_gap(): .6f}  energy={eed: .6f}  time={t_ed:6.2f}s")
    print(f"[2d-SC:{label}] KPM gap={hkpm.get_gap(): .6f}  energy={ekpm: .6f}  time={t_kpm:6.2f}s")
    print(f"[2d-SC:{label}] |delta gap|={abs(hed.get_gap()-hkpm.get_gap()):.2e}  "
          f"|delta E|={abs(eed-ekpm):.2e}")
    return hed, hkpm


def triangular_swave():
    h = geometry.triangular_lattice().get_hamiltonian()
    h.turn_nambu()
    return h


def triangular_triplet():
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_exchange([0., 0., 1.])
    h.setup_nambu_spinor()
    return h


# Case 1: onsite attractive Hubbard U, conventional (swave) singlet
# pairing, dilute filling -- as in examples/readme_examples/scf_SC
run_case("U-swave", triangular_swave, U=-1.0, filling=0.05, mf="swave",
        npol=600)

# Case 2: attractive first-neighbor V1 with a ferromagnetic exchange field,
# unconventional (odd-parity/triplet) pairing -- as in
# examples/readme_examples/scf_SC_triplet. The d-vector non-unitarity
# (get_dvector_non_unitarity) is a triplet-specific diagnostic beyond the
# gap/energy: it is nonzero only for a genuinely non-unitary triplet
# state, so matching it confirms the KPM path reproduces the unconventional
# character of the state, not just its overall energy scale.
hed_t, hkpm_t = run_case("V1-triplet", triangular_triplet, V1=-1.0,
        filling=0.3, mf="random", npol=400, seed=2)
d_ed = hed_t.get_dvector_non_unitarity()
d_kpm = hkpm_t.get_dvector_non_unitarity()
print(f"[2d-SC:V1-triplet] ED  <|d-vector non-unitarity|>={np.mean(np.abs(d_ed)):.4e}")
print(f"[2d-SC:V1-triplet] KPM <|d-vector non-unitarity|>={np.mean(np.abs(d_kpm)):.4e}")
