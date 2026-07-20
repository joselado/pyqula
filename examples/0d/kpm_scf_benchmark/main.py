# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Benchmark: exact-diagonalization SCF (get_mean_field_hamiltonian) versus
# the KPM/sparse SCF (get_mean_field_hamiltonian_kpm) on a 0d system (a
# finite honeycomb flake), for both an onsite Hubbard U and a first
# neighbor V1 interaction.
import time
import numpy as np
from pyqula import islands
from pyqula.selfconsistency.densitydensity_kpm import get_dm_kpm
from pyqula.selfconsistency.densitydensity import get_dc_energy, get_mf

common = dict(maxerror=1e-4, mix=0.5, maxite=300, verbose=0,
              load_mf=False, return_total_energy=True)


def crosscheck(hed, npol=200):
    """Independent SCF trajectories that use aggressive mixing can settle
    into distinct (but individually valid) self-consistent solutions, so
    comparing their converged energies alone conflates "is get_dm_kpm
    accurate" with "did the two fixed-point iterations take the same
    path". This isolates the first question: freeze the ED-converged
    Hamiltonian (hed.V holds the interaction dict it was built with) and
    check that KPM's selectively-computed density matrix reproduces the
    same mean field/double-counting energy as ED's at that exact point."""
    v = hed.V
    dm_ed = hed.get_density_matrix(ds=list(v.keys()), nk=1)
    dm_kpm = get_dm_kpm(hed, v, nk=1, npol=npol, scale=None)
    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)
    mfdiff = max(np.max(np.abs(mf_ed[d]-mf_kpm[d])) for d in v)
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_kpm))
    return mfdiff, ediff


def run_case(label, build_h, filling=0.5, **kwargs):
    h_ed = build_h()
    t0 = time.time()
    hed, eed = h_ed.get_mean_field_hamiltonian(filling=filling, **common,
            **kwargs)
    t_ed = time.time()-t0

    h_kpm = build_h()
    t0 = time.time()
    hkpm, ekpm = h_kpm.get_mean_field_hamiltonian_kpm(filling=filling,
            npol=200, **common, **kwargs)
    t_kpm = time.time()-t0

    print(f"[0d:{label}] ED  energy={eed: .6f}  time={t_ed:6.2f}s")
    print(f"[0d:{label}] KPM energy={ekpm: .6f}  time={t_kpm:6.2f}s")
    print(f"[0d:{label}] |delta E| (independent SCF runs) = {abs(eed-ekpm):.2e}")
    mfdiff, ediff = crosscheck(hed)
    print(f"[0d:{label}] cross-check at ED's fixed point: "
          f"|delta mf|={mfdiff:.2e}  |delta E_dc|={ediff:.2e}")
    return hed, hkpm


def flake():
    return islands.get_geometry(name="honeycomb", n=2, nedges=3).get_hamiltonian()


# Case 1: onsite Hubbard U, ferromagnetic guess (as in
# examples/readme_examples/scf_island)
hed_u, hkpm_u = run_case("U", flake, U=1.5, mf="ferro")
print("[0d:U] ED  <|m_z|> =", np.mean(np.abs(hed_u.get_magnetization()[:,2])))
print("[0d:U] KPM <|m_z|> =", np.mean(np.abs(hkpm_u.get_magnetization()[:,2])))


# Case 2: first-neighbor V1 interaction, quarter filling, seeded with a
# charge-density-wave (sublattice imbalance) initial guess so that the SCF
# breaks the symmetry towards a nontrivial CDW order
def flake_v1_guess():
    h = flake()
    h0 = h.copy()
    h0.add_sublattice_imbalance(0.2)
    return h0


h0 = flake_v1_guess()
hed_v, hkpm_v = run_case("V1", flake, V1=1.0, mf=h0, filling=0.25)
print("[0d:V1] ED  charge disproportion  =", np.std(hed_v.get_vev()))
print("[0d:V1] KPM charge disproportion =", np.std(hkpm_v.get_vev()))

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(hed_u.get_magnetization()[:,2],marker="o",label="ED")
plt.plot(hkpm_u.get_magnetization()[:,2],marker="s",label="KPM")
plt.xlabel("Site index")
plt.ylabel("$m_z$")
plt.legend()
plt.subplot(1,2,2)
plt.plot(hed_v.get_vev(),marker="o",label="ED")
plt.plot(hkpm_v.get_vev(),marker="s",label="KPM")
plt.xlabel("Site index")
plt.ylabel("Charge")
plt.legend()
plt.show()
