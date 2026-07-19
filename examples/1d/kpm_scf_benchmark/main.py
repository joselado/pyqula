# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Benchmark: exact-diagonalization SCF (get_mean_field_hamiltonian) versus
# the KPM/sparse SCF (get_mean_field_hamiltonian_kpm) on a 1d system (a
# honeycomb zigzag ribbon, as in examples/1d/zigzag_ribbon_SCF), for both
# an onsite Hubbard U and a first neighbor V1 interaction. nk plays the
# same role in both methods (k-point-sampling density); the KPM path
# samples its own k-mesh at nk_kpm (deliberately coarser than ED's nk_ed
# below, since KPM only needs enough k-points to resolve the density
# matrix to within the KPM/Chebyshev accuracy set by npol -- see the
# cross-check numbers printed below) and gets each needed density-matrix
# element via Chebyshev recursion on the small Bloch Hamiltonian H(k)
# instead of diagonalizing it -- see kpmtk.densitymatrix_kpm.
import time
import numpy as np
from pyqula import geometry
from pyqula.selfconsistency.densitydensity_kpm import get_dm_kpm
from pyqula.selfconsistency.densitydensity import get_dc_energy, get_mf

common = dict(maxerror=1e-4, mix=0.5, maxite=200, verbose=0,
              load_mf=False, return_total_energy=True)


def crosscheck(hed, nk_kpm, npol=150):
    """Independent SCF trajectories that use aggressive mixing can settle
    into distinct (but individually valid) self-consistent solutions, so
    comparing their converged energies alone conflates "is get_dm_kpm
    accurate" with "did the two fixed-point iterations take the same
    path". This isolates the first question: freeze the ED-converged
    Hamiltonian (hed.V holds the interaction dict it was built with) and
    check that KPM's selectively-computed density matrix reproduces the
    same mean field/double-counting energy as ED's at that exact point,
    sampled at the same k-point density (hed.nk) ED itself converged at."""
    v = hed.V
    dm_ed = hed.get_density_matrix(ds=list(v.keys()), nk=hed.nk)
    dm_kpm = get_dm_kpm(hed, v, nk=nk_kpm, npol=npol, scale=None)
    mf_ed = get_mf(v, dm_ed)
    mf_kpm = get_mf(v, dm_kpm)
    mfdiff = max(np.max(np.abs(mf_ed[d]-mf_kpm[d])) for d in v)
    ediff = abs(get_dc_energy(v, dm_ed) - get_dc_energy(v, dm_kpm))
    return mfdiff, ediff


def run_case(label, build_h, nk_ed=60, nk_kpm=16, filling=0.5, **kwargs):
    h_ed = build_h()
    t0 = time.time()
    hed, eed = h_ed.get_mean_field_hamiltonian(nk=nk_ed, filling=filling,
            **common, **kwargs)
    t_ed = time.time()-t0

    h_kpm = build_h()
    t0 = time.time()
    hkpm, ekpm = h_kpm.get_mean_field_hamiltonian_kpm(nk=nk_kpm, npol=150,
            filling=filling, **common, **kwargs)
    t_kpm = time.time()-t0

    print(f"[1d:{label}] ED  (nk={nk_ed:3d}) energy={eed: .6f}  time={t_ed:6.2f}s")
    print(f"[1d:{label}] KPM (nk={nk_kpm:3d}) energy={ekpm: .6f}  time={t_kpm:6.2f}s")
    print(f"[1d:{label}] |delta E| (independent SCF runs) = {abs(eed-ekpm):.2e}")
    mfdiff, ediff = crosscheck(hed, nk_kpm)
    print(f"[1d:{label}] cross-check at ED's fixed point: "
          f"|delta mf|={mfdiff:.2e}  |delta E_dc|={ediff:.2e}")
    return hed, hkpm


def ribbon():
    return geometry.honeycomb_zigzag_ribbon(3).get_hamiltonian()


# Case 1: onsite Hubbard U, antiferromagnetic guess
hed_u, hkpm_u = run_case("U", ribbon, U=1.5, mf="antiferro")
print("[1d:U] ED  <|m_z|> =", np.mean(np.abs(hed_u.get_magnetization()[:,2])))
print("[1d:U] KPM <|m_z|> =", np.mean(np.abs(hkpm_u.get_magnetization()[:,2])))


# Case 2: first-neighbor V1 interaction at quarter filling, seeded with a
# charge-density-wave (sublattice imbalance) initial guess
def ribbon_v1_guess():
    h = ribbon()
    h0 = h.copy()
    h0.add_sublattice_imbalance(0.2)
    return h0


h0 = ribbon_v1_guess()
hed_v, hkpm_v = run_case("V1", ribbon, V1=1.0, mf=h0, filling=0.25)
print("[1d:V1] ED  charge disproportion  =", np.std(hed_v.get_vev()))
print("[1d:V1] KPM charge disproportion =", np.std(hkpm_v.get_vev()))
