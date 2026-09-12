import numpy as np

from pyqula import geometry, meanfield


def _converged_electrons(filling, T, nk=40):
    h = geometry.chain().get_hamiltonian()
    scf = meanfield.Vinteraction(h, U=1.0, filling=filling, T=T, nk=nk,
            mf="ferroZ", mix=0.3, maxerror=1e-7, maxite=400,
            load_mf=False, verbose=0)
    return np.trace(np.array(scf.dm[(0, 0, 0)])).real


def test_converged_electron_count_matches_the_filling_at_finite_T(tmp_path,
                                                                  monkeypatch):
    """`filling` states the electron count the SCF is supposed to run at,
    so the oracle is the argument's own claim: Tr dm[(0,0,0)] == 2*filling
    per cell (2 for the two spin channels), at every temperature.

    The Fermi level used to come from a T=0 sort-and-count while the
    density matrix was then built with Fermi-Dirac at T, so away from a
    particle-hole-symmetric point -- where the asymmetry of the density of
    states about mu does not cancel -- the converged count drifted as T
    grew, silently running the calculation at a filling the user never
    asked for."""
    monkeypatch.chdir(tmp_path)
    for (filling, T) in [(0.1, 1e-7), (0.1, 0.05), (0.3, 0.2)]:
        n = _converged_electrons(filling, T)
        assert abs(n-2*filling) < 1e-6, (filling, T, n)
