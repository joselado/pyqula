import numpy as np

from pyqula import geometry


def test_dos_kpm_2d_matches_ed_independent_of_nk(tmp_path, monkeypatch):
    """Regression check for dos.py's dos_kpm k-mesh normalization on a 2D
    lattice: it used to divide by `nk` instead of `numk = len(ks)`, which
    for a 2D k-mesh (numk = nk**2) left the total DOS too large by a
    factor of nk -- invisible on a 1D lattice (numk == nk there) but wrong
    for 2D/3D. KPM and ED are different algorithms so they aren't expected
    to match tightly, but their ratio must stay close to 1 and, crucially,
    must NOT grow with nk the way the bug did (ratio scaled linearly with
    nk before the fix, e.g. ~13x at nk=10 growing to ~82x at nk=80)."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    energies = np.linspace(-3.0, 3.0, 60)

    ratios = []
    for nk in (10, 30):
        (_, ys_ed) = h.get_dos(mode="ED", energies=energies, nk=nk)
        (_, ys_kpm) = h.get_dos(mode="KPM", energies=energies, nk=nk, delta=0.02)
        ratios.append(np.sum(ys_kpm) / np.sum(ys_ed))

    for ratio in ratios:
        assert np.isclose(ratio, 1.0, atol=0.5)
    # the buggy normalization roughly tripled this ratio between nk=10 and
    # nk=30; the correct one should stay flat within KPM's own discretization
    # noise
    assert ratios[1] / ratios[0] < 1.5
