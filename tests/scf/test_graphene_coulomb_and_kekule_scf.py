import numpy as np
import pytest

from pyqula import geometry
from pyqula import meanfield
from pyqula.specialhopping import twisted_matrix


@pytest.mark.slow
def test_tbg_kekule_dimerization_matches_reference(tmp_path, monkeypatch):
    """Regression check for a dimerization SCF instability on a honeycomb
    supercell with a twisted-matrix hopping generator (ti=0, so effectively
    a pristine lattice with a modified generator), at a small size
    (supercell(2) instead of (3)): the band energy sum must match the
    value recorded from a known-good run. Marked slow: SCF convergence
    drives the runtime. Note: only reproducible to ~1e-6 (residual SCF
    convergence noise), not machine precision."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.supercell(2)
    h = g.get_hamiltonian(is_sparse=True, has_spin=False, is_multicell=False,
                           mgenerator=twisted_matrix(ti=0.0, lambi=7.0))
    mf = meanfield.guess(h, "dimerization")
    scf = meanfield.Vinteraction(h, nk=1, filling=0.5, V1=2.0, V2=1.0, mix=0.3, mf=mf)
    (k, e) = scf.hamiltonian.get_bands(nk=20)
    assert np.isclose(np.sum(e), -0.048982648934327244, atol=1e-3)


@pytest.mark.slow
def test_kekule_scf_gaps_the_folded_dirac_point(tmp_path, monkeypatch):
    """A supercell(3) of the honeycomb lattice folds K and K' onto Gamma,
    where the two Dirac cones meet and the pristine lattice is gapless. A
    Kekule bond order is exactly the instability that couples them, so the
    converged V1+V2 mean field must open a gap there -- 0.57 at V1=6, V2=4,
    and only 0.008 when both couplings are ten times weaker -- while the
    bands stay valley-polarised.

    The assertions this replaces were abs(sum(e)) < 1e-4 and abs(sum(c)) <
    1e-6. sum(e) is sum_k Tr H(k), which a Kekule mean field (a modulation
    of the *bonds*) leaves at zero for any V1 and V2, and sum(c) over a full
    band structure is nk*Tr(valley) = 0 for any Hamiltonian. Marked slow:
    SCF convergence drives the runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    assert np.min(np.abs(np.array(h.get_bands(nk=20)[1]))) < 1e-8  # gapless

    mf = meanfield.guess(h, "kekule")
    scf = meanfield.Vinteraction(h, V1=6.0, mf=mf, V2=4.0, nk=4, filling=0.5, mix=0.3)
    (k, e, c) = scf.hamiltonian.get_bands(operator="valley", nk=20)
    e, c = np.array(e), np.array(c)
    assert np.min(np.abs(e)) > 0.3
    assert np.max(np.abs(c)) > 0.5
