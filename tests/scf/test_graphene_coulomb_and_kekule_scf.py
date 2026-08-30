import numpy as np
import pytest

from pyqula import geometry
from pyqula import scftypes
from pyqula import meanfield
from pyqula.specialhopping import twisted_matrix


@pytest.mark.slow
def test_graphene_coulomb_interaction_scf_matches_reference(tmp_path, monkeypatch):
    """Regression check for a ferromagnetic-guess Coulomb (fastCoulomb
    mode) SCF calculation on a triangular-lattice supercell, at a small
    size (supercell(2) instead of (6), nk=4 instead of 10): the
    sz-resolved band energies must match the values recorded from a
    known-good run. Marked slow: SCF convergence and the all-pairs Coulomb
    sum drive the runtime, not just the k-mesh.

    The interaction is now given as `Vr`, the current spelling of a
    distance-dependent density-density interaction. It used to be passed as
    the old scftypes.selfconsistency arguments `mode="fastCoulomb"`,
    `vfun=` and `g=3.0`; none of them exist in the signature that name is
    now aliased to, so all three were dropped and the loop ran with no
    interaction at all -- hence the re-recorded reference. `g` has no
    counterpart here (in the dead code path it was overloaded, naming the
    geometry in one place and a coupling in another), so the interaction is
    `vfun` alone."""
    monkeypatch.chdir(tmp_path)
    g = geometry.triangular_lattice()
    g = g.supercell(2)
    h = g.get_hamiltonian(has_spin=True)
    h = h.get_multicell()
    mf = scftypes.guess(h, mode="ferro", fun=1.0)

    def Vr(r1, r2):
        r = np.linalg.norm(np.array(r1) - np.array(r2))
        if r < 1e-2: return 0.0
        else: return 2.0 * np.exp(-r)

    scf = scftypes.selfconsistency(h, nk=4, filling=0.5,
                    mix=0.9, mf=mf, Vr=Vr)
    (k, e, c) = scf.hamiltonian.get_bands(operator="sz", nk=20)
    assert np.isclose(np.sum(e), 141.0056528200021, atol=1e-4)
    assert np.isclose(np.sum(c), 0.0, atol=1e-6)


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
def test_kekule_honeycomb_scf_matches_reference(tmp_path, monkeypatch):
    """Regression check for a Kekule-guess V1+V2 SCF calculation on a
    honeycomb supercell(3) (the periodicity the Kekule distortion
    requires): the valley-resolved band energies must match the values
    recorded from a known-good run. Marked slow: SCF convergence drives
    the runtime, not the k-mesh."""
    monkeypatch.chdir(tmp_path)
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    mf = meanfield.guess(h, "kekule")
    scf = meanfield.Vinteraction(h, V1=6.0, mf=mf, V2=4.0, nk=4, filling=0.5, mix=0.3)
    (k, e, c) = scf.hamiltonian.get_bands(operator="valley", nk=20)
    assert abs(np.sum(e)) < 1e-4
    assert abs(np.sum(c)) < 1e-6
