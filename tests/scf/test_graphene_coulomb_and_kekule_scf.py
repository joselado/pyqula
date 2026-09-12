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
    e, c = np.array(e), np.array(c)
    assert np.isclose(np.sum(e), 141.0056528200021, atol=1e-4)

    # The ferromagnetic guess survives the loop, and the state it converges
    # to is collinear: sz stays a good quantum number, so every band is a
    # pure spin state, every site carries the same moment along z, and the
    # two spin species are pushed apart by an exchange splitting that is
    # linear in the interaction (0.0078 at this Vr, 7.8e-5 at a hundredth
    # of it).
    #
    # The assertion this replaces was sum(c) == 0, which is nk*Tr(sz) over
    # a full band structure: zero for every Hamiltonian with a spin index,
    # interacting or not, magnetic or not.
    assert np.allclose(np.abs(c), 1., atol=1e-6)
    mag = np.array(scf.hamiltonian.get_magnetization())
    assert np.allclose(mag[:, :2], 0., atol=1e-6)  # collinear, along z
    assert np.all(np.abs(mag[:, 2]) > 0.02)  # and ferromagnetic
    assert np.allclose(mag[:, 2], mag[0, 2], atol=1e-3)  # the same on every site
    up, dn = np.sort(e[c > 0.5]), np.sort(e[c < -0.5])
    assert len(up) == len(dn)
    assert abs(np.mean(dn - up)) > 0.005


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
