import numpy as np

from pyqula import geometry
from pyqula import meanfield

MAXERROR = 1e-6


def test_vjinteraction_reduces_to_jinteraction_with_only_J():
    """VJinteraction combines density-density (V/U) and spin-spin exchange
    (isotropic J1/J2/J3 plus an optional J1x/J1y/J1z anisotropic correction
    on the first-neighbor shell) mean field into one SCF loop. With
    V1=V2=V3=U=0 and only J1z set (pure Sz-Sz, since J1 defaults to 0), it
    must reduce exactly to Jinteraction's Jz1-only case."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(has_spin=True)
    scf_j = meanfield.Jinteraction(h1, Jz1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=0.2)
    h2 = g.get_hamiltonian(has_spin=True)
    scf_vj = meanfield.VJinteraction(h2, J1z=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=0.2)
    assert scf_j.converged and scf_vj.converged
    assert np.isclose(scf_j.total_energy, scf_vj.total_energy, atol=1e-4), \
        (scf_j.total_energy, scf_vj.total_energy)
    m_j = scf_j.hamiltonian.get_magnetization()
    m_vj = scf_vj.hamiltonian.get_magnetization()
    assert np.mean(np.abs(m_j - m_vj)) < 1e-3


def test_vjinteraction_reduces_to_vinteraction_with_only_V():
    """With J1x=J1y=J1z=0, VJinteraction must reduce exactly to Vinteraction
    (here exercised through its U onsite Hubbard term)."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(has_spin=True)
    scf_v = meanfield.Vinteraction(h1, U=10.0, mf="ferro", nk=10,
            maxerror=MAXERROR, mix=0.3, filling=0.2)
    h2 = g.get_hamiltonian(has_spin=True)
    scf_vj = meanfield.VJinteraction(h2, U=10.0, mf="ferro", nk=10,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=0.2)
    assert scf_v.converged and scf_vj.converged
    assert np.isclose(scf_v.total_energy, scf_vj.total_energy, atol=1e-4), \
        (scf_v.total_energy, scf_vj.total_energy)
    m_v = scf_v.hamiltonian.get_magnetization()
    m_vj = scf_vj.hamiltonian.get_magnetization()
    assert np.mean(np.abs(m_v - m_vj)) < 1e-3


def test_vjinteraction_combined_U_and_J_reinforce_each_other():
    """A sanity check that both channels genuinely contribute when combined:
    U (onsite Hubbard, favors a moment along any direction) and a
    ferromagnetic Jz (favors z order specifically) should reinforce each
    other, giving a larger z moment than either alone at the same
    strength."""
    g = geometry.chain()
    params = dict(mf="ferroZ", nk=10, maxerror=MAXERROR, mix=0.2,
            maxite=400, filling=0.2)

    h_u = g.get_hamiltonian(has_spin=True)
    scf_u = meanfield.VJinteraction(h_u, U=5.0, **params)
    h_j = g.get_hamiltonian(has_spin=True)
    scf_j = meanfield.VJinteraction(h_j, J1z=-1.0, **params)
    h_uj = g.get_hamiltonian(has_spin=True)
    scf_uj = meanfield.VJinteraction(h_uj, U=5.0, J1z=-1.0, **params)
    assert scf_u.converged and scf_j.converged and scf_uj.converged

    mz_u = np.mean(np.abs(scf_u.hamiltonian.get_magnetization()[:, 2]))
    mz_j = np.mean(np.abs(scf_j.hamiltonian.get_magnetization()[:, 2]))
    mz_uj = np.mean(np.abs(scf_uj.hamiltonian.get_magnetization()[:, 2]))
    assert mz_uj > max(mz_u, mz_j), \
        f"combined moment ({mz_uj}) should exceed either channel alone " \
        f"(U-only {mz_u}, Jz-only {mz_j})"


def test_vjinteraction_J1_matches_setting_J1x_J1y_J1z_equal():
    """J1 (isotropic first-neighbor exchange) must be exactly equivalent to
    setting J1x=J1y=J1z to the same value with J1=0, since J1 is defined as
    an isotropic Heisenberg coupling added identically to all three axes on
    the first-neighbor shell."""
    g = geometry.chain()
    params = dict(mf="ferroZ", nk=10, maxerror=MAXERROR, mix=0.3,
            maxite=300, filling=0.2)
    h1 = g.get_hamiltonian(has_spin=True)
    scf_j1 = meanfield.VJinteraction(h1, J1=-2.0, **params)
    h2 = g.get_hamiltonian(has_spin=True)
    scf_xyz = meanfield.VJinteraction(h2, J1x=-2.0, J1y=-2.0, J1z=-2.0, **params)
    assert scf_j1.converged and scf_xyz.converged
    assert np.isclose(scf_j1.total_energy, scf_xyz.total_energy, atol=1e-4), \
        (scf_j1.total_energy, scf_xyz.total_energy)
    m1 = scf_j1.hamiltonian.get_magnetization()
    m2 = scf_xyz.hamiltonian.get_magnetization()
    assert np.mean(np.abs(m1 - m2)) < 1e-3


def test_vjinteraction_isotropic_combination_preserves_su2_symmetry():
    """U (already SU(2)-symmetric) combined with an isotropic J1 exchange
    (J1x=J1y=J1z=0, the pure Heisenberg case) must still have full SU(2)
    symmetry: a random-direction initial guess must converge to a moment
    collinear with it, with no hidden preferred axis introduced by how the
    two channels are summed into one z-channel matrix (see
    VJinteraction/_build_density_v)."""
    g = geometry.chain()
    rng = np.random.default_rng(3)
    for _ in range(4):
        v = rng.random(3) - 0.5
        v = v/np.linalg.norm(v)
        h = g.get_hamiltonian(has_spin=True)
        guess = h.copy()
        guess.add_exchange(0.1*v)
        scf = meanfield.VJinteraction(h, U=1.0, J1=-2.0,
                mf=guess, nk=10, maxerror=MAXERROR, mix=0.2, maxite=300,
                filling=0.2)
        assert scf.converged
        m = np.mean(scf.hamiltonian.get_magnetization(), axis=0)
        assert np.linalg.norm(m) > 0.05
        cos_angle = np.dot(m/np.linalg.norm(m), v)
        assert cos_angle > 1 - 1e-3, \
            f"moment {m} not collinear with guess {v} (cos={cos_angle})"
