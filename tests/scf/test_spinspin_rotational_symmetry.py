import numpy as np
import pytest

from pyqula import geometry
from pyqula import meanfield

MAXERROR = 1e-6

_AXIS = {"x": 0, "y": 1, "z": 2}
_FUNCTION = {"x": meanfield.SxSx, "y": meanfield.SySy, "z": meanfield.SzSz}
_GUESS = {"x": "ferroX", "y": "ferroY", "z": "ferroZ"}


def _run(axis, J1=-2.0, filling=0.2, nk=10):
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    scf = _FUNCTION[axis](h, J1=J1, mf=_GUESS[axis], nk=nk,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=filling)
    assert scf.converged, f"SCF for axis {axis} did not converge"
    return scf


def test_szsz_sxsx_sysy_are_rotations_of_each_other():
    """SzSz, SxSx and SySy differ only by which axis the (otherwise
    identical, spin-degenerate) Sa_i Sa_j interaction picks out. By the
    SU(2) symmetry of the bare interaction, the self-consistent total
    energy and the magnitude of the ordered moment must be identical for
    all three axes -- only the direction of the ordering differs. This is
    the key cross-check that the SxSx/SySy rotate-solve-rotate-back
    implementation (see selfconsistency/spinspin.py) has the right
    rotation angle and sign."""
    scfs = {axis: _run(axis) for axis in ("x", "y", "z")}
    etots = np.array([scfs[axis].total_energy for axis in ("x", "y", "z")])
    assert np.max(np.abs(etots - np.mean(etots))) < 1e-5, \
        f"Total energies are not rotationally invariant: {etots}"
    moments = {axis: scfs[axis].hamiltonian.get_magnetization()
               for axis in ("x", "y", "z")}
    magnitudes = []
    for axis in ("x", "y", "z"):
        m = moments[axis]
        ordered_component = np.mean(np.abs(m[:, _AXIS[axis]]))
        other_components = [np.mean(np.abs(m[:, _AXIS[o]]))
                for o in ("x", "y", "z") if o != axis]
        magnitudes.append(ordered_component)
        assert ordered_component > 0.05, \
            f"axis {axis}: no sizable ordered moment along its own axis"
        assert max(other_components) < 1e-3, \
            f"axis {axis}: spurious moment along other axes: {other_components}"
    magnitudes = np.array(magnitudes)
    assert np.max(np.abs(magnitudes - np.mean(magnitudes))) < 1e-3, \
        f"Ordered moment magnitude is not rotationally invariant: {magnitudes}"


def test_jinteraction_single_axis_matches_dedicated_function():
    """Jinteraction with only one of Jx/Jy/Jz nonzero must reproduce the
    corresponding dedicated SxSx/SySy/SzSz result: the combined SCF loop's
    per-channel rotate/decouple/rotate-back logic is exactly the same
    recipe as the single-axis functions, just folded into one loop."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    scf_ref = meanfield.SzSz(h, J1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=0.2)
    scf_combined = meanfield.Jinteraction(h, Jz1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, maxite=300, filling=0.2)
    assert scf_ref.converged and scf_combined.converged
    assert np.isclose(scf_ref.total_energy, scf_combined.total_energy,
            atol=1e-4), \
            (scf_ref.total_energy, scf_combined.total_energy)
    m_ref = scf_ref.hamiltonian.get_magnetization()
    m_combined = scf_combined.hamiltonian.get_magnetization()
    assert np.mean(np.abs(m_ref - m_combined)) < 1e-3


@pytest.mark.parametrize("seed", range(6))
def test_jinteraction_random_direction_guess_gives_collinear_moment(seed):
    """For an isotropic (Jx=Jy=Jz) exchange, the bare interaction has no
    preferred axis, so the direction the system orders along must be set
    entirely by the initial guess: a random-direction guess must converge
    to a moment collinear with it. This is the key regression test for a
    bug where Jinteraction's per-iteration density-matrix rotation
    (selfconsistency/spinspin.py's _rotate_dm, used to fold the x/y
    channels into the lab-frame SCF loop) used the wrong convention and
    silently flipped the sign of the y Pauli component -- invisible to
    energy/magnitude-only checks (a pure-y guess still self-consistently
    finds a same-magnitude, same-energy y-ordered solution even with the
    sign flipped), but not to a random, generically-3-component direction:
    before the fix, every direction collapsed onto the x axis instead."""
    rng = np.random.default_rng(seed)
    v = rng.random(3) - 0.5
    v = v/np.linalg.norm(v)
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    guess = h.copy()
    guess.add_exchange(0.1*v)
    scf = meanfield.Jinteraction(h, Jx1=-2.0, Jy1=-2.0, Jz1=-2.0, mf=guess,
            nk=10, maxerror=MAXERROR, mix=0.2, maxite=300, filling=0.2)
    assert scf.converged
    m = np.mean(scf.hamiltonian.get_magnetization(), axis=0)
    assert np.linalg.norm(m) > 0.05, "no sizable ordered moment developed"
    cos_angle = np.dot(m/np.linalg.norm(m), v)
    assert cos_angle > 1 - 1e-3, \
        f"moment {m} is not collinear with guess direction {v} (cos={cos_angle})"
