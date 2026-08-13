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
    implementation (see scftk/spinspin.py) has the right
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
    (scftk/spinspin.py's _rotate_dm, used to fold the x/y
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


def test_sxsx_constrains_apply_in_the_lab_frame():
    """`constrains` passed to SxSx/SySy must be enforced against the LAB
    frame's axes, not the internally-rotated frame (where the requested
    axis, not z, is the computational z axis). Regression test: passing
    "no_offplane_magnetism" (removes lab-frame z) to an SxSx run must leave
    the x-ordered moment untouched (z is ~0 there anyway), while
    "no_inplane_magnetism" (removes lab-frame x,y) must suppress it
    entirely, since this system has no mechanism to order along z. Before
    the fix, constrains were enforced against the rotated frame's z axis
    -- i.e. against the physical x axis for SxSx -- so
    "no_offplane_magnetism" would have wiped out the x order instead of
    leaving it alone."""
    g = geometry.chain()
    h1 = g.get_hamiltonian(has_spin=True)
    scf1 = meanfield.SxSx(h1, J1=-2.0, mf="ferroX", nk=10, maxerror=MAXERROR,
            mix=0.3, maxite=300, filling=0.2,
            constrains=["no_offplane_magnetism"])
    assert scf1.converged
    mx1 = np.mean(np.abs(scf1.hamiltonian.get_magnetization()[:, 0]))
    assert mx1 > 0.05, \
        "no_offplane_magnetism should not remove the (lab-frame) x order"

    h2 = g.get_hamiltonian(has_spin=True)
    scf2 = meanfield.SxSx(h2, J1=-2.0, mf="ferroX", nk=10, maxerror=MAXERROR,
            mix=0.3, maxite=300, filling=0.2,
            constrains=["no_inplane_magnetism"])
    assert scf2.converged
    m2 = np.mean(np.abs(scf2.hamiltonian.get_magnetization()), axis=0)
    assert np.max(m2) < 1e-3, \
        f"no_inplane_magnetism should remove all order here: {m2}"


def test_jinteraction_single_axis_converges_from_default_guess():
    """Regression test for a bug where Jinteraction's default (mf=None)
    random guess was seeded only over vz's own bond-direction keys, not the
    full vz/vx/vy union: for a coupling active only on an axis whose own
    vz-equivalent has fewer keys (e.g. only Jx1 nonzero, so vz is reduced
    to just the trivial onsite key), the very first convergence check could
    be blind to the growing x-channel bond mean field, risking a spurious
    early "converged". Checked against the known reference total energy for
    this exact (Jx1-only, filling=0.2) setup."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    scf = meanfield.Jinteraction(h, Jx1=-2.0, nk=10, maxerror=MAXERROR,
            mix=0.3, maxite=300, filling=0.2, mf=None)
    assert scf.converged
    assert np.isclose(scf.total_energy, -0.7043362879187306, atol=1e-4), \
        scf.total_energy


def test_jinteraction_rejects_unrecognized_kwargs():
    """Jinteraction must not silently discard unrecognized keywords (it
    used to accept and drop them via a dead **kwargs catch-all)."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    with pytest.raises(TypeError):
        meanfield.Jinteraction(h, Jz1=-1.0, solver="anderson")


def test_szsz_sxsx_sysy_return_none_instead_of_crashing_when_unsupported():
    """SzSz/SxSx/SySy return the NotImplemented sentinel for spinless or
    BdG Hamiltonians; the get_*_mean_field_hamiltonian wrappers must handle
    that (returning None) rather than crashing on scf.converged."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    assert h.get_szsz_mean_field_hamiltonian(J1=1.0) is None
    assert h.get_sxsx_mean_field_hamiltonian(J1=1.0) is None
    assert h.get_sysy_mean_field_hamiltonian(J1=1.0) is None
    hamiltonian, etot = h.get_szsz_mean_field_hamiltonian(J1=1.0,
            return_total_energy=True)
    assert hamiltonian is None and etot is None


def test_jinteraction_and_vjinteraction_also_return_none_when_unsupported():
    """Jinteraction/VJinteraction must behave the same as their SzSz/SxSx/
    SySy siblings for a spinless Hamiltonian (return the NotImplemented
    sentinel / None through the wrapper), not raise -- regression test for
    an inconsistency where they used to raise ValueError instead."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    assert meanfield.Jinteraction(h, Jz1=1.0) is NotImplemented
    assert meanfield.VJinteraction(h, U=1.0) is NotImplemented
    assert h.get_exchange_mean_field_hamiltonian(Jz1=1.0) is None
    assert h.get_combined_mean_field_hamiltonian(U=1.0) is None


def test_jinteraction_result_has_V_set():
    """The Hamiltonian returned by get_exchange_mean_field_hamiltonian must
    have .V set, like Vinteraction/SzSz/SxSx/SySy's results already do --
    regression test for a gap where Jinteraction/VJinteraction's SCF loop
    never set it, leaving h.V as None inconsistently with the rest of the
    mean-field family."""
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h2 = h.get_exchange_mean_field_hamiltonian(Jz1=-2.0, mf="ferroZ", nk=10,
            maxerror=MAXERROR, mix=0.3, filling=0.2)
    assert h2.V is not None
