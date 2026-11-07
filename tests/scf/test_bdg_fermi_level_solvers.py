"""The Fermi level of a Bogoliubov-de Gennes mean field at a filling under
the quasi-Newton solvers (scftk.superscf.BdGFermiLevel). Broyden mixing
and the scipy solvers take the offset x of the chemical potential as one
more entry of their vector and fit a model of the map from that vector to
the next one, so the step of x has to be a function of the vector alone.
The plain loop keeps its adaptive gain and its give-up, which depend on
the history of the loop."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.scftk import superscf
from pyqula.scftk.densitydensity import Vinteraction
from pyqula.superconductivity import get_eh_sector
from pyqula import algebra

NK = 16


def _h0(zeeman=0.):
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    if zeeman: h.add_zeeman([0., 0., zeeman])
    h.setup_nambu_spinor()
    return h


def _dwave_guess(delta=0.1):
    """A first-neighbor d_{x^2-y^2} pairing of amplitude delta, as a mean
    field (the difference of two Hamiltonians)"""
    h, hp = _h0(), _h0()
    hp.add_pairing(mode="dx2y2", delta=delta)
    d0, dp = h.get_dict(), hp.get_dict()
    return {k: np.asarray(dp[k]-d0[k]) for k in dp}


def _dwave(**kw):
    """The d-wave state from V1=-2 with U=1 at filling 0.4"""
    np.random.seed(1)
    return Vinteraction(_h0(), V1=-2., U=1., filling=0.4, mf=_dwave_guess(),
            nk=NK, solver="broyden_mixing", maxerror=1e-10, load_mf=False,
            write=False, verbose=0, **kw)


def test_the_solver_step_is_a_function_of_the_offset_and_the_density_matrix():
    """The same offset and the same density matrix give the same next
    offset, whatever was evaluated in between. The adaptive step of the
    plain loop does not: its gain halves on a change of sign of the charge
    error, which is what a multisecant model of the map cannot fit"""
    h = _h0()
    h.add_swave(0.2)
    def search(hh, f): return hh.get_fermi4filling(f, nk=8)
    def evaluate(fl, x, adaptive):
        hh = fl.shift(h.copy(), x)
        dm = hh.get_density_matrix(ds=[(0, 0, 0)], nk=8)
        return fl.step(x, dm, adaptive=adaptive)
    # offsets on the two sides of the one that holds the charge, so that
    # the charge error changes sign between them
    x1 = evaluate(superscf.BdGFermiLevel(h, 0.3, search), 0., False)
    assert abs(x1) > 1e-3
    xs = [0., 2*x1, 0., 2*x1, 0.]
    for adaptive in (False, True):
        fl = superscf.BdGFermiLevel(h, 0.3, search)
        out = [evaluate(fl, x, adaptive) for x in xs]
        if adaptive: assert abs(out[0]-out[-1]) > 1e-4 # the gain moved
        else:
            assert out[0] == out[2] == out[4] == x1
            assert out[1] == out[3]


def _adaptive_everywhere(monkeypatch):
    """Route the solvers through the adaptive step of the plain loop, as
    before this change"""
    step = superscf.BdGFermiLevel.step
    monkeypatch.setattr(superscf.BdGFermiLevel, "step",
            lambda self, x, dm, adaptive=True: step(self, x, dm))


@pytest.mark.parametrize("integration", ["ed", "triangle"])
def test_broyden_needs_fewer_iterations_with_the_solver_step(integration,
        monkeypatch):
    """The same d-wave state from the same guess, with the solver step and
    with the adaptive step of the plain loop: the same energy and the same
    charge, in fewer iterations with the step the solver can model, and
    the same count twice in a row"""
    new = [_dwave(integration=integration) for i in range(2)]
    _adaptive_everywhere(monkeypatch)
    old = _dwave(integration=integration)
    for scf in new+[old]: assert scf.converged
    assert new[0].iterations == new[1].iterations
    assert new[0].total_energy == new[1].total_energy
    assert abs(new[0].total_energy-old.total_energy) < 1e-8
    charge = [superscf.get_charge_bdg(s.dm) for s in (new[0], old)]
    assert abs(charge[0]-0.8) < 1e-8 and abs(charge[1]-0.8) < 1e-8
    assert new[0].iterations < old.iterations


def test_a_gapless_state_gives_the_same_result_with_the_solver_step(
        monkeypatch):
    """s-wave with a Zeeman field larger than the gap: the pairing dies,
    the paired charge is a staircase in mu and no offset holds it. The
    solver step ends at the Fermi level of the normal state, as the
    give-up of the adaptive step does"""
    def run():
        np.random.seed(1)
        return Vinteraction(_h0(0.3), U=-2., filling=0.3, mf="swave",
                nk=NK, solver="broyden_mixing", maxerror=1e-10,
                load_mf=False, write=False, verbose=0)
    new = run()
    _adaptive_everywhere(monkeypatch)
    old = run()
    assert new.converged and old.converged
    assert abs(new.total_energy-old.total_energy) < 1e-8
    assert abs(new.charge_error-old.charge_error) < 1e-8
    assert abs(new.hamiltonian.fermi-old.hamiltonian.fermi) < 1e-8


def test_no_gap_warning_after_the_maxite_cut(capsys):
    """A loop stopped at maxite says nothing about the gap of its state,
    so the warning of a charge that is not held is not printed"""
    np.random.seed(1)
    scf = Vinteraction(_h0(), V1=-1.5, filling=0.3, nk=8, maxite=10,
            load_mf=False, write=False, verbose=0)
    assert not scf.converged
    assert "no gap" not in capsys.readouterr().out
    fl = superscf.BdGFermiLevel(_h0(), 0.3, None)
    fl.error = 0.1
    fl.warn(1e-5, converged=False)
    assert capsys.readouterr().out == ""
    fl.warn(1e-5, converged=True)
    assert "no gap" in capsys.readouterr().out


def test_pairing_amplitude_is_the_largest_anomalous_entry():
    """The pairing amplitude read from the Nambu indices directly, against
    the anomalous block of the reordered matrix, on a Hamiltonian with
    hoppings between cells and several sites"""
    h = geometry.honeycomb_lattice().get_supercell(2).get_hamiltonian(
            has_spin=True)
    h.setup_nambu_spinor()
    h.add_pairing(mode="dx2y2", delta=0.3)
    h.add_swave(0.1)
    ref = 0.
    for m in h.get_dict().values():
        a = get_eh_sector(algebra.todense(m), i=0, j=1)
        ref = max(ref, float(np.max(np.abs(a))))
    assert ref > 0.1
    assert abs(superscf.pairing_amplitude(h)-ref) < 1e-14
