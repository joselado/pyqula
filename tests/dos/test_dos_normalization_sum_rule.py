import numpy as np
import pytest

from pyqula import geometry, dos

# The density of states obeys an exact sum rule: its integral over all
# energies is the number of states per unit cell, whatever the algorithm
# used to compute it. That is the oracle used throughout this file,
# rather than a recorded number, because it is the one statement that is
# true of every DOS mode simultaneously and so pins them against each
# other.
#
# dostk.eigtodos.calculate_dos returns sum_i delta/(delta^2+(E-E_i)^2),
# which is pi times a normalized Lorentzian; the caller is the one that
# applies the 1/pi. dos_kmesh (mode="ED") and calculate_dos_hkgen did,
# the adaptive DOS and both energy-window routines did not, so they came
# out exactly pi too large.


def _integral(x, y):
    return np.trapezoid(np.array(y), np.array(x))


def test_ed_and_adaptive_dos_obey_the_sum_rule():
    """A spinless chain has exactly one state per cell."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    energies = np.linspace(-8., 8., 1000)
    (xe, ye) = h.get_dos(mode="ED", energies=energies, nk=400, delta=2e-2,
                         write=False)
    (xa, ya) = h.get_dos(mode="adaptive", energies=energies, delta=2e-2,
                         error=1e-3)
    assert abs(_integral(xe, ye) - 1.0) < 3e-2
    assert abs(_integral(xa, ya) - 1.0) < 3e-2


def test_adaptive_dos_agrees_pointwise_with_exact_diagonalization():
    """Same system, same broadening, two independent integrators: the
    curves must coincide, not differ by a constant factor. Converged on
    both sides (adaptive_dos turns nk into the number of subdivisions the
    quadrature may use, limit=nk//30) they agree to machine precision, so
    this also rules out the pi being hidden in a k-mesh discrepancy."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    energies = np.linspace(-3., 3., 60)
    (_, ye) = h.get_dos(mode="ED", energies=energies, nk=2000, delta=5e-2,
                        write=False)
    (_, ya) = h.get_dos(mode="adaptive", energies=energies, delta=5e-2,
                        error=1e-5, nk=3000)
    ye, ya = np.array(ye), np.array(ya)
    assert np.linalg.norm(ya - ye)/np.linalg.norm(ye) < 1e-6


def test_adaptive_dos_with_an_operator_agrees_with_exact_diagonalization():
    """The operator branch of the adaptive DOS is a separate call to
    calculate_dos and needs the same normalization. sz on a polarized
    chain is a sign-changing curve, so this is not the sum rule but the
    ED path computing the same projected DOS."""
    h = geometry.chain().get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    energies = np.linspace(-3., 3., 60)
    (_, ye) = h.get_dos(mode="ED", energies=energies, nk=2000, delta=5e-2,
                        operator="sz", write=False)
    (_, ya) = h.get_dos(mode="adaptive", energies=energies, delta=5e-2,
                        operator="sz", error=1e-5, nk=3000)
    ye, ya = np.array(ye), np.array(ya)
    assert np.linalg.norm(ya - ye)/np.linalg.norm(ye) < 1e-6


def test_zero_dimensional_adaptive_dos_counts_the_states():
    """In 0d the adaptive DOS is a bare call to the broadening kernel,
    with no integration at all, so it isolates the normalization."""
    g = geometry.chain().supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    energies = np.linspace(-10., 10., 2000)
    (x, y) = h.get_dos(mode="adaptive", energies=energies, delta=2e-2)
    assert abs(_integral(x, y) - 4.0) < 5e-2


def _ewindow_dos(h, tmp_path, **kwargs):
    """dos_ewindow writes its result and returns None."""
    dos.dos_ewindow(h, **kwargs)
    out = np.genfromtxt(tmp_path/"DOS.OUT").T
    return (out[0], out[1])


def test_dos1d_ewindow_obeys_the_sum_rule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=False)
    energies = np.linspace(-8., 8., 600)
    (x, y) = _ewindow_dos(h, tmp_path, energies=energies, delta=2e-2,
                          use_green=False, nk=400)
    assert abs(_integral(x, y) - 1.0) < 3e-2


def test_dos2d_ewindow_obeys_the_sum_rule(tmp_path, monkeypatch):
    """Honeycomb: two orbitals per cell, so two states per cell."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    energies = np.linspace(-8., 8., 400)
    (x, y) = _ewindow_dos(h, tmp_path, energies=energies, delta=5e-2,
                          use_green=False, nk=40)
    assert abs(_integral(x, y) - 2.0) < 6e-2


def test_ewindow_green_branch_runs_and_matches_the_diagonalization_one(
        tmp_path, monkeypatch):
    """dos1d_ewindow declared use_green and then opened its body with
    `if True: # do not use green function`, so the argument was dead,
    while dos2d_ewindow's own Green branch indexed the scalar returned by
    np.trace and raised IndexError. Both now run, and both are the same
    DOS as the diagonalization branch."""
    monkeypatch.chdir(tmp_path)
    energies = np.linspace(-1., 1., 8)
    for h in [geometry.chain().get_hamiltonian(has_spin=False),
              geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)]:
        (_, yg) = _ewindow_dos(h, tmp_path, energies=energies, delta=0.2,
                               use_green=True, nk=60)
        (_, yd) = _ewindow_dos(h, tmp_path, energies=energies, delta=0.2,
                               use_green=False, nk=60)
        yg, yd = np.array(yg), np.array(yd)
        assert np.linalg.norm(yg - yd)/np.linalg.norm(yd) < 0.1
