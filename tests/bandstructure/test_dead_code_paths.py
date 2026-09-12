"""Four public entry points that could not run at all: two that called
exit() -- which kills the caller's interpreter with no traceback and no
failing status, so a script simply stops -- and two that raised NameError
on an undefined name before reaching anything they meant to do.

These live here rather than in tests/algebra, tests/energetics or a
tests/unfolding of their own because the code they cover is spread over
four modules with no single matching topic directory.
"""

import numpy as np
import pytest

from pyqula import geometry, algebra, specialhamiltonian, unfolding, hamiltonians
from pyqula.energeticstk.alloytk import Alloy


def test_alloy_energy_returns_instead_of_exiting():
    """Alloy.get_energy went through get_energy_i, whose second statement
    was a debug leftover `print(len(r)); exit()`. The energy is a pure
    pair sum over the sites and their 27 periodic images in the -1..1
    range setup_distances uses, so a unit interaction counts them: n*n*27
    for n sites, whatever the lattice is."""
    n = 4
    g = geometry.chain().supercell(n)
    a = Alloy(g)
    assert a.get_energy() == 0.0  # the default interaction is zero
    a.fenergy = lambda d, si, sj: 1.0  # count every pair and image
    a.setup_interaction()  # rewire get_energy_i to the default path
    assert abs(a.get_energy() - n*n*27) < 1e-8


def test_spectral_gap_enlarges_the_window():
    """algebra.spectral_gap's fallback for 'no state of one sign among the
    computed ones' called gap(), a name that does not exist, so it raised
    NameError exactly when it was needed; and it asked smalleig for a
    hardcoded 10 eigenvalues instead of numw, so enlarging the window
    could not have helped even with the name right. Oracle: the gap of a
    matrix whose spectrum is known by construction."""
    np.random.seed(3)
    n = 30
    d = np.concatenate([np.linspace(-100., -80., 18), np.linspace(1., 12., 12)])
    q, _ = np.linalg.qr(np.random.random((n, n)) + 1j*np.random.random((n, n)))
    m = q@np.diag(d)@np.conjugate(q).T
    m = (m + np.conjugate(m).T)/2.  # Hermitian, spectrum d
    # the 10 eigenvalues closest to zero are all positive here, so the
    # first pass finds no valence state and the fallback has to fire
    assert np.min(algebra.smalleig(m, numw=10)) > 0.
    assert abs(algebra.spectral_gap(m) - (80. + 1.)) < 1e-8


def test_unfolded_bands_says_it_is_not_implemented():
    """unfolded_bands is a stub, but line 13 read an unbound name, so it
    raised NameError inside the k-loop instead of the NotImplementedError
    it carries at the bottom."""
    g = geometry.chain()
    hp = g.get_hamiltonian(has_spin=False)
    hf = g.supercell(3).get_hamiltonian(has_spin=False)
    with pytest.raises(NotImplementedError):
        unfolding.unfolded_bands(hf, hp, [np.array([0.1, 0., 0.])])


def test_triangular_pi_flux_reports_instead_of_exiting(monkeypatch, capsys):
    """Two blockers. The time-reversal guard printed seven matrices to
    stdout before raising, and past it sat a bare exit() that made the
    `return h` unreachable, so no argument set could ever produce a
    Hamiltonian. The guard must report the refusal in the exception, and
    a Hamiltonian that passes it must come back."""
    with pytest.raises(ValueError):
        specialhamiltonian.triangular_pi_flux(has_spin=False)
    # the matrices used to be dumped to stdout before the raise; the
    # refusal belongs in the exception, not in the terminal
    assert "[[" not in capsys.readouterr().out
    # with the guard satisfied, the function must RETURN the Hamiltonian
    monkeypatch.setattr(hamiltonians.Hamiltonian,
                        "has_time_reversal_symmetry", lambda self: True)
    h = specialhamiltonian.triangular_pi_flux(has_spin=False)
    assert h.intra.shape[0] == 2  # the two-site pi-flux cell
