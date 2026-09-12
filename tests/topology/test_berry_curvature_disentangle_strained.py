import numpy as np
import pytest

from pyqula import geometry
from pyqula import potentials


def _valley_bands(amplitude, nk=40, kpath=None):
    """Valley-resolved bands of a honeycomb supercell(4) carrying a
    commensurate sublattice-imbalance potential of the given amplitude."""
    g = geometry.honeycomb_lattice()
    g = g.get_supercell(4)
    h = g.get_hamiltonian()
    h.remove_spin()
    f3 = potentials.commensurate_potential(g, minmax=[-amplitude, amplitude])
    h.add_sublattice_imbalance(f3)
    op = h.get_operator("valley")
    if kpath is not None:
        (k, e, c) = h.get_bands(operator=op, kpath=kpath)
    else:
        (k, e, c) = h.get_bands(operator=op, nk=nk)
    return np.array(k), np.array(e), np.array(c)


def _symmetric_kpath(n=12):
    """k-path visiting every point together with its opposite."""
    ks = []
    for t in np.linspace(0.02, 0.45, n):
        ks.append([t, t / 2., 0.])
        ks.append([-t, -t / 2., 0.])
    return np.array(ks)


@pytest.mark.slow
def test_valley_expectation_is_odd_in_k_and_survives_the_commensurate_potential(
        tmp_path, monkeypatch):
    """The valley operator is odd under time reversal and the sublattice
    potential is real, so the bands must be even in k while their valley
    expectation values are exactly odd: matching the bands at k and -k by
    energy, <valley> reverses sign. That is the statement the old sum(c)
    reference gestured at without making -- sum over a full band structure
    of any operator expectation is nk*Tr(O), so sum(c) is zero for every
    Hamiltonian, and sum(e) = sum_k Tr H(k) = 0 because the imbalance puts
    +f on one sublattice and -f on the other.

    Beside the symmetry, two statements about the amplitude itself: 0.6 is
    strong enough to gap the folded Dirac points (a hundred times weaker
    leaves them essentially degenerate) and still weak enough that the
    bands stay valley-polarised (ten times stronger mixes the valleys and
    the polarisation collapses).

    Marked slow: this is already the smallest meaningful supercell for the
    commensurate potential."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    amplitude = 0.6

    kpath = _symmetric_kpath()
    (k, e, c) = _valley_bands(amplitude, kpath=kpath)
    nb = len(e) // len(kpath)
    E, C = e.reshape(len(kpath), nb), c.reshape(len(kpath), nb)
    Ep, Em = E[0::2], E[1::2]
    op, om = np.argsort(Ep, axis=1), np.argsort(Em, axis=1)
    assert np.allclose(np.take_along_axis(Ep, op, 1),
                       np.take_along_axis(Em, om, 1), atol=1e-8)
    assert np.allclose(np.take_along_axis(C[0::2], op, 1),
                       -np.take_along_axis(C[1::2], om, 1), atol=1e-6)

    (k, e, c) = _valley_bands(amplitude)
    assert np.min(np.abs(e)) > 0.05  # the potential gaps the Dirac points
    assert np.max(np.abs(c)) > 0.9  # and the bands stay valley-polarised
