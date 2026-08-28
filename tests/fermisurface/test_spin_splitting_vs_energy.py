import numpy as np
import pytest

from pyqula import algebra, specialhamiltonian
from pyqula.klist import kmesh


def _reference(h, nk):
    """Signed Delta_n(k) and Ebar_n(k), computed independently of the
    function under test: same spin blocks, but an explicit loop with no
    binning, so the two agree only if the pairing and the reduction are
    both right."""
    hup = h.copy(); hup.remove_spin(channel="up")
    hdn = h.copy(); hdn.remove_spin(channel="dn")
    hkup = hup.get_hk_gen(); hkdn = hdn.get_hk_gen()
    ks = kmesh(h.geometry.dimensionality, nk=nk)
    delta, ebar = [], []
    for k in ks:
        eup = np.sort(algebra.eigvalsh(hkup(k)))
        edn = np.sort(algebra.eigvalsh(hkdn(k)))
        delta.append(eup-edn)
        ebar.append((eup+edn)/2.)
    return np.concatenate(delta), np.concatenate(ebar)


def test_bz_maximum_matches_brute_force():
    """The curve's global maximum must be the largest |Delta| anywhere on
    the mesh, and every individual bin must hold the largest |Delta| among
    the pairs whose Ebar falls in it. This is the property the method
    exists for -- a bound over the whole zone rather than along one cut."""
    h = specialhamiltonian.square_altermagnet(am=1.)
    nk = 24
    E, D = h.get_spin_splitting_vs_energy(nk=nk, nbins=60)
    delta, ebar = _reference(h, nk)
    assert np.isclose(D.max(), np.max(np.abs(delta)))
    # bin by bin, not just the global maximum: a reduction that binned at
    # the wrong energy would still pass a check on the maximum alone
    edges = (E[1:]+E[:-1])/2.
    idx = np.searchsorted(edges, ebar)
    for i in range(len(E)):
        sel = idx == i
        expected = np.max(np.abs(delta[sel])) if np.any(sel) else 0.
        assert np.isclose(D[i], expected)


@pytest.mark.parametrize("am", [0.5, 1.0])
@pytest.mark.parametrize("nk", [50, 100])
def test_square_altermagnet_maximum_is_analytic(am, nk):
    """The d-wave altermagnet's splitting goes as am*(cos kx - cos ky) up to
    a factor, so its BZ maximum is exactly 4*am and does not depend on the
    mesh. Anything that broke the band-index pairing, the sign handling or
    the binning energy would move this number."""
    h = specialhamiltonian.square_altermagnet(am=am)
    E, D = h.get_spin_splitting_vs_energy(nk=nk, nbins=400)
    assert np.isclose(D.max(), 4.*am, atol=1e-10)
    # the extremum sits at the band center, not at a zone edge
    assert abs(E[np.argmax(D)]) < 0.05


def test_altermagnet_symmetries_of_the_signed_splitting():
    """Two symmetries of the d-wave altermagnet, on the signed Delta the
    public curve reduces away: the two spin sublattices are related by a
    rotation, so the set of splittings is symmetric about zero; and
    spinless time reversal gives Delta(-k) = Delta(k)."""
    h = specialhamiltonian.square_altermagnet(am=1.)
    delta, _ = _reference(h, nk=24)
    assert np.isclose(np.max(delta), -np.min(delta))
    # Delta(-k) = Delta(k), checked on explicit +-k pairs
    hup = h.copy(); hup.remove_spin(channel="up")
    hdn = h.copy(); hdn.remove_spin(channel="dn")
    hkup = hup.get_hk_gen(); hkdn = hdn.get_hk_gen()

    def d(k):
        return (np.sort(algebra.eigvalsh(hkup(k)))
                - np.sort(algebra.eigvalsh(hkdn(k))))

    for k in ([0.13, 0.27, 0.], [0.41, -0.09, 0.], [0.5, 0.25, 0.]):
        k = np.array(k)
        assert np.allclose(d(k), d(-k), atol=1e-10)


def test_noncollinear_hamiltonian_is_refused():
    """remove_spin drops the spin off-diagonal block without warning, so
    with Rashba the answer would be wrong rather than absent. It must
    raise instead."""
    h = specialhamiltonian.square_altermagnet(am=1.)
    h.add_rashba(0.3)
    with pytest.raises(ValueError):
        h.get_spin_splitting_vs_energy(nk=4, nbins=10)


def test_empty_bins_are_zero_and_outside_states_are_dropped():
    """Bins with no states return 0.0 rather than NaN, so the curve plots;
    and states outside an explicitly requested window are dropped rather
    than clamped onto the end bins, which would invent an edge peak that
    is not there."""
    h = specialhamiltonian.square_altermagnet(am=1.)
    delta, ebar = _reference(h, nk=24)
    # a window covering only part of the occupied range
    lo, hi = np.min(ebar)+0.1, np.min(ebar)+0.6
    energies = np.linspace(lo, hi, 40)
    E, D = h.get_spin_splitting_vs_energy(nk=24, energies=energies)
    assert not np.isnan(D).any()
    assert np.all(D >= 0.)
    # nothing from outside the window may appear in it
    half = (energies[1]-energies[0])/2.
    inside = (ebar >= lo-half) & (ebar <= hi+half)
    assert np.isclose(D.max(), np.max(np.abs(delta[inside])))
    assert D.max() < np.max(np.abs(delta))  # the window really does exclude


def test_matches_density_convention():
    """Same return convention as get_spin_splitting_density -- two 1D
    arrays of equal length -- so the two can be plotted together."""
    h = specialhamiltonian.square_altermagnet(am=1.)
    E, D = h.get_spin_splitting_vs_energy(nk=12, nbins=50)
    xs, ys = h.get_spin_splitting_density(nk=12, energies=np.linspace(-3., 3., 50))
    assert E.shape == D.shape == (50,)
    assert np.array(xs).shape == np.array(ys).shape
