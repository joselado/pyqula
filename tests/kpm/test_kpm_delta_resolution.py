import numpy as np
import pytest

from pyqula import geometry
from pyqula.ldos import dos_site


def _dimer():
    """Two sites joined by a hopping: two levels at -1 and +1, far enough
    apart that each peak can be measured on its own"""
    g = geometry.chain().get_supercell(2)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=False)


def _hwhm(x, y):
    """Half width at half maximum of the single peak of y(x), from the two
    half-maximum crossings, interpolated linearly between grid points"""
    k = np.argmax(y)
    half = y[k]/2.
    left = np.where(y[:k] < half)[0][-1] # last point below half, left side
    right = k + np.where(y[k:] < half)[0][0] # first point below half, right
    xl = np.interp(half, [y[left], y[left+1]], [x[left], x[left+1]])
    xr = np.interp(half, [y[right], y[right-1]], [x[right], x[right-1]])
    return (xr-xl)/2.


def test_delta_is_the_same_peak_width_in_every_mode(tmp_path, monkeypatch):
    """delta is the half width at half maximum of the peak that a single
    level gives, in the exact-diagonalization mode (a Lorentzian of width
    delta) and in the two KPM routes, the stochastic DOS of
    h.get_dos(mode="KPM") and the local DOS of dos_site(mode="KPM").

    The two KPM routes used to turn delta into different numbers of
    Chebyshev moments, 2*scale/delta for the DOS and 10*scale/delta for the
    local DOS, so the same delta gave peaks of half width 1.86*delta and
    0.37*delta. The Jackson kernel broadens a level into a near Gaussian of
    standard deviation pi*scale/N for N moments (Weisse et al., Rev. Mod.
    Phys. 78, 275 (2006), arXiv:cond-mat/0504627, Eqs. (75) and (76)),
    whose half width is sqrt(2 ln 2) times that"""
    monkeypatch.chdir(tmp_path)
    h = _dimer()
    es = np.linspace(0.6, 1.4, 1601) # fine grid around the level at +1
    for delta in [0.05, 0.1]:
        np.random.seed(0)
        (_, y_ed) = h.get_dos(mode="ED", energies=es, delta=delta, write=False)
        (_, y_kpm) = h.get_dos(mode="KPM", energies=es, delta=delta,
                               ntries=4, write=False)
        (_, y_site) = dos_site(h, i=0, mode="KPM", energies=es, delta=delta)
        assert np.isclose(_hwhm(es, y_ed), delta, rtol=0.03)
        assert np.isclose(_hwhm(es, y_kpm), delta, rtol=0.1)
        assert np.isclose(_hwhm(es, y_site), delta, rtol=0.1)


def test_jackson_npol_and_hwhm_are_inverse():
    """jackson_hwhm(scale, jackson_npol(scale, delta)) gives back delta, up
    to rounding npol up to an integer, and a non-positive delta is refused"""
    from pyqula.kpmtk.kernels import jackson_npol, jackson_hwhm
    for scale in [3., 10.]:
        for delta in [1e-3, 1e-2, 0.3]:
            npol = jackson_npol(scale, delta)
            assert jackson_hwhm(scale, npol) <= delta
            assert jackson_hwhm(scale, npol-1) > delta
    for delta in [0., -0.1]:
        with pytest.raises(ValueError, match="delta"):
            jackson_npol(10., delta)
