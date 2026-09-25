import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pyqula import geometry, kpm, dos, kdos
from pyqula.kpmtk import ldos, density

# The KPM DOS routines take a scale (10 by default) and expand m/scale, which
# is only meaningful when the spectrum of m lies inside [-scale,scale]. A
# scale that is too small used to be taken on trust and gave a diverging,
# meaningless profile. The guard is on the moments themselves, bounded
# exactly when the spectrum is inside, so on a matrix whose spectrum is
# exactly [-2,2] a scale of 2 is accepted and 1.999 refused.


def _matrix_with_spectrum_minus2_to_2(n=40, seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.random((n, n)) + 1j*rng.random((n, n)))
    return csc_matrix(q @ np.diag(np.linspace(-2., 2., n)) @ q.conj().T)


def _routes(m):
    n = m.shape[0]
    v1, v2 = np.random.random(n), np.random.random(n)
    op = csc_matrix(np.diag(3.*np.random.random(n)))
    return {
        "tdos": lambda s: kpm.tdos(m, scale=s, npol=200, ne=200),
        "tdos_single": lambda s: kpm.tdos(m, scale=s, npol=200, ne=200,
                                          kpm_prec="single"),
        "tdos_operator": lambda s: kpm.tdos(m, scale=s, npol=200, ne=200,
                                            operator=op),
        "ldos": lambda s: ldos.get_ldos(m, i=3, scale=s, npol=200, ne=200),
        "dm_ij_energy": lambda s: kpm.dm_ij_energy(m, i=2, j=5, scale=s,
                                                   npol=200),
        "correlator0d": lambda s: kpm.correlator0d(m, i=2, j=5, scale=s,
                                                   npol=200, write=False),
        "dm_vivj_energy": lambda s: kpm.dm_vivj_energy(m, v1, v2, scale=s,
                                                       npol=200),
        "dos": lambda s: kpm.dos(m, np.linspace(-.9, .9, 10), n=200, scale=s),
        "density": lambda s: density.get_density(m, i=3, scale=s, npol=200),
    }


@pytest.mark.parametrize("route", list(_routes(_matrix_with_spectrum_minus2_to_2())))
def test_a_scale_at_the_spectral_radius_is_accepted(route):
    _routes(_matrix_with_spectrum_minus2_to_2())[route](2.0)


@pytest.mark.parametrize("route", list(_routes(_matrix_with_spectrum_minus2_to_2())))
def test_a_scale_below_the_spectral_radius_is_refused(route):
    with pytest.raises(ValueError, match="does not cover the spectrum"):
        _routes(_matrix_with_spectrum_minus2_to_2())[route](1.999)


def test_the_accepted_dos_integrates_to_one():
    """the guard does not change a valid result: the DOS per state of a
    matrix whose spectrum is inside the window integrates to one"""
    x, y = kpm.tdos(_matrix_with_spectrum_minus2_to_2(), scale=2.5, npol=200,
                    ne=2000, ntries=40)
    assert abs(np.trapezoid(y, x) - 1.) < 2e-2


# dos0d_sites, dos1d_sites and kdos1d_sites called kpm.local_dos, which does
# not exist, so they died with an AttributeError; and like dos0d_kpm they
# wrote a density per unit x=E/scale, integrating to scale times the number
# of states, where every other KPM DOS writes one per unit energy.


def _finite_chain(n=8):
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    return g.get_hamiltonian(has_spin=False)


def test_dos0d_sites_counts_the_states_of_the_sites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dos.dos0d_sites(_finite_chain(), sites=[0, 1, 2], scale=3., npol=200)
    d = np.loadtxt("DOS.OUT")
    assert abs(np.trapezoid(d[:, 1], d[:, 0]) - 3.) < 1e-3


def test_dos1d_sites_counts_the_states_of_the_sites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=False)
    dos.dos1d_sites(h, sites=[0], scale=3., nk=20, npol=200)
    d = np.loadtxt("DOS.OUT")
    assert abs(np.trapezoid(d[:, 1], d[:, 0]) - 1.) < 1e-3


def test_kdos1d_sites_runs_and_refuses_a_short_scale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=False)
    kdos.kdos1d_sites(h, sites=[0], scale=3., nk=5, npol=100)
    assert len(np.genfromtxt("KDOS.OUT", dtype=str)) == 25
    with pytest.raises(ValueError, match="does not cover the spectrum"):
        kdos.kdos1d_sites(h, sites=[0], scale=1.5, nk=5, npol=100)


def test_dos0d_kpm_is_per_unit_energy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dos.dos0d_kpm(_finite_chain(), scale=3., npol=200, ntries=40)
    d = np.loadtxt("DOS.OUT")
    assert abs(np.trapezoid(d[:, 1], d[:, 0]) - 1.) < 2e-2
