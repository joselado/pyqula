import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpmtk.densitymatrix_kpm import (get_dm_kpm, required_elements,
        _estimate_kpm_scale)
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm
from pyqula.scftk.spinspin import VJinteraction

# The Chebyshev expansion is defined only for a spectrum inside
# [-scale,scale], and a scale given by the caller used to be taken on
# trust: too small, the recursion overflowed into a NaN that surfaced as a
# root-finder error (Vinteraction_kpm) or as a loop iterating a NaN residual
# up to maxite (VJinteraction). The guard is on the moments themselves,
# which are bounded by one exactly when the spectrum is inside, so it must
# refuse a scale below the spectral radius and accept one above it even
# where the Gershgorin bound is larger.


def _gapped_honeycomb():
    """Spinless honeycomb with a sublattice imbalance of 1: the spectrum
    reaches sqrt(1+3**2)=3.16, while the Gershgorin bound on H(k) is 1+3=4"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(1.0)
    return h


def test_gershgorin_is_loose_here():
    """the premise of the next test: a valid scale of 3.3 sits between the
    spectral radius and the Gershgorin bound"""
    h = _gapped_honeycomb()
    ks = h.geometry.get_kmesh(nk=6)
    radius = max(np.max(np.abs(np.linalg.eigvalsh(h.get_hk_gen()(k))))
            for k in ks)
    assert radius < 3.3 < _estimate_kpm_scale(h.get_hk_gen(), ks)/1.1


def test_a_scale_above_the_spectral_radius_is_accepted():
    """3.3 covers the spectrum, so it must give the same density matrix as
    exact diagonalization, to the accuracy of the expansion"""
    h = _gapped_honeycomb()
    v = h.geometry.get_hamiltonian(has_spin=False).get_hopping_dict()
    ds = sorted({d for (d, i, j) in required_elements(v)})
    dm = get_dm_kpm(h, v, nk=6, npol=300, scale=3.3)
    dm_ed = h.get_density_matrix(ds=ds, nk=6)
    for (d, i, j) in required_elements(v):
        assert abs(dm[d][i, j] - dm_ed[d][i, j]) < 1e-2, (d, i, j)


def test_a_scale_below_the_spectral_radius_is_refused():
    h = _gapped_honeycomb()
    v = h.geometry.get_hamiltonian(has_spin=False).get_hopping_dict()
    with pytest.raises(ValueError, match="does not cover the spectrum"):
        get_dm_kpm(h, v, nk=6, npol=300, scale=3.0)


# A spinful chain has its spectrum in [-2,2], but at filling=0.1 the loop
# expands the Hamiltonian after a Fermi shift of about 1.9, so a scale of
# 2.5 passes the Fermi search on the unshifted Hamiltonian and fails only on
# the shifted one.

def test_vinteraction_kpm_refuses_a_scale_short_of_the_shifted_spectrum(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError, match="does not cover the spectrum"):
        Vinteraction_kpm(h, U=1.0, filling=0.1, nk=10, npol=100, scale=2.5,
                maxite=3, load_mf=False, verbose=0)


def test_vjinteraction_kpm_refuses_a_scale_short_of_the_shifted_spectrum(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError, match="does not cover the spectrum"):
        VJinteraction(h, U=1.0, filling=0.1, nk=10, npol=100, scale=2.5,
                maxite=3, integration="kpm", verbose=0)
