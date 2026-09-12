import numpy as np
import pytest

from pyqula import geometry


def _ferro_chain(U, filling=0.2, nk=20):
    """Hubbard mean field on a chain, started from a ferromagnetic guess."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=U, filling=filling, mf="ferro", nk=nk)


def _spin_resolved(h, nk=20):
    """Sorted spin-up and spin-down band energies, plus the <sz> values."""
    (k, e, c) = h.get_bands(operator="sz", nk=nk)
    e, c = np.array(e), np.array(c)
    return np.sort(e[c > 0.5]), np.sort(e[c < -0.5]), c


@pytest.mark.slow
def test_strong_coupling_ferromagnetic_chain_saturates_to_full_polarisation(
        tmp_path, monkeypatch):
    """A Hubbard chain driven far past the Stoner threshold polarises
    completely: every electron ends up in one spin species, so the moment
    saturates at m = 2*filling and the mean field becomes a rigid spin
    splitting of exactly U*m = 2*U*filling. At U=10 and filling=0.2 that is
    4.0, which the converged mean field reproduces to machine precision; at
    a hundredth of that U the chain is far from saturation and the splitting
    is 0.01 rather than 0.04.

    The old assertion was sum(c) == 0, which is nk*Tr(sz) over a full band
    structure and is therefore zero for any Hamiltonian with a spin index,
    magnetic or not.

    The SCF mesh is nk=20 rather than the nk=4 the file used to pass: at
    nk=4 the loop converges to the *paramagnetic* solution (the up and down
    spectra come out exactly equal, even at U=40), so nothing the file's
    name promises was being computed. Marked slow: the SCF convergence
    drives the runtime."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT and MF.pkl to cwd
    U, filling = 10.0, 0.2
    h = _ferro_chain(U, filling=filling)
    (up, dn, c) = _spin_resolved(h)

    # a collinear mean field keeps sz a good quantum number
    assert np.allclose(np.abs(c), 1., atol=1e-6)
    assert len(up) == len(dn)
    # ... and acts as a rigid shift between the two spin species
    split = dn - up
    assert np.allclose(split, split[0], atol=1e-6)
    assert np.isclose(abs(split[0]), 2 * U * filling, atol=1e-5)

    # well below saturation the splitting is no longer U*2*filling
    (up0, dn0, c0) = _spin_resolved(_ferro_chain(0.1, filling=filling))
    assert abs((dn0 - up0)[0]) < 0.5 * 2 * 0.1 * filling
