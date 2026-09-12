"""The non-Hermitian copies of get_bands_nd and get_dos drifted from the
Hermitian originals they were copied from: the bands ignored write and
output_file (so h.get_bands(write=True) wrote nothing where the Hermitian
sibling writes BANDS.OUT) and died with NameError on num_bands, and the
DOS overwrote the caller's mode with 'ED' unconditionally, so a typo was
absorbed silently where the Hermitian twin lists the accepted values."""

import numpy as np
import pytest

from pyqula import geometry


def _nh(n=4):
    """A non-Hermitian chain: a complex onsite, which is what makes the
    spectrum complex and sends h.get_bands down the non-Hermitian path."""
    g = geometry.chain().get_supercell(n)
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    h.add_onsite(0.3j)  # a uniform decay rate: the spectrum is complex
    assert h.non_hermitian
    return h


def test_non_hermitian_bands_write_a_file(tmp_path, monkeypatch):
    """write=True has to produce the file it names, as the Hermitian
    get_bands_nd does from the same call. The file must also carry the
    same energies as the returned array -- both parts of a complex
    eigenvalue, since dropping the imaginary part is dropping the physics
    a non-Hermitian Hamiltonian was used for."""
    monkeypatch.chdir(tmp_path)
    h = _nh()
    out = h.get_bands(nk=8, write=True)
    assert (tmp_path / "BANDS.OUT").exists()
    m = np.genfromtxt("BANDS.OUT").T
    assert m.shape[1] == out.shape[1]          # one row per band and kpoint
    assert np.allclose(m[0], out[0].real)      # kpoint index
    assert np.allclose(m[1], out[1].real)      # real part of the energy
    assert np.allclose(m[2], out[1].imag)      # and the imaginary part
    assert np.max(np.abs(m[2])) > 1e-6         # which is not zero here


def test_non_hermitian_bands_honour_output_file(tmp_path, monkeypatch):
    """output_file names the file, and write=False writes none."""
    monkeypatch.chdir(tmp_path)
    h = _nh()
    h.get_bands(nk=6, write=True, output_file="NH_BANDS.OUT")
    assert (tmp_path / "NH_BANDS.OUT").exists()
    h.get_bands(nk=6, write=False, output_file="NOT_WRITTEN.OUT")
    assert not (tmp_path / "NOT_WRITTEN.OUT").exists()


def test_non_hermitian_bands_accept_num_bands(tmp_path, monkeypatch):
    """num_bands used to raise NameError -- slg, arpack_tol and
    arpack_maxiter were never imported -- before doing any work. The few
    bands it returns must be the ones closest to central_energy, which the
    full diagonalization of the same Hamiltonian tells us."""
    monkeypatch.chdir(tmp_path)
    h = _nh(6)
    nb = 4
    out = h.get_bands(nk=3, num_bands=nb, write=False)
    assert out.shape[1] == nb*3  # nb bands at each of the 3 kpoints
    full = h.get_bands(nk=3, write=False)
    for ik in range(3):
        es = out[1][out[0].real.astype(int) == ik]
        ref = full[1][full[0].real.astype(int) == ik]
        # every band arpack returns is one of the bands the full
        # diagonalization finds at that same kpoint
        for e in es:
            assert np.min(np.abs(ref - e)) < 1e-6


def test_non_hermitian_dos_refuses_a_mode_it_does_not_implement(tmp_path,
                                                                monkeypatch):
    """Every mode string, typos included, used to return the same ED
    array. The Hermitian twin raises a ValueError listing what it takes;
    an unimplemented mode here must not be silently swapped for another
    one."""
    monkeypatch.chdir(tmp_path)
    h = _nh()
    energies = np.linspace(-2., 2., 10)
    (x, y) = h.get_dos(energies=energies, mode="ED")  # the one that works
    assert len(y) == len(energies)
    with pytest.raises((ValueError, NotImplementedError)):
        h.get_dos(energies=energies, mode="bogus")
    with pytest.raises((ValueError, NotImplementedError)):
        h.get_dos(energies=energies, mode="KPM")
    with pytest.raises((ValueError, NotImplementedError)):
        h.get_dos(energies=energies, use_kpm=True)
