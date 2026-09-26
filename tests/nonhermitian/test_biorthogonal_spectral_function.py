"""The two spectral functions of a non-Hermitian Hamiltonian.

Weighing each eigenstate by its right eigenvector, <R|O|R>/<R|R>, with a
Lorentzian of width delta at Re E, is get_kdos_bands' default. The Green's
function one, -Im Tr[O G]/pi with G = (w + i delta - H)^-1, puts the
biorthogonal weight <L|O|R>/<L|R> on a pole at the complex E, so that a
state with Im E < 0 is broadened by its lifetime (Kozii and Fu,
arXiv:1708.05841, Eq. 24); biorthogonal=True asks for it, from the
eigenstates with mode="ED" and from the Green's function with mode="green",
and the two have to agree to rounding."""

import numpy as np
import pytest
from scipy.linalg import eig

from pyqula import geometry


def modulated_chain(amplitude=0.3):
    """A four-cell chain supercell with a lossy onsite modulation, whose
    eigenvalues all have Im E < 0"""
    g = geometry.chain().get_supercell(4, store_primal=True)
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    h.add_onsite(lambda r: -1j * amplitude * (1. + np.cos(np.pi * r[0] / 2.)))
    return h


def lossy_chain(gamma):
    """A clean four-cell chain supercell with a uniform loss -i gamma, so
    that its eigenvalues are 2cos(k) - i gamma, degenerate where the bands
    of the chain fold onto each other"""
    g = geometry.chain().get_supercell(4, store_primal=True)
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    h.add_onsite(-1j * gamma)
    return h


KPATH = np.array([[0., 0., 0.], [0.21, 0., 0.], [0.5, 0., 0.]])
ENERGIES = np.linspace(-2.5, 2.5, 41)


@pytest.mark.parametrize("operator", [None, "unfold"])
@pytest.mark.parametrize("h", [modulated_chain(), lossy_chain(0.25)])
def test_eigenstates_and_green_function_agree(h, operator, tmp_path,
                                              monkeypatch):
    """k=0 and k=1/2 are where the folded bands are degenerate, which the
    left vectors taken as the rows of R^-1 have to survive"""
    monkeypatch.chdir(tmp_path)
    kw = dict(kpath=KPATH, operator=operator, energies=ENERGIES, delta=0.1,
              biorthogonal=True)
    ed = h.get_kdos_bands(mode="ED", **kw)[2]
    gr = h.get_kdos_bands(mode="green", **kw)[2]
    assert np.allclose(ed, gr, atol=1e-10)


def test_uniform_loss_broadens_by_the_lifetime(tmp_path, monkeypatch):
    """With E = 2cos(k) - i gamma the Green's function spectral function is
    a Lorentzian of width delta+gamma at each band, while the right
    eigenvector one keeps the width delta"""
    monkeypatch.chdir(tmp_path)
    gamma, delta = 0.25, 0.1
    h = lossy_chain(gamma)
    kw = dict(kpath=KPATH, energies=ENERGIES, delta=delta)
    bio = h.get_kdos_bands(biorthogonal=True, **kw)[2]
    right = h.get_kdos_bands(**kw)[2]
    for (i, k) in enumerate(KPATH):
        eps = np.real(eig(h.get_hk_gen()(k))[0])  # 2cos of the folded k
        def lorentz(w):
            return np.sum(w / ((ENERGIES[:, None] - eps[None, :])**2 + w**2),
                          axis=1) / np.pi
        sl = slice(i * len(ENERGIES), (i + 1) * len(ENERGIES))
        assert np.allclose(bio[sl], lorentz(delta + gamma), atol=1e-10)
        assert np.allclose(right[sl], lorentz(delta), atol=1e-10)


def test_biorthogonal_weights_add_up_to_the_trace(tmp_path, monkeypatch):
    """sum_n <L_n|O|R_n>/<L_n|R_n> = Tr O at every kpoint, exactly, where
    the right eigenvector weights add up to it only approximately; without
    loss the two weights are the same, state by state, which needs the
    right eigenvectors orthonormal inside the levels that k=0 and k=1/2
    fold together (scipy's eig returns some basis of such a level, and a
    level of weight 4 used to come out 4.17)"""
    monkeypatch.chdir(tmp_path)
    h = modulated_chain()
    n = h.intra.shape[0]
    (k, e, w) = h.get_bands(kpath=KPATH, operator="unfold", biorthogonal=True,
                            write=False)
    for ik in range(len(KPATH)):
        assert np.isclose(np.sum(w[np.real(k) == ik]), n, atol=1e-10)
    h0 = modulated_chain(amplitude=0.)
    kb, eb, wb = h0.get_bands(kpath=KPATH, operator="unfold",
                              biorthogonal=True, write=False)
    kr, er, wr = h0.get_bands(kpath=KPATH, operator="unfold", write=False)
    assert np.allclose(np.sort(np.real(wb)), np.sort(np.real(wr)), atol=1e-8)
    assert np.allclose(np.imag(wb), 0., atol=1e-8)


def test_what_cannot_be_asked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = modulated_chain()
    kw = dict(kpath=KPATH, energies=ENERGIES)
    with pytest.raises(ValueError):  # the Green's function is biorthogonal
        h.get_kdos_bands(mode="green", **kw)
    with pytest.raises(NotImplementedError):  # no real spectrum to expand
        h.get_kdos_bands(mode="KPM", **kw)
    with pytest.raises(ValueError):  # the lifetime is in Im E
        h.get_kdos_bands(biorthogonal=True, eigmode="real", **kw)
    with pytest.raises(NotImplementedError):  # no left vectors from ARPACK
        h.get_bands(kpath=KPATH, operator="unfold", biorthogonal=True,
                    num_bands=2, write=False)
    with pytest.raises(NotImplementedError):
        h.get_dos(biorthogonal=True, nk=4, write=False)
