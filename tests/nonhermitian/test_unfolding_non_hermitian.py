import numpy as np

from pyqula import geometry

_N = 10  # supercell multiplier


def _modulated_chain(amplitude):
    """Chain supercell of _N cells carrying a purely imaginary (hence
    non-Hermitian) onsite modulation commensurate with the supercell."""
    g0 = geometry.chain()
    g = g0.get_supercell(_N, store_primal=True)
    h = g.get_hamiltonian(has_spin=False, non_hermitian=True)
    omega = 1. / _N
    h.add_onsite(lambda r: amplitude * 1j * np.cos(np.pi * 2 * omega * r[0]))
    return g, h


def test_unfolding_recovers_the_primitive_chain_dispersion(tmp_path, monkeypatch):
    """Unfolding exists to map a supercell calculation back onto the
    primitive cell's Brillouin zone, so the test of it is that it returns
    the primitive chain: the band carrying the unfolding weight at each k
    must follow E(k) = 2t cos(k), to within the modulation that was added.
    Two further statements, neither of which the old recorded constants
    made:

    * spectral weight is conserved by unfolding -- the weights at each k sum
      to the number of supercell bands;
    * the modulation is i*amplitude*cos, a Hermitian hopping plus i times a
      real diagonal bounded by `amplitude`, so no eigenvalue can acquire a
      larger imaginary part than that.

    The old assertions were sum(es) -- which is sum_k Tr H(k), and the
    cosine modulation averages to zero over the supercell, so it vanishes
    for every amplitude -- sum(ds), the total unfolding weight, and sum of
    the whole get_kdos_bands array, which adds the k and energy *columns*
    to the spectral weights and so is dominated by the mesh coordinates."""
    monkeypatch.chdir(tmp_path)  # get_bands writes BANDS.OUT to cwd
    amplitude = 0.1
    (g, h) = _modulated_chain(amplitude)
    kpath = g.get_kpath() * _N
    (ks, es, ds) = h.get_bands(operator="unfold", kpath=kpath)
    ks, es, ds = np.array(ks), np.array(es), np.array(ds)
    nk = len(kpath)
    nb = len(es) // nk
    E, D = es.reshape(nk, nb), ds.reshape(nk, nb).real

    # spectral weight is conserved (exactly on average over the path; the
    # 3% per-k slack is the left/right eigenvector mismatch of a
    # non-Hermitian Hamiltonian)
    weight = np.sum(D, axis=1)
    assert abs(np.mean(weight) - nb) < 1e-3 * nb
    assert np.allclose(weight, nb, rtol=0.03)
    # ... and no eigenvalue leaves the strip |Im E| <= amplitude
    assert np.max(np.abs(es.imag)) <= amplitude + 1e-8

    # the unfolded band is the primitive chain's
    pristine = 2. * np.cos(2 * np.pi * kpath[:, 0] / _N)
    dominant = E[np.arange(nk), np.argmax(D, axis=1)].real
    assert np.max(np.abs(dominant - pristine)) < amplitude


def test_unfolded_kdos_peaks_on_the_primitive_chain_dispersion(tmp_path,
                                                               monkeypatch):
    """get_kdos_bands computes the same unfolded spectral function through
    the Green's function instead of the eigenvectors, so it must peak where
    the unfolded band sits: at each k inside the energy window, the KDOS
    maximum must land on E(k) = 2t cos(k) to within one point of the energy
    mesh. It must also be non-negative, being a spectral weight."""
    monkeypatch.chdir(tmp_path)
    amplitude = 0.1
    (g, h) = _modulated_chain(amplitude)
    kpath = g.get_kpath() * _N
    energies = np.linspace(0., 1., 30)
    out = np.array(h.get_kdos_bands(operator="unfold", kpath=kpath,
                                     energies=energies, eigmode="real"))
    kdos = out[2].reshape(len(kpath), len(energies))
    assert np.min(kdos) >= 0.

    pristine = 2. * np.cos(2 * np.pi * kpath[:, 0] / _N)
    inside = (pristine > 0.2) & (pristine < 0.8)
    assert np.sum(inside) > 0
    peak = energies[np.argmax(kdos, axis=1)]
    de = energies[1] - energies[0]
    assert np.max(np.abs(peak[inside] - pristine[inside])) < de
