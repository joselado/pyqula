import os

import numpy as np

from pyqula import convolution, geometry, interpolation
from pyqula.fermisurface import fermi_surface_generator


def test_selfconvolve_fft_matches_brute_force_on_random_arrays():
    """selfconvolve_fft (Wiener-Khinchin, O(n^2 log n)) must reproduce
    selfconvolve_jit (the O(n^4) reference double loop) exactly, up to
    float roundoff, for arbitrary periodic 2d data."""
    rng = np.random.default_rng(0)
    for n in [5, 6, 7, 8, 16]:
        ds = rng.standard_normal((n, n))
        ref = convolution.selfconvolve_jit(ds, ds * 0.0)
        fft = convolution.selfconvolve_fft(ds)
        rel = np.max(np.abs(ref - fft)) / np.max(np.abs(ref))
        assert rel < 1e-8, n


def _honeycomb_cases():
    """A handful of physically distinct Hamiltonians (spin-orbit coupling,
    exchange, both together, and a different lattice) exercising get_qpi's
    mode="pm" convolution path with real k-resolved DOS data, not just
    synthetic arrays."""
    cases = {}
    g = geometry.honeycomb_lattice()
    cases["plain_honeycomb"] = g.get_hamiltonian()
    h = g.get_hamiltonian(); h.add_soc(0.15)
    cases["soc_honeycomb"] = h
    h = g.get_hamiltonian(); h.add_exchange([0.1, 0.05, 0.2])
    cases["exchange_honeycomb"] = h
    h = g.get_hamiltonian(); h.add_soc(0.15); h.add_exchange([0.1, 0.0, 0.15])
    cases["soc_exchange_honeycomb"] = h
    gk = geometry.kagome_lattice()
    h = gk.get_hamiltonian(); h.add_soc(0.1)
    cases["soc_kagome"] = h
    return cases


def test_selfconvolve_fft_matches_brute_force_on_physical_dos_grids():
    """Run the same DOS(k)-on-a-grid pipeline poor_man_qpi_convolve uses
    (fermi_surface_generator + periodic 2d interpolation) for Hamiltonians
    with SOC, exchange, and both combined, and check the FFT convolution
    agrees with the brute-force reference on that real data."""
    nk, nq = 12, 10
    for name, h in _honeycomb_cases().items():
        es, ks, ds = fermi_surface_generator(h, reciprocal=False,
                energies=[0.1], delta=0.15, full_bz=True, nsuper=1, nk=nk)
        d = ds[:, 0]
        grid_kx, grid_ky = np.mgrid[0:1:nq * 1j, 0:1:nq * 1j]
        f = interpolation.interpolator2d(ks[:, 0], ks[:, 1], d,
                mode="periodic")
        ksg = np.array([grid_kx, grid_ky]).reshape((2, nq * nq)).T
        dsg = f(ksg).reshape((nq, nq))
        ref = convolution.selfconvolve_jit(dsg, dsg * 0.0)
        fft = convolution.selfconvolve_fft(dsg)
        rel = np.max(np.abs(ref - fft)) / np.max(np.abs(ref))
        assert rel < 1e-8, name


def _run_get_qpi(h, use_fft):
    """Run get_qpi(mode="pm") with either the FFT or brute-force self
    convolution and return the per-energy QPI pattern (q,ω) array."""
    orig = convolution.selfconvolve
    if not use_fft:
        convolution.selfconvolve = lambda ds: convolution.selfconvolve_jit(
                ds, ds * 0.0)
    try:
        h.get_qpi(mode="pm", nk=8, energies=np.linspace(-3, 3, 4),
                delta=0.2, output_folder="MULTIQPI")
    finally:
        convolution.selfconvolve = orig
    files = sorted(fn for fn in os.listdir("MULTIQPI") if fn.endswith("_.OUT"))
    return np.array([np.loadtxt(os.path.join("MULTIQPI", fn))[:, 2]
            for fn in files])


def test_get_qpi_pm_mode_fft_matches_brute_force_end_to_end(tmp_path,
        monkeypatch):
    """End-to-end check of the actual public get_qpi(mode="pm") entry
    point: the FFT-accelerated convolution must reproduce the same QPI
    pattern as the original O(n^4) convolution for Hamiltonians with SOC,
    exchange, and both together."""
    for name, h in _honeycomb_cases().items():
        d = tmp_path / name
        d.mkdir()
        monkeypatch.chdir(d)
        old = _run_get_qpi(h, use_fft=False)
        new = _run_get_qpi(h, use_fft=True)
        rel = np.max(np.abs(old - new)) / np.max(np.abs(old))
        assert rel < 1e-6, name
