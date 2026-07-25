import numpy as np

from pyqula import geometry, kpm
from pyqula.kpmtk.kpmnumba import kpm_moments_vivj


def _physical_cases():
    """A handful of physically distinct finite Hamiltonians (spin-orbit
    coupling, exchange, both together, and a different lattice), to check
    the fused numba KPM recursion against physics beyond the random
    dense matrices the other kpm/ tests use."""
    cases = {}
    g = geometry.honeycomb_lattice().get_supercell(4)
    g.dimensionality = 0
    cases["plain_honeycomb"] = g.get_hamiltonian(is_sparse=True)
    h = g.get_hamiltonian(is_sparse=True); h.add_soc(0.15)
    cases["soc_honeycomb"] = h
    h = g.get_hamiltonian(is_sparse=True); h.add_exchange([0.1, 0.05, 0.2])
    cases["exchange_honeycomb"] = h
    h = g.get_hamiltonian(is_sparse=True); h.add_zeeman([0., 0., 0.3])
    cases["zeeman_honeycomb"] = h
    h = g.get_hamiltonian(is_sparse=True); h.add_soc(0.15)
    h.add_exchange([0.1, 0.0, 0.15])
    cases["soc_exchange_honeycomb"] = h
    gk = geometry.kagome_lattice().get_supercell(3)
    gk.dimensionality = 0
    h = gk.get_hamiltonian(is_sparse=True); h.add_soc(0.1)
    cases["soc_kagome"] = h
    return cases


def test_fused_kpm_moments_v_matches_independent_reference():
    """kpm.get_moments_v (the fused numba recursion) must match
    kpm.python_kpm_moments (an independent, unfused m@v implementation)
    for real Hamiltonians with SOC, exchange, and both combined."""
    rng = np.random.default_rng(0)
    for name, h in _physical_cases().items():
        m = h.intra / 10.0  # rescale into KPM's (-1, 1) spectral window
        n = m.shape[0]
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v = v / np.sqrt(np.abs(np.vdot(v, v)))
        ref = kpm.python_kpm_moments(v.astype(complex), m, n=40)
        mus = kpm.get_moments_v(v, m, n=40)
        assert np.max(np.abs(mus - ref)) < 1e-6, name


def _dense_reference_ij_moments(m, vi, vj, n):
    """Independent, non-numba reference for <vj|T_n(H)|vi>, via the plain
    Chebyshev recursion done with dense numpy operations."""
    am = vi.copy()
    a = m @ vi
    mus = np.zeros(n, dtype=complex)
    mus[0] = np.vdot(vj, vi)
    mus[1] = np.vdot(vj, a)
    for i in range(2, n):
        ap = 2 * (m @ a) - am
        mus[i] = np.vdot(vj, ap)
        am, a = a, ap
    return mus


def test_fused_kpm_moments_ij_matches_dense_reference():
    """kpm_moments_vivj (the fused numba |vi><vj| recursion) must match a
    plain dense Chebyshev recursion for the same SOC/exchange
    Hamiltonians, for several site pairs."""
    for name, h in _physical_cases().items():
        m = h.intra / 10.0
        n = m.shape[0]
        md = np.asarray(m.todense()) if hasattr(m, "todense") else m
        for i, j in [(0, 0), (0, min(3, n - 1)), (min(1, n - 1), n - 1)]:
            vi = np.zeros(n, dtype=complex); vi[i] = 1.0
            vj = np.zeros(n, dtype=complex); vj[j] = 1.0
            ref = _dense_reference_ij_moments(md, vi, vj, 60)
            mus = kpm_moments_vivj(m, vi, vj, n=30)  # n=30 -> 2*n=60 moments
            assert np.max(np.abs(mus - ref)) < 1e-8, (name, i, j)
