"""The jax/GPU Lindhard kernel must reproduce the numba one exactly.

chitk/chijax.py rewrites chiAB_matrix's four-fold loop as one GEMM per
frequency over a *gathered* list of the (a,b) pairs that survive the
occupation cutoff, padded to a fixed length so that jax compiles the
kernel once for a whole k-mesh instead of once per k-point. Both of those
-- the gather and the padding -- are places where a wrong index or an
off-by-one would still produce a plausible-looking response, so the tests
here diff against the numba kernel rather than against expectations.

They run through jax's transparent CPU fallback on a machine without a
GPU, which validates correctness and says nothing at all about speed: see
future_development/gpu_rpa_spin_response.md for why any speedup number has
to come from a measurement on an actual device.

The end-to-end tests assert that chi_cpugpu="GPU" *reaches* the kernel,
with a spy, not merely that the answer is right -- a dropped kwarg would
silently give the right answer on the CPU, which is exactly how the
analogous KPM switch landed broken the first time.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import chiAB as chiAB_mod
from pyqula.chitk.rpa import build_ops_projectors
from pyqula.chitk.spinchi import _full_spin_operators

NK = 4  # small on purpose: these are correctness tests, not benchmarks
ENERGIES = np.linspace(0.01, 1.0, 12)
DELTA = 0.05


# ---------------------------------------------------------------- systems


def _honeycomb_neel():
    """A gapped antiferromagnetic insulator"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="antiferro",
                                        nk=NK, maxerror=1e-8)


def _doped_ferro_chain():
    """A partially filled (metallic) ferromagnetic two-site chain"""
    h = geometry.chain().supercell(2).get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=2.0, filling=0.3, mf="ferro",
                                        nk=NK, maxerror=1e-8)


SYSTEMS = {"gapped": _honeycomb_neel, "metallic": _doped_ferro_chain}


def _kernel_input(h, q, nops=3, temp=DELTA, delta=DELTA):
    """Everything chiAB_matrix takes, built the way chiAB_q builds it"""
    from pyqula import algebra
    hk = h.get_hk_gen()
    k = np.array([0.1, 0.2, 0.])[:3]
    es1, ws1 = algebra.eigh(hk(k))
    es2, ws2 = algebra.eigh(hk(k + np.array(q)))
    ws1 = np.array(ws1.T, dtype=np.complex128)  # states as rows
    ws2 = np.array(ws2.T, dtype=np.complex128)
    ops = _full_spin_operators(h)[:nops]
    pAs, pBs = build_ops_projectors(h, ops)
    return (ws1, es1, ws2, es2, ENERGIES, np.array(pAs), np.array(pBs),
            temp, delta)


# ------------------------------------------------------------- the kernel


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("q", [[0., 0., 0.], [0.2, 0.1, 0.]])
@pytest.mark.parametrize("nops", [1, 3])  # the ladder and the full response
def test_gpu_kernel_matches_the_numba_kernel(system, q, nops):
    """chiAB_matrix_gpu is a drop-in for chiAB_matrix: same input, same
    (nw,ni,nj) output, to the last bits of complex128"""
    pytest.importorskip("jax")
    from pyqula.chitk import chijax
    h = SYSTEMS[system]()
    args = _kernel_input(h, q, nops=nops)
    ref = chiAB_mod.chiAB_matrix(*args)
    got = chijax.chiAB_matrix_gpu(*args)
    assert got.shape == ref.shape
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(ref))


def test_gpu_kernel_matches_at_low_temperature():
    """The occupation cutoff |f_a-f_b| < delta/100 discards every
    occupied-occupied and empty-empty pair once temp << the gap, which is
    the regime where the gather has the most to get wrong"""
    pytest.importorskip("jax")
    from pyqula.chitk import chijax
    h = _honeycomb_neel()
    args = _kernel_input(h, [0.2, 0.1, 0.], temp=1e-3, delta=1e-2)
    ref = chiAB_mod.chiAB_matrix(*args)
    got = chijax.chiAB_matrix_gpu(*args)
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(ref))


@pytest.mark.parametrize("extra", [1, 7, 33])
def test_padding_does_not_change_the_answer(extra):
    """Padding entries carry f_a-f_b = 0, so a pair list padded to a
    length that is not the survivor count -- and not a round number
    either -- must give exactly the same response"""
    pytest.importorskip("jax")
    from pyqula.chitk import chijax
    h = _doped_ferro_chain()
    args = _kernel_input(h, [0.2, 0., 0.])
    ws1, es1, ws2, es2 = args[0], args[1], args[2], args[3]
    n = len(es1)
    # how many pairs actually survive the cutoff, without any padding
    o1 = chijax._occupations(es1, DELTA)
    o2 = chijax._occupations(es2, DELTA)
    count = int(np.sum(np.abs(o1[:, None] - o2[None, :]) >= DELTA/100.))
    assert 0 < count < n*n, "this system does not exercise the gather at all"
    ref = chiAB_mod.chiAB_matrix(*args)
    got = chijax.chiAB_matrix_gpu(*args, pair_pad=min(count + extra, n*n))
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(ref))


def test_too_small_a_pair_pad_raises_instead_of_truncating():
    """Silently dropping contributing pairs would be a wrong answer that
    still looks like a response function"""
    pytest.importorskip("jax")
    from pyqula.chitk import chijax
    h = _doped_ferro_chain()
    _, es1, _, es2 = _kernel_input(h, [0.2, 0., 0.])[:4]
    with pytest.raises(ValueError):
        chijax.pair_plan(es1, es2, DELTA, DELTA, pair_pad=1)


# ----------------------------------------------------------- the k-mesh


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("q", [[0., 0., 0.], [0.2, 0.1, 0.]])
@pytest.mark.parametrize("rpa", [True, False])
def test_spinchi_full_matches_between_backends(system, q, rpa):
    """The whole chain get_spinchi_full -> chi_ops_RPA ->
    _chi_ops_matrix_vectorized -> chiAB -> chiAB_q -> kernel, on both
    backends, including the RPA dressing"""
    pytest.importorskip("jax")
    h = SYSTEMS[system]()
    kw = dict(q=q, nk=NK, energies=ENERGIES, delta=DELTA, RPA=rpa)
    _, cpu = h.get_spinchi_full(**kw)
    _, gpu = h.get_spinchi_full(chi_cpugpu="GPU", **kw)
    cpu, gpu = np.array(cpu), np.array(gpu)
    assert np.max(np.abs(cpu - gpu)) < 1e-10*np.max(np.abs(cpu))


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_spinchi_ladder_matches_between_backends(system):
    pytest.importorskip("jax")
    h = SYSTEMS[system]()
    kw = dict(q=[0.2, 0.1, 0.], nk=NK, energies=ENERGIES, delta=DELTA)
    _, cpu = h.get_spinchi_ladder(**kw)
    _, gpu = h.get_spinchi_ladder(chi_cpugpu="GPU", **kw)
    cpu, gpu = np.array(cpu), np.array(gpu)
    assert np.max(np.abs(cpu - gpu)) < 1e-10*np.max(np.abs(cpu))


def test_the_kernel_is_compiled_once_for_a_whole_q_scan():
    """A shape that varies with k or q turns one compilation into one per
    iteration and would eat the entire speedup -- this is the single
    largest implementation risk of the port, so it is asserted rather
    than assumed"""
    pytest.importorskip("jax")
    from pyqula.chitk import chijax
    h = _doped_ferro_chain()
    hk = h.get_hk_gen()
    ks = h.geometry.get_kmesh(nk=6)
    ops = _full_spin_operators(h)
    pAs, pBs = build_ops_projectors(h, ops)
    before = chijax._chi_from_tensors_jit._cache_size()
    for q in ([0., 0., 0.], [0.1, 0., 0.], [0.3, 0., 0.]):
        hks1 = np.array([hk(k) for k in ks])
        hks2 = np.array([hk(np.array(k) + np.array(q)) for k in ks])
        chijax.chi_matrix_kmesh_gpu(hks1, hks2, ENERGIES, np.array(pAs),
                                    np.array(pBs), DELTA, DELTA)
    grown = chijax._chi_from_tensors_jit._cache_size() - before
    assert grown <= 1, f"{grown} new compilations over a 3-q, 6-k scan"


# ------------------------------------------------------- reachability


class _Spy:
    """Wrap chijax.chi_matrix_kmesh_gpu and count how often it is used"""

    def __init__(self, monkeypatch):
        from pyqula.chitk import chijax
        self.calls = 0
        real = chijax.chi_matrix_kmesh_gpu

        def wrapper(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(chijax, "chi_matrix_kmesh_gpu", wrapper)


def test_the_gpu_kwarg_reaches_the_kernel_from_every_entry_point(monkeypatch):
    """Right answers are not evidence the switch works: the CPU path also
    gives right answers. Assert the device code actually ran"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = _doped_ferro_chain()
    h.get_spinchi_full(q=[0.2, 0., 0.], nk=NK, energies=ENERGIES,
                       delta=DELTA, chi_cpugpu="GPU")
    assert spy.calls > 0, "get_spinchi_full dropped chi_cpugpu"
    n = spy.calls
    h.get_spinchi_ladder(q=[0.2, 0., 0.], nk=NK, energies=ENERGIES,
                         delta=DELTA, chi_cpugpu="GPU")
    assert spy.calls > n, "get_spinchi_ladder dropped chi_cpugpu"
    n = spy.calls
    h.get_iets_ldos(nk=NK, delta=DELTA, e=0.1, chi_cpugpu="GPU")
    assert spy.calls > n, "get_iets_ldos dropped chi_cpugpu"
    n = spy.calls
    h.get_magnon_bands(method="rpa", nq=2, nk=NK, energies=ENERGIES,
                       delta=DELTA, chi_cpugpu="GPU")
    assert spy.calls > n, "get_magnon_bands(method='rpa') dropped chi_cpugpu"
    n = spy.calls
    h.get_qdos_iets(energies=ENERGIES, nq=2, nk=NK, delta=DELTA,
                    chi_cpugpu="GPU")
    assert spy.calls > n, "get_qdos_iets dropped chi_cpugpu"
    n = spy.calls
    h.get_densitychi_RPA(V1=0.5, q=[0.2, 0., 0.], nk=NK, energies=ENERGIES,
                         delta=DELTA, chi_cpugpu="GPU")
    assert spy.calls > n, "get_densitychi_RPA dropped chi_cpugpu"


def test_the_default_backend_never_touches_the_device_code(monkeypatch):
    """chi_cpugpu defaults to CPU, and jax must stay out of the way there
    -- importing chijax flips process-global jax configuration, which the
    fork-based parallel.pcall pool of the CPU path must not inherit"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = _doped_ferro_chain()
    h.get_spinchi_full(q=[0.2, 0., 0.], nk=NK, energies=ENERGIES,
                       delta=DELTA)
    assert spy.calls == 0


# --------------------------------------------------- refusals, not fallbacks


def test_unsupported_settings_raise_rather_than_falling_back():
    """A silent fallback to the CPU is the failure mode this switch exists
    to avoid, so every combination the device path does not implement is
    an error"""
    h = _doped_ferro_chain()
    ops = _full_spin_operators(h)
    pAs, pBs = build_ops_projectors(h, ops)
    common = dict(q=[0.2, 0., 0.], nk=2, energies=ENERGIES, delta=DELTA)
    with pytest.raises(ValueError):  # not a backend name
        chiAB_mod.chiAB_q(h, pAs=pAs, pBs=pBs, chi_cpugpu="gpu", **common)
    with pytest.raises(ValueError):  # trace/diagonal use a different kernel
        chiAB_mod.chiAB_q(h, mode="trace", chi_cpugpu="GPU", **common)
    with pytest.raises(ValueError):  # the adaptive integrator is per-point
        chiAB_mod.chiAB_q(h, pAs=pAs, pBs=pBs, imode="adaptive",
                          chi_cpugpu="GPU", **common)


# ------------------------------------------------------ physics invariants


@pytest.mark.slow
def test_goldstone_mode_survives_the_gpu_path():
    """The site-basis Goldstone check of test_magnon_goldstone_doped_chain,
    re-run on the device path: a saturated ferromagnetic chain's q=0,w=0
    RPA kernel residual must be a pure delta artifact (residual/delta
    roughly constant as delta shrinks), not a gap. Numerical agreement
    with the CPU kernel does not by itself guarantee this, because the
    kernel is inverted afterwards and the Goldstone eigenvalue is the
    small difference of large numbers."""
    pytest.importorskip("jax")
    from pyqula.chitk.rpa import rpa_kernel_ops
    from pyqula.chitk.spinchi import _full_spin_U
    h = geometry.chain().get_hamiltonian()
    hmf = h.get_mean_field_hamiltonian(U=10.0, filling=0.2, mf="ferro", nk=300)
    Ss = _full_spin_operators(hmf)
    Uv = _full_spin_U(hmf)

    def residual(delta, backend):
        _, kernels = rpa_kernel_ops(hmf, ops=Ss, V=Uv, q=[0., 0., 0.],
                                     energies=np.array([0.0]), delta=delta,
                                     nk=300, chi_cpugpu=backend)
        return np.min(np.abs(np.linalg.eigvals(kernels[0])))

    for delta in (0.02, 0.005):  # the two backends must not merely agree
        cpu = residual(delta, "CPU")   # with each other, they must both
        gpu = residual(delta, "GPU")   # show the delta-linear signature
        assert abs(cpu - gpu) < 1e-8*max(abs(cpu), 1e-12)
    ratio1 = residual(0.02, "GPU")/0.02
    ratio2 = residual(0.005, "GPU")/0.005
    assert 0.5 < ratio1/ratio2 < 2.0, (
        f"residual/delta moved from {ratio1} to {ratio2}: not a Goldstone mode")
