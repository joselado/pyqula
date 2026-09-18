"""The jax/GPU pair-basis Lindhard kernel must reproduce the numba one.

chitk/pairchijax.py rewrites chitk/pairchi.py's `_accumulate` -- a
four-fold loop over band pairs, pairs of pair operators and frequencies --
as one GEMM per frequency over a *gathered* list of the band pairs that
contribute, padded to a fixed length so that jax compiles the kernel once
for a whole k-mesh rather than once per k-point. The gather and the
padding are both places where a wrong index would still produce a
plausible-looking response, so these tests diff against the numba kernel
rather than against expectations.

The selection rule is this module's own and differs from the site-basis
kernel's: `_accumulate` drops a band pair on `f_n - f_m == 0` exactly,
where chiAB_matrix drops it below delta/100. Reproducing the wrong one
would be invisible at zero temperature and wrong at finite temperature, so
both are covered below.

The end-to-end tests assert with a spy that pyqula.gpu.set_gpu(True)
*reaches* the kernel, not merely that the answer is right: a backend that
never took effect would silently give the right answer on the CPU, which
is how the analogous KPM switch landed broken the first time. Note that
this route used to refuse the device outright (chitk/spinchi.py raised for
any interaction that couples different sites), so "the switch is honoured
here at all" is itself the new behaviour.

They run through jax's CPU fallback on a machine with no GPU, which
validates correctness and says nothing about speed -- see
future_development/gpu_rpa_spin_response.md for the device measurements.
"""
import functools

import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk.interaction import bare_interaction
from pyqula.chitk import pairchi
from pyqula.meanfield import VJinteraction
from testutils import gpu_backend

NK = 4  # small on purpose: these are correctness tests, not benchmarks
ENERGIES = np.linspace(0.05, 1.2, 8)
DELTA = 0.05
# chi_prec and the relative agreement with the double precision numba
# kernel it must reach, for the bare response and for the RPA-dressed one.
# Same shape as tests/chi/test_chi_gpu.py's table, and for the same reason:
# a float32 GEMM lands ~1e-7 from double however carefully its inputs are
# formed, and the dressing amplifies that by the condition number of
# 1 + K*chi0, which is large next to a Goldstone mode
PRECISIONS = [("double", 1e-10, 1e-10), ("single", 1e-5, 1e-3)]


# ---------------------------------------------------------------- systems


@functools.lru_cache(maxsize=None)
def _neel_exchange():
    """A J1 Neel honeycomb: an exchange interaction, so the pair basis
    carries the transverse rung through the recorded spin channels"""
    g = geometry.honeycomb_lattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                         nk=NK, maxerror=1e-9, mix=0.3, maxite=3000,
                         J1=3.0).hamiltonian


@functools.lru_cache(maxsize=None)
def _doped_v1_chain():
    """A partially filled (metallic) chain ordered by a neighbour-shell V1,
    where the occupations are not a clean step and every band pair counts"""
    g = geometry.chain().get_supercell(2)
    g.get_sublattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.3, mf="ferro",
                         nk=NK, maxerror=1e-9, mix=0.3, maxite=3000,
                         U=2.0, V1=0.5).hamiltonian


# the SCFs above are the slow part of this file, so they are cached and the
# tests share the converged Hamiltonians rather than re-converging them
SYSTEMS = {"exchange": _neel_exchange, "metallic": _doped_v1_chain}


def _pairs(h):
    norb = h.get_multicell().get_dense().intra.shape[0]
    return pairchi.spinorbital_pairs(bare_interaction(h), norb)[0]


def _chi0(h, pairs, on_gpu, chi_prec=None, q=(0.2, 0.1, 0.), T=None):
    with gpu_backend(on_gpu):
        return pairchi.pair_chi0(h, pairs, q=list(q), energies=ENERGIES,
                                 delta=DELTA, nk=NK, T=T, chi_prec=chi_prec)


def _kmesh_input(h, pairs, q=(0.2, 0.1, 0.)):
    """Everything pair_chi0_kmesh_gpu takes, built the way pair_chi0 builds
    it -- for the tests that reach past pair_chi0 to pass a pair_pad, which
    is a device-side implementation detail and not an argument of the
    public routine"""
    from pyqula import algebra
    hd = h.get_multicell().get_dense()
    hk = hd.get_hk_gen()
    g = hd.geometry
    qv = np.array(q, dtype=np.float64)
    ks = [np.array(k, dtype=np.float64) for k in g.get_kmesh(nk=NK)]
    hks1 = np.array([algebra.todense(hk(k)) for k in ks], dtype=np.complex128)
    hks2 = np.array([algebra.todense(hk(k + qv)) for k in ks],
                    dtype=np.complex128)
    phases = np.array([[g.bloch_phase(p[2], k + qv) for p in pairs]
                       for k in ks], dtype=np.complex128)
    iP = np.array([p[0] for p in pairs], dtype=np.int64)
    jP = np.array([p[1] for p in pairs], dtype=np.int64)
    return hks1, hks2, phases, iP, jP


# ------------------------------------------------------------- the kernel


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("q", [[0., 0., 0.], [0.2, 0.1, 0.]])
@pytest.mark.parametrize("T", [None, DELTA])
def test_gpu_kernel_matches_the_numba_kernel(system, q, T):
    """The device path is a drop-in for the per-k loop over _accumulate:
    same (npair,npair,nw) output, to the last bits of complex128.

    T=None is the step occupations, where only occupied-empty band pairs
    survive the gather; T=delta is the finite-temperature case, where
    _accumulate keeps essentially all of them"""
    pytest.importorskip("jax")
    h = SYSTEMS[system]()
    pairs = _pairs(h)
    ref = _chi0(h, pairs, False, q=q, T=T)
    got = _chi0(h, pairs, True, chi_prec="double", q=q, T=T)
    assert got.shape == ref.shape
    assert got.dtype == np.complex128
    assert np.max(np.abs(ref)) > 1e-3, "this system has no response to check"
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(ref))


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_single_precision_is_close_to_double_but_not_identical(system):
    """chi_prec='single' rounds the contraction in complex64 and comes back
    complex128. It must agree with double to rounding -- and not to the
    last bit, which would mean single precision never ran"""
    pytest.importorskip("jax")
    h = SYSTEMS[system]()
    pairs = _pairs(h)
    ref = _chi0(h, pairs, False)
    got = _chi0(h, pairs, True, chi_prec="single")
    assert got.dtype == np.complex128
    err = np.max(np.abs(ref - got))/np.max(np.abs(ref))
    assert 1e-12 < err < 1e-5, f"relative error {err}"


def test_the_gather_keeps_the_pairs_a_finite_temperature_adds():
    """_accumulate skips a band pair only when f_n-f_m is exactly zero, not
    when it is small. Borrowing the site-basis kernel's delta/100 cutoff
    instead would agree at T=None and quietly drop contributions here"""
    pytest.importorskip("jax")
    from pyqula.chitk import pairchijax
    h = _doped_v1_chain()
    hd = h.get_multicell().get_dense()
    hk = hd.get_hk_gen()
    from pyqula import algebra
    k = np.array([0.1, 0., 0.])
    # a small q: the two spectra then differ by much less than the width
    # the site-basis cutoff discards, so the band pairs that contribute do
    # so with a small but strictly nonzero occupation difference. This is
    # the q -> 0 limit of the response, not a contrived input
    e1 = algebra.eigh(hk(k))[0]
    e2 = algebra.eigh(hk(k + np.array([1e-6, 0., 0.])))[0]
    f1 = pairchi._occupations(e1, DELTA)
    f2 = pairchi._occupations(e2, DELTA)
    df = f1[:, None] - f2[None, :]
    exact = int(np.sum(df != 0.))  # this module's rule
    cutoff = int(np.sum(np.abs(df) >= DELTA/100.))  # the site-basis one
    assert cutoff < exact, "the two selection rules agree here, so this "\
                           "system cannot tell them apart"
    facs = pairchijax.band_pair_plan(e1, e2, f1, f2)[2]
    assert int(np.sum(np.asarray(facs) != 0.)) == exact


@pytest.mark.parametrize("extra", [1, 7, 33])
def test_padding_does_not_change_the_answer(extra):
    """Padding entries carry f_n-f_m = 0, so a list padded to a length that
    is neither the survivor count nor a round number must give exactly the
    same response"""
    pytest.importorskip("jax")
    from pyqula.chitk import pairchijax
    h = _neel_exchange()  # gapped: a step occupation, so the gather bites
    pairs = _pairs(h)
    hd = h.get_multicell().get_dense()
    e1 = np.linalg.eigvalsh(hd.get_hk_gen()(np.array([0.1, 0.2, 0.])))
    f1 = pairchi._occupations(e1, None)
    count = int(np.sum(f1[:, None] - f1[None, :] != 0.))
    nb = len(e1)
    assert 0 < count < nb*nb, "this system does not exercise the gather"
    ref = _chi0(h, pairs, False)
    args = _kmesh_input(h, pairs)
    with gpu_backend():
        got = pairchijax.pair_chi0_kmesh_gpu(*args, ENERGIES, None, DELTA,
                                             pair_pad=min(count + extra,
                                                          nb*nb),
                                             chi_prec="double")
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(ref))


def test_too_small_a_pair_pad_raises_instead_of_truncating():
    """Silently dropping contributing band pairs would be a wrong answer
    that still looks like a response function"""
    pytest.importorskip("jax")
    from pyqula.chitk import pairchijax
    e = np.linspace(-1., 1., 6)
    f = pairchi._occupations(e, None)
    with pytest.raises(ValueError, match="pair_pad"):
        pairchijax.band_pair_plan(e, e, f, f, pair_pad=1)


def test_one_compilation_covers_a_whole_q_scan():
    """The padded length is quantized and chosen for the whole mesh, so a
    scan over several q-points must not retrace the kernel: a retrace per
    k-point is the failure mode that makes a device path slower than the
    CPU it replaced"""
    pytest.importorskip("jax")
    from pyqula.chitk import pairchijax
    h = _neel_exchange()
    pairs = _pairs(h)
    with gpu_backend():
        _chi0(h, pairs, True, chi_prec="double", q=(0., 0., 0.))  # compile
        before = pairchijax._chi_from_tensor_jit._cache_size()
        for q in ([0.1, 0., 0.], [0.2, 0.1, 0.], [0.3, 0.2, 0.]):
            _chi0(h, pairs, True, chi_prec="double", q=q)
    grown = pairchijax._chi_from_tensor_jit._cache_size() - before
    assert grown <= 1, f"{grown} new compilations over a 3-q scan"


# --------------------------------------------------------------- the ladder


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("chi_prec,tol_bare,tol_rpa", PRECISIONS)
def test_the_dressed_response_matches_the_cpu(system, chi_prec, tol_bare,
                                              tol_rpa):
    """The whole chain pair_chi_rpa -> pair_chi0 -> kernel, including the
    Dyson dressing and the contraction onto the spin operators"""
    pytest.importorskip("jax")
    h = SYSTEMS[system]()
    kw = dict(q=[0.15, 0., 0.], energies=ENERGIES, delta=DELTA, nk=NK)
    _, cpu = pairchi.pair_chi_rpa(h, **kw)
    with gpu_backend():
        _, got = pairchi.pair_chi_rpa(h, chi_prec=chi_prec, **kw)
    assert got.dtype == np.complex128
    assert np.max(np.abs(cpu)) > 1e-3
    assert np.max(np.abs(cpu - got)) < tol_rpa*np.max(np.abs(cpu))


@pytest.mark.parametrize("chi_prec,tol_bare,tol_rpa", PRECISIONS)
def test_the_public_spin_response_matches_the_cpu(chi_prec, tol_bare, tol_rpa):
    """get_spinchi_full on a Hamiltonian whose interaction couples
    different sites is rerouted to the pair basis by chitk/spinchi.py, and
    that reroute used to refuse the device outright"""
    pytest.importorskip("jax")
    h = _neel_exchange()
    kw = dict(q=[0.15, 0., 0.], energies=ENERGIES, delta=DELTA, nk=NK)
    _, cpu = h.get_spinchi_full(**kw)
    with gpu_backend():
        _, got = h.get_spinchi_full(chi_prec=chi_prec, **kw)
    cpu, got = np.array(cpu), np.array(got)
    assert np.max(np.abs(cpu)) > 1e-3
    assert np.max(np.abs(cpu - got)) < tol_rpa*np.max(np.abs(cpu))


def test_the_magnon_bands_match_the_cpu():
    """magnon_bands_pair scans a q-path, which on the CPU runs under
    parallel.pcall and on the device must stay in this process"""
    pytest.importorskip("jax")
    h = _neel_exchange()
    kw = dict(nq=2, nk=NK, energies=np.linspace(0.05, 3.0, 40), delta=0.1)
    qs, ws, gm = pairchi.magnon_bands_pair(h, **kw)
    with gpu_backend():
        qs2, ws2, gm2 = pairchi.magnon_bands_pair(h, chi_prec="double", **kw)
    assert len(ws) > 0, "no magnon found on the CPU, nothing to compare"
    assert np.array_equal(qs, qs2)
    assert np.max(np.abs(np.array(ws) - np.array(ws2))) < 1e-8


# ------------------------------------------------------------ reachability


class _Spy:
    """Wrap pairchijax.pair_chi0_kmesh_gpu and count how often it is used"""

    def __init__(self, monkeypatch):
        from pyqula.chitk import pairchijax
        self.calls = 0
        self.precisions = []
        real = pairchijax.pair_chi0_kmesh_gpu

        def wrapper(*args, **kwargs):
            self.calls += 1
            self.precisions.append(kwargs.get("chi_prec"))
            return real(*args, **kwargs)

        monkeypatch.setattr(pairchijax, "pair_chi0_kmesh_gpu", wrapper)


def test_the_switch_reaches_the_kernel_from_every_entry_point(monkeypatch):
    """Right answers are not evidence the switch works: the CPU path also
    gives right answers. Assert the device code actually ran"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = _neel_exchange()
    kw = dict(q=[0.15, 0., 0.], energies=ENERGIES, delta=DELTA, nk=NK)
    with gpu_backend():
        h.get_spinchi_full(**kw)
        assert spy.calls > 0, "get_spinchi_full ignored the switch"
        n = spy.calls
        h.get_spinchi_ladder(**kw)
        assert spy.calls > n, "get_spinchi_ladder ignored the switch"
        n = spy.calls
        h.get_transverse_spinchi(**kw)
        assert spy.calls > n, "get_transverse_spinchi ignored the switch"
        n = spy.calls
        h.get_magnon_bands(method="pair", nq=2, nk=NK, energies=ENERGIES,
                           delta=DELTA)
        assert spy.calls > n, "get_magnon_bands(method='pair') ignored it"


def test_chi_prec_reaches_the_kernel_and_defaults_to_single(monkeypatch):
    """chi_prec travels a kwarg chain of its own, so it can be dropped
    somewhere along it; a dropped chi_prec would silently run the default
    precision, which on the device is single"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = _neel_exchange()
    kw = dict(q=[0.15, 0., 0.], energies=ENERGIES, delta=DELTA, nk=NK)
    with gpu_backend():
        h.get_spinchi_full(**kw)
        h.get_spinchi_full(chi_prec="double", **kw)
        pairchi.pair_chi_rpa(h, chi_prec="double", **kw)
    assert spy.precisions == ["single", "double", "double"]


def test_the_default_backend_never_touches_the_device_code(monkeypatch):
    """The switch defaults to the CPU, and jax must stay out of the way
    there -- importing pairchijax flips process-global jax configuration,
    which the fork-based parallel.pcall pool of the CPU path must not
    inherit"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = _neel_exchange()
    h.get_spinchi_full(q=[0.15, 0., 0.], energies=ENERGIES, delta=DELTA,
                       nk=NK)
    assert spy.calls == 0


def test_single_precision_on_the_cpu_is_refused_not_silently_doubled():
    """The numba kernel is complex128 only. Answering in double after being
    asked for single is the silent-fallback failure the switch exists to
    avoid"""
    h = _neel_exchange()
    pairs = _pairs(h)
    with pytest.raises(NotImplementedError, match="chi_prec"):
        _chi0(h, pairs, False, chi_prec="single")
    with pytest.raises(ValueError, match="chi_prec"):
        _chi0(h, pairs, False, chi_prec="half")


# ------------------------------------------------------ physics invariants


@pytest.mark.slow
def test_goldstone_mode_survives_the_device_path():
    """A J1 Neel honeycomb has a Goldstone mode: the smallest eigenvalue of
    1 + K*chi0 at q=0, w=0 must go to zero with the broadening rather than
    sit at a gap. Agreement with the CPU kernel does not by itself
    guarantee this, because the kernel is a small difference of large
    numbers there -- which is also why single precision is checked and not
    assumed."""
    pytest.importorskip("jax")
    h = _neel_exchange()

    def residual(delta, on_gpu, chi_prec=None):
        with gpu_backend(on_gpu):
            _, K = pairchi.pair_rpa_kernel(h, q=[0., 0., 0.],
                                           energies=np.array([0.0]),
                                           delta=delta, nk=NK,
                                           chi_prec=chi_prec)
        return np.min(np.abs(np.linalg.eigvals(K[0])))

    for delta in (1e-2, 1e-3):
        cpu = residual(delta, False)
        dev = residual(delta, True, "double")
        # absolutely, against the O(1) scale of the kernel's own
        # eigenvalues: the residual itself is the near-cancellation being
        # measured (5e-8 at delta=1e-3), so a tolerance relative to it
        # would be asking the two backends to agree far below the double
        # precision of either
        assert abs(cpu - dev) < 1e-12
    # the residual of this mode goes as delta^2, so a hundredfold drop over
    # a tenfold delta; single precision must not put a floor under it
    for chi_prec in ("double", "single"):
        big = residual(1e-2, True, chi_prec)
        small = residual(1e-3, True, chi_prec)
        assert small < big/10., (f"{chi_prec}: residual went from {big} to "
                                 f"{small}, not a Goldstone mode")
