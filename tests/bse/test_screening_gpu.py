"""The jax/GPU static polarizability must reproduce the numba one.

bsetk/screeningjax.py rewrites bsetk/screening.py's `polarizability_jit`
-- a five-fold loop over q, the Brillouin zone, both band indices and the
orbital outer product -- as one GEMM per q-point over every transition,
with the contributing ones selected by a mask built on the device rather
than by a host-side gather (the module docstring says why this kernel
masks where chitk/chijax.py gathers, and what was measured).

The mask carries both of the numba kernel's skips: an occupation
difference of exactly zero, and a transition excluded by the cRPA window.
Getting either wrong still produces a plausible-looking response, so these
tests diff against the numba kernel rather than against expectations, and
re-run on the device the two exact identities bsetk/screening.py depends
on downstream -- Hermiticity, and chi0(-q) = conj(chi0(q)), which
kernel.build_blocks assumes when it writes the antiresonant block as
conj(W(Q)).

They run through jax's CPU fallback on a machine with no GPU, which
validates correctness and says nothing about speed; the device
measurements, including where this kernel does NOT pay, are in
future_development/gpu_rpa_spin_response.md.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb, gpu_backend
from pyqula.bsetk import screening as sc
from pyqula.bsetk.interaction import density_interaction, qkey

NK = 4  # small on purpose: these are correctness tests, not benchmarks
# chi_prec and the relative agreement with the double precision numba
# kernel it must reach. Single precision lands ~1e-7 from double however
# carefully its inputs are formed, the same as every other route
PRECISIONS = [("double", 1e-10), ("single", 1e-5)]


def _magnetic_chain():
    """A gapped chain with neither time-reversal nor a spin axis, which is
    the case that separates a correctly summed chi0 from one that took the
    'twice one ordering' shortcut"""
    h = gapped_ionic_chain()
    h.add_zeeman([0., 0., 0.3])
    h.add_exchange(lambda r: [0.15, 0.05, 0.25])
    return h


def _models():
    return [("chain", gapped_ionic_chain()),
            ("honeycomb", gapped_honeycomb(mass=1.0)),
            ("spinless", gapped_honeycomb(spinful=False, mass=1.0)),
            ("magnetic", _magnetic_chain())]


def _reverse(qs):
    """Pair each q with the index of -q on the same mesh"""
    index = {qkey(q): i for i, q in enumerate(qs)}
    return [(i, index[qkey(-q)]) for i, q in enumerate(qs)]


# ------------------------------------------------------------- the kernel


@pytest.mark.parametrize("name,h", _models())
@pytest.mark.parametrize("chi_prec,tol", PRECISIONS)
def test_gpu_polarizability_matches_the_numba_kernel(name, h, chi_prec, tol):
    """The device path is a drop-in for polarizability_jit: same
    (nq,norb,norb) output, to the last bits of complex128 in double"""
    pytest.importorskip("jax")
    qs, ref = sc.static_polarizability(h, nk=NK)
    with gpu_backend():
        qs2, got = sc.static_polarizability(h, nk=NK, chi_prec=chi_prec)
    assert got.shape == ref.shape
    assert got.dtype == np.complex128
    assert np.max(np.abs(ref)) > 1e-3, "this model has no response to check"
    assert np.max(np.abs(qs - qs2)) == 0.
    assert np.max(np.abs(ref - got)) < tol*np.max(np.abs(ref))


def test_single_precision_is_close_to_double_but_not_identical():
    """chi_prec='single' rounds the GEMM in complex64 and comes back
    complex128; agreeing to the last bit would mean single never ran"""
    pytest.importorskip("jax")
    h = gapped_honeycomb(mass=1.0)
    _, ref = sc.static_polarizability(h, nk=NK)
    with gpu_backend():
        got = sc.static_polarizability(h, nk=NK, chi_prec="single")[1]
    err = np.max(np.abs(ref - got))/np.max(np.abs(ref))
    assert 1e-12 < err < 1e-5, f"relative error {err}"


def test_the_crpa_window_reaches_the_device_kernel():
    """`allowed` is the second of the two masks, and it is what makes cRPA
    cRPA: a device path that ignored it would silently return the full RPA
    polarizability, which is a larger screening, not an error"""
    pytest.importorskip("jax")
    from pyqula import geometry
    onsite = [1.3, -0.9, 0.6, -1.1]
    h = geometry.chain().supercell(4).get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: onsite[int(round(r[0] - 0.5)) % 4])
    h = h.get_multicell().get_dense()
    exclude = ([1], [2])  # the frontier pair only, a genuine subset
    _, ref = sc.static_polarizability(h, nk=NK, exclude=exclude)
    _, full = sc.static_polarizability(h, nk=NK)
    with gpu_backend():
        got = sc.static_polarizability(h, nk=NK, exclude=exclude,
                                       chi_prec="double")[1]
    assert np.max(np.abs(ref - got)) < 1e-10*np.max(np.abs(full))
    # and the window actually removed something, so the check has teeth
    assert np.max(np.abs(full - ref)) > 1e-6*np.max(np.abs(full))


# --------------------------------------------------- the exact identities


@pytest.mark.parametrize("chi_prec,tol", PRECISIONS)
def test_reciprocity_and_hermiticity_hold_on_the_device(chi_prec, tol):
    """chi0 Hermitian and chi0(-q) = conj(chi0(q)), re-run on the device
    path on a model with no time-reversal symmetry. Both follow from
    summing BOTH orderings of the band pair, i.e. from the mask keeping
    every transition with df != 0 rather than half of them"""
    pytest.importorskip("jax")
    h = _magnetic_chain()
    with gpu_backend():
        qs, chi0 = sc.static_polarizability(h, nk=NK, chi_prec=chi_prec)
    scale = np.max(np.abs(chi0))
    for i, j in _reverse(qs):
        assert np.max(np.abs(chi0[j] - np.conj(chi0[i]))) < tol*scale
    for i in range(len(qs)):
        assert np.max(np.abs(chi0[i] - chi0[i].conj().T)) < tol*scale


def test_the_screened_interaction_matches_end_to_end():
    """The whole chain screened_interaction -> static_polarizability ->
    kernel, including the dielectric inversion, which is where a small
    error in chi0 would be amplified"""
    pytest.importorskip("jax")
    h = gapped_ionic_chain()
    V = density_interaction(h, U=0.6, V1=0.4)
    ref = sc.screened_interaction(h, V=V, nk=NK)
    with gpu_backend():
        got = sc.screened_interaction(h, V=V, nk=NK, chi_prec="double")
    assert np.max(np.abs(ref.Wq - got.Wq)) < 1e-10*np.max(np.abs(ref.Wq))
    assert abs(ref.epsmin - got.epsmin) < 1e-10


# ------------------------------------------------------------ reachability


class _Spy:
    """Wrap screeningjax.polarizability_gpu and count how often it is used"""

    def __init__(self, monkeypatch):
        from pyqula.bsetk import screeningjax
        self.calls = 0
        self.precisions = []
        real = screeningjax.polarizability_gpu

        def wrapper(*args, **kwargs):
            self.calls += 1
            self.precisions.append(kwargs.get("chi_prec"))
            return real(*args, **kwargs)

        monkeypatch.setattr(screeningjax, "polarizability_gpu", wrapper)


def test_the_switch_reaches_the_kernel_from_every_entry_point(monkeypatch):
    """Right answers are not evidence the switch works: the CPU path also
    gives right answers. Assert the device code actually ran"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    h = gapped_ionic_chain()
    V = density_interaction(h, U=0.6, V1=0.4)
    with gpu_backend():
        h.get_polarizability(nk=NK)
        assert spy.calls > 0, "get_polarizability ignored the switch"
        n = spy.calls
        h.get_screened_interaction(V=V, nk=NK)
        assert spy.calls > n, "get_screened_interaction ignored the switch"
        n = spy.calls
        h.get_bse(V=V, nk=NK, screening="rpa", nv=1, nc=1)
        assert spy.calls > n, "get_bse(screening='rpa') ignored the switch"
    # and single precision is what an unset chi_prec means on the device
    assert spy.precisions[0] == "single"


def test_the_default_backend_never_touches_the_device_code(monkeypatch):
    """The switch defaults to the CPU, and jax must stay out of the way
    there -- importing screeningjax flips process-global jax
    configuration, which the fork-based parallel.pcall pool must not
    inherit"""
    pytest.importorskip("jax")
    spy = _Spy(monkeypatch)
    sc.static_polarizability(gapped_ionic_chain(), nk=NK)
    assert spy.calls == 0


def test_single_precision_on_the_cpu_is_refused_not_silently_doubled():
    """polarizability_jit is complex128 only. Answering in double after
    being asked for single is the silent fallback the switch exists to
    avoid"""
    h = gapped_ionic_chain()
    with pytest.raises(NotImplementedError, match="chi_prec"):
        sc.static_polarizability(h, nk=NK, chi_prec="single")
    with pytest.raises(ValueError, match="chi_prec"):
        sc.static_polarizability(h, nk=NK, chi_prec="half")


def test_one_compilation_covers_repeated_calls():
    """Every device-side shape here is fixed by (nk,nb,norb) alone -- that
    is the point of masking rather than gathering -- so a second call on
    the same mesh must not retrace"""
    pytest.importorskip("jax")
    from pyqula.bsetk import screeningjax
    h = gapped_honeycomb(mass=1.0)
    with gpu_backend():
        sc.static_polarizability(h, nk=NK, chi_prec="double")  # compile
        before = screeningjax._chi0_all_q._cache_size()
        for _ in range(3):
            sc.static_polarizability(h, nk=NK, chi_prec="double")
    assert screeningjax._chi0_all_q._cache_size() == before
