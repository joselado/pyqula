"""greentk.selfenergy.bloch_selfenergy_batch must answer, for a whole set
of energies at once, exactly what bloch_selfenergy answers one at a time.

Only a 1d first-neighbour Hamiltonian solved by decimation actually
batches (one numba prange-parallel Sancho-Rubio call for the whole set);
everything else falls back to a loop, so the contract this file pins is
that the two branches are indistinguishable from the outside.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.greentk.selfenergy import bloch_selfenergy, bloch_selfenergy_batch


ENERGIES = np.array([-0.4, -0.05, 0.0, 0.13, 0.62])


def _scalar(h, es, **kwargs):
    out = [bloch_selfenergy(h, energy=e, **kwargs) for e in es]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]))


@pytest.mark.parametrize("gtype", ["bulk", "surface"])
def test_batch_matches_scalar_on_the_batched_branch(gtype):
    """A superconducting chain: 1d, first-neighbour, mode='adaptive' --
    the one shape that goes through green_renormalization_jit_batch."""
    h = geometry.chain().get_hamiltonian()
    h.shift_fermi(1.); h.add_swave(0.1)
    kwargs = dict(delta=1e-3, mode="adaptive", gtype=gtype)
    gr, sr = _scalar(h, ENERGIES, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES, **kwargs)
    assert gb.shape == gr.shape and sb.shape == sr.shape
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_matches_scalar_on_the_fallback_branch():
    """A 2d Hamiltonian is not batchable (the adaptive selfenergy there
    integrates over a transverse k), so this exercises the loop."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    kwargs = dict(delta=1e-2, mode="adaptive", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], **kwargs)
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_matches_scalar_for_longer_range_hoppings():
    """Second-neighbour hopping in 1d leaves the decimation for the Dyson
    route, which the batch does not reimplement either."""
    h = geometry.chain().get_hamiltonian(tij=[1., 0.4])
    kwargs = dict(delta=1e-2, mode="adaptive", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], **kwargs)
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def test_batch_refuses_an_unknown_gtype():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError):
        bloch_selfenergy_batch(h, ENERGIES, delta=1e-2, gtype="middle")


def test_batch_matches_scalar_on_the_tetrahedron_branch():
    """mode='triangle' of a 3d crystal batches too: one diagonalization of
    the mesh for every energy, the same simplex sum at each energy, so
    the matrices agree to rounding (the matrix products of the energies
    are grouped differently)"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    kwargs = dict(delta=0.05, mode="triangle", nk=6)
    gr, sr = _scalar(h, ENERGIES, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES, **kwargs)
    assert gb.shape == gr.shape and sb.shape == sr.shape
    assert np.max(np.abs(gr)) > 1e-2
    assert np.max(np.abs(gb-gr)) < 1e-12*np.max(np.abs(gr))
    assert np.max(np.abs(sb-sr)) < 1e-10*np.max(np.abs(sr))


def test_batch_passes_nk_to_the_fallback_loop():
    """mode='full' sums the Green's function over nk k-points, and the
    batch hands its nk to every call of the loop: the same matrices as the
    scalar calls with that nk, and not the ones of another nk"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    kwargs = dict(delta=5e-2, mode="full", gtype="bulk")
    gr, sr = _scalar(h, ENERGIES[:3], nk=7, **kwargs)
    go, so = _scalar(h, ENERGIES[:3], nk=9, **kwargs)
    gb, sb = bloch_selfenergy_batch(h, ENERGIES[:3], nk=7, **kwargs)
    assert np.max(np.abs(gr-go)) > 1e-3*np.max(np.abs(gr))
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.


def _diamond():
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    h.add_exchange([0.1, 0.2, 0.3])
    return h


def _mesh_green(h, z, nk):
    """(1/N) sum_k (z - H_k)^-1 on the uniform nk^3 mesh"""
    hk = h.get_hk_gen()
    iden = np.identity(h.intra.shape[0])
    ks = np.linspace(0., 1., nk, endpoint=False)
    out = 0.
    for a in ks:
        for b in ks:
            for c in ks:
                out = out + np.linalg.inv(z*iden - hk([a, b, c]))
    return out/nk**3


def test_the_tetrahedra_take_a_complex_energy_as_a_broadening():
    """The Green's function depends on z = energy + i delta alone, so
    mode='triangle' at a complex energy is the real energy Re z with the
    broadening Im z, and below the real axis it is the Hermitian
    conjugate. A complex energy was first cast to its real part with only a
    ComplexWarning, so an embedding got the bulk at the real energy and the
    Dyson step at the complex one, and then refused with a ValueError, which
    left the density-matrix contour of a 3d embedding without any mode.
    This test pins the contract: the two ways of writing one z end in the
    same call of the tetrahedra, so their agreement says nothing about the
    Green's function itself. The next test compares both half planes with
    an independent sum"""
    h = _diamond()
    iden = np.identity(h.intra.shape[0])
    intra = np.array(h.get_dense().intra)
    delta, nk = 0.05, 12
    kwargs = dict(mode="triangle", nk=nk)
    for z in [np.complex128(0.3+0.4j), 0.3+0.4j]:
        g, s = bloch_selfenergy(h, energy=z, delta=delta, **kwargs)
        g0, s0 = bloch_selfenergy(h, energy=z.real, delta=delta+z.imag,
                                  **kwargs)
        assert np.max(np.abs(g-g0)) < 1e-12*np.max(np.abs(g0))
        assert np.max(np.abs(s-s0)) < 1e-10*np.max(np.abs(s0))
        # the selfenergy is the one of that Green's function at z
        assert np.max(np.abs(s-((z+1j*delta)*iden-intra-np.linalg.inv(g)))
                      ) < 1e-10
        # the lower half plane, the other half of the contour
        gl, sl = bloch_selfenergy(h, energy=np.conj(z), delta=-delta,
                                  **kwargs)
        assert np.max(np.abs(gl-g.conj().T)) < 1e-12*np.max(np.abs(g))
        assert np.max(np.abs(sl-s.conj().T)) < 1e-10*np.max(np.abs(s))
        # not the Green's function at the real part of the energy
        gr = bloch_selfenergy(h, energy=z.real, delta=delta, **kwargs)[0]
        assert np.max(np.abs(g-gr)) > 0.1


@pytest.mark.parametrize("half", ["upper", "lower"])
def test_the_tetrahedra_at_a_complex_energy_converge_to_the_mesh_mean(half):
    """An independent route, in both half planes: far from the real axis
    the plain mean of (z - H_k)^-1 over a uniform mesh converges fast, and
    the tetrahedra approach it with their error of second order in 1/nk.
    The crystal has a gap and no band crossing, where the sorted bands
    would be interpolated with a kink, and a field along y, so that the
    Green's function is not a symmetric matrix and the lower half plane
    tells the Hermitian conjugate from the plain one"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(1.0)
    h.add_exchange([0.2, 0.3, 0.1])
    z, delta = 0.3+0.8j, 0.05
    if half == "lower": z, delta = np.conj(z), -delta
    ref = _mesh_green(h, z+1j*delta, 14)
    err = [np.max(np.abs(bloch_selfenergy(h, energy=z, delta=delta, nk=nk,
           mode="triangle")[0]-ref)) for nk in [6, 12]]
    assert err[1] < 2e-2*np.max(np.abs(ref))
    assert err[1] < 0.4*err[0]
    assert np.max(np.abs(ref-ref.T)) > 5.*err[1] # the transpose is seen


def test_the_tetrahedron_batch_groups_complex_energies_by_broadening(
        monkeypatch):
    """A batch of complex energies gives the single calls, with one call
    of the tetrahedra, one diagonalization of the mesh, for each value of
    |Im z|, so a sweep at real energies stays one call"""
    from pyqula.dostk import simplexdos
    calls = []
    green_triangle = simplexdos.green_triangle

    def counted(h, energies, **kw):
        calls.append(len(np.atleast_1d(energies)))
        return green_triangle(h, energies, **kw)

    h = _diamond()
    kwargs = dict(delta=0.0625, mode="triangle", nk=4)
    # Im z = 0.0625, 0.5625, 0.5625 and -0.5625, exact in binary
    es = np.array([0.1, 0.3+0.5j, -0.2+0.5j, 0.6-0.625j])
    gr, sr = _scalar(h, es, **kwargs)
    monkeypatch.setattr(simplexdos, "green_triangle", counted)
    gb, sb = bloch_selfenergy_batch(h, es, **kwargs)
    assert sorted(calls) == [1, 3]
    assert np.max(np.abs(gb-gr)) < 1e-12*np.max(np.abs(gr))
    assert np.max(np.abs(sb-sr)) < 1e-10*np.max(np.abs(sr))
    calls.clear()
    bloch_selfenergy_batch(h, ENERGIES, **kwargs)
    assert calls == [len(ENERGIES)]


def test_the_tetrahedra_refuse_the_real_axis():
    """On the real axis the real part of the simplex Green's function has
    logarithmic divergences. The message names the energy and the delta
    that were passed, not their sum. An imaginary part of the energy that
    cancels delta up to the rounding is the real axis too, and a small
    broadening that was asked for is not"""
    h = _diamond()
    with pytest.raises(ValueError, match="off the real axis") as info:
        bloch_selfenergy(h, energy=0.3-0.05j, delta=0.05, mode="triangle",
                         nk=4)
    assert "0.3-0.05j" in str(info.value) and "delta=0.05" in str(info.value)
    with pytest.raises(ValueError, match="off the real axis"):
        bloch_selfenergy_batch(h, ENERGIES, delta=0., mode="triangle", nk=4)
    assert abs(-0.3+(0.1+0.2)) > 0. # 5.6e-17, what the rounding leaves
    with pytest.raises(ValueError, match="off the real axis"):
        bloch_selfenergy(h, energy=0.3-0.3j, delta=0.1+0.2, mode="triangle",
                         nk=4)
    g = bloch_selfenergy(h, energy=0.3-0.3j, delta=0.3+1e-9, mode="triangle",
                         nk=4)[0]
    g0 = bloch_selfenergy(h, energy=0.3, delta=(0.3+1e-9)-0.3,
                          mode="triangle", nk=4)[0]
    assert np.all(np.isfinite(g)) and np.max(np.abs(g-g0)) == 0.


@pytest.mark.parametrize("mode", ["adaptive", "renormalization", "full",
                                  "full_adaptive"])
def test_the_modes_that_stop_in_3d_name_the_one_that_works(mode):
    """In 3d the tetrahedra are the only Bloch selfenergy. The other four
    modes stop there, and three of them did not say which mode to use;
    "renormalization" is the default of an Embedding, so its message also
    says how an Embedding takes the mode"""
    h = _diamond()
    with pytest.raises(NotImplementedError, match="mode='triangle'") as info:
        bloch_selfenergy(h, energy=0.3, delta=0.05, mode=mode, nk=4)
    assert "gf_mode='triangle'" in str(info.value)


def test_the_decimation_batch_keeps_a_complex_energy():
    """The batch of a 1d chain by decimation takes real energies, and a
    complex array was cast to them, dropping the imaginary part with only
    a ComplexWarning. Complex energies go through the loop of the single
    calls instead"""
    h = geometry.chain().get_hamiltonian()
    h.shift_fermi(1.); h.add_swave(0.1)
    kwargs = dict(delta=1e-2, mode="adaptive", gtype="bulk")
    es = np.array([-0.4+0.3j, 0.13+0.3j])
    gr, sr = _scalar(h, es, **kwargs)
    g0 = _scalar(h, es.real, **kwargs)[0]
    gb, sb = bloch_selfenergy_batch(h, es, **kwargs)
    assert np.max(np.abs(gr-g0)) > 1e-2 # the imaginary part matters
    assert np.max(np.abs(gb-gr)) == 0.
    assert np.max(np.abs(sb-sr)) == 0.
