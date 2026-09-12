import numpy as np
import pytest

from pyqula.greentk import rg

# The Sancho-Rubio decimation in greentk/rg.py is not unconditionally
# accurate: at an energy sitting on a level of the lead's own cell it
# starts by inverting (energy-intra) = i*delta, and for a small delta the
# whole recursion then runs on catastrophically cancelled numbers. It
# still "converges" -- the decimated couplings do fall below threshold --
# but to a wrong fixed point. The only thing that can tell a right answer
# from a wrong one here is the equation the surface Green's function is
# defined by,
#
#     g_s = (e - intra - inter g_s inter^dagger)^(-1)
#
# so every assertion below is on that residual (rg.surface_dyson_residual)
# rather than on a recorded number.
#
# A single-orbital chain is exact at every delta, which is why a suite of
# chain() fixtures cannot see any of this. What is needed is a lead with
# more than one orbital per cell and a genuinely non-Hermitian,
# non-symmetric intercell block -- the Bloch Hamiltonian stays Hermitian,
# but the decimation no longer reduces to a scalar recursion. With
# intra=0 such a lead has a surface state exactly at E=0, where the true
# g_s grows like 1/delta and the whole problem stops being resolvable in
# double precision: |g_s| ~ 5e11 has to come out of an inversion of a
# matrix built by cancelling numbers of that same size. No algorithm gets
# that right in float64, so the contract asserted here is that the
# routine either returns a g_s that satisfies its own equation or refuses
# -- and never quietly returns one that does not.

BACKENDS = [False, True]  # pure python and numba decimation


def lead(n, seed, intra="zero"):
    """Multi-orbital 1D lead. `inter` is an unconstrained random complex
    matrix, so inter != inter^dagger and inter != inter^T."""
    rng = np.random.default_rng(seed)
    inter = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    if intra == "zero":
        ons = np.zeros((n, n), dtype=np.complex128)
    else:
        a = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
        ons = a + a.conj().T
    return ons, inter


def residual(g_surf, intra, inter, energy, delta):
    n = intra.shape[0]
    e = np.identity(n, dtype=np.complex128)*(energy + 1j*delta)
    return rg.surface_dyson_residual(np.array(g_surf), intra, inter, e)


def test_the_fixtures_are_really_non_hermitian_leads():
    """Guard the premise of the whole file: a lead whose intercell block
    happened to be Hermitian or symmetric would not exercise anything."""
    for n in [2, 3, 4]:
        for seed in [0, 1, 2, 3]:
            _, inter = lead(n, seed)
            assert np.max(np.abs(inter - inter.conj().T)) > 0.1
            assert np.max(np.abs(inter - inter.T)) > 0.1


@pytest.mark.parametrize("numba", BACKENDS)
@pytest.mark.parametrize("delta", [1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("energy", [0.0, 0.7])
@pytest.mark.parametrize("kind", ["zero", "generic"])
@pytest.mark.parametrize("n,seed", [(2, 0), (2, 1), (3, 0), (3, 2), (4, 1)])
def test_surface_green_satisfies_its_own_dyson_equation(n, seed, kind,
                                                        energy, delta, numba):
    """Wherever double precision can resolve the answer at all -- which,
    measured against a 60-digit decimation, is delta >= 1e-6 even on the
    degenerate E=0/intra=0 fixtures -- the returned surface Green's
    function must satisfy the equation that defines it."""
    intra, inter = lead(n, seed, intra=kind)
    _, g_surf = rg.green_renormalization(intra, inter, energy=energy,
                                         delta=delta, numba=numba)
    assert residual(g_surf, intra, inter, energy, delta) < rg.dyson_tolerance


@pytest.mark.parametrize("numba", BACKENDS)
@pytest.mark.parametrize("delta", [1e-8, 1e-12])
@pytest.mark.parametrize("n,seed", [(2, 0), (2, 3), (3, 0), (3, 1), (3, 2),
                                    (3, 3), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_unresolvable_surface_state_is_refused_rather_than_guessed(n, seed,
                                                                   delta,
                                                                   numba):
    """These leads have a surface state exactly at E=0, where g_s ~ 1/delta
    and float64 cannot hold the cancellation. The decimation used to
    return an answer with an O(1) Dyson residual -- a silently wrong
    Green's function -- or die inside the fixed-point fallback with a bare
    numpy 'Singular matrix'. Neither is acceptable: the caller has to be
    told, with the energy and broadening that caused it."""
    intra, inter = lead(n, seed, intra="zero")
    with pytest.raises(ValueError) as info:
        rg.green_renormalization(intra, inter, energy=0.0, delta=delta,
                                 numba=numba)
    # a raw LinAlgError is a ValueError in numpy 2, and is exactly what
    # this must not be
    assert not isinstance(info.value, np.linalg.LinAlgError)
    msg = str(info.value)
    assert "delta" in msg and "residual" in msg


@pytest.mark.parametrize("n,seed", [(2, 1), (2, 2)])
def test_a_lead_without_a_surface_state_is_untouched_at_the_same_energy(n,
                                                                       seed):
    """Control: the refusal above is about the surface state, not about
    small deltas. These two leads have no state at E=0, and the very same
    call at delta=1e-12 must go through and satisfy its Dyson equation."""
    intra, inter = lead(n, seed, intra="zero")
    for numba in BACKENDS:
        _, g_surf = rg.green_renormalization(intra, inter, energy=0.0,
                                             delta=1e-12, numba=numba)
        assert residual(g_surf, intra, inter, 0.0, 1e-12) < rg.dyson_tolerance


def test_batched_decimation_obeys_the_same_contract():
    """green_renormalization_jit_batch runs the whole energy batch through
    the same validity check one energy at a time, so a batch containing a
    degenerate energy must refuse as well -- not return a wrong row."""
    intra, inter = lead(3, 0, intra="zero")
    energies = np.array([0.0, 0.7])
    with pytest.raises(ValueError) as info:
        rg.green_renormalization_jit_batch(intra, inter, energies,
                                           delta=1e-12)
    assert not isinstance(info.value, np.linalg.LinAlgError)
    # a batch of resolvable energies still works, and every row satisfies
    # the Dyson equation
    energies = np.array([0.3, 0.7, 1.1])
    _, g_surf = rg.green_renormalization_jit_batch(intra, inter, energies,
                                                   delta=1e-3)
    res = rg.surface_dyson_residual_batch(g_surf, intra, inter, energies,
                                          1e-3)
    assert np.max(res) < rg.dyson_tolerance


def test_fixed_point_fallback_never_raises_a_bare_linalg_error():
    """surface_green_dyson is the fallback _fix_green_renormalization
    reaches for when the decimation fails its own Dyson equation. It is
    allowed to come back with a bad answer -- the caller checks -- but it
    must not blow up with an opaque numpy error on the way."""
    for n, seed in [(2, 0), (2, 3), (4, 1), (4, 3)]:
        intra, inter = lead(n, seed, intra="zero")
        e = np.identity(n, dtype=np.complex128)*1j*1e-12
        g = rg.surface_green_dyson(intra, inter, e)
        assert np.all(np.isfinite(g))
