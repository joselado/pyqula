"""Exact jax.grad-based replacement for the 2-point log-log secant
transporttk.kappa.get_power uses to estimate kappa = d(log G)/d(log t),
restricted to the case where it is both cheap and provably exact: a
LocalProbe (always 1D, see LocalProbe's own docstring) at zero
temperature, whose probe lead is NOT itself superconducting (so
transporttk.didv.didv's "auto" dispatch routes through the BdG "smatrix"
formula, not the Floquet-Keldysh one).

Why this case is cheap: LocalProbe.get_central_gmatrix's only place the
probe-sample coupling `T` enters is as a scalar multiplying the coupling
block (hlist[0][1] = -P.lead.inter*P.T); neither the probe's own
selfenergy (the Sancho-Rubio iteration) nor the sample's local selfenergy
depends on T at all. So the expensive part is computed once, reusing the
library's own LocalProbe.get_selfenergy/get_central_gmatrix unchanged, and
only the cheap T-dependent tail -- a small 2N x 2N complex matrix inverse
plus a Fisher-Lee trace -- is re-expressed in jax.numpy so jax.grad can
differentiate it exactly, instead of sampling G at two nearby couplings
(the default 0.9T/1.1T window) and fitting a secant slope through them.

Because that tail is a hand-reimplementation of transporttk.smatrix's
formula (minus unitarize.check_and_fix's unitarity correction, and minus
LocalProbe's own has_spin=False/spinless-Nambu handling -- see
applicable()'s docstring), kappa_branch cross-checks its own G(T) against
one reference get_smatrix(...,check=True) call (reusing the already-solved
selfenergies) each time and raises if they disagree beyond
`_reference_rtol`; get_kappa_ratio_jax turns that into a clean fallback
rather than ever returning a silently-wrong value.

Benchmarked on examples/transport/localprobe_kappa_1D (101-point energy
sweep, T=0.2, delta=1e-3): ~1.25x faster than transporttk.kappa's existing
get_kappa_ratio on this LocalProbe (after paying for the reference
cross-check above), and matches it to within ~3e-3 out of O(1) values --
that residual is the secant's own finite-difference curvature bias
(confirmed by tightening the secant window), not error introduced here.
See tests/transport/test_kappa_jax.py.

Not covered here (get_kappa_ratio_jax returns None so callers fall back to
transporttk.kappa's numeric path): finite temperature
(get_kappa_finite_temperature_energies), a superconducting probe lead
(Floquet-Keldysh dc_current, e.g. examples/transport/decay_constant_keldysh),
a spinless-Nambu system, and Heterostructure (as opposed to LocalProbe)
objects.
"""
import numpy as np

from .. import algebra
from .localprobe import LocalProbe, delta_smatrix as _delta_smatrix
from .didv import _both_leads_superconducting

dagger = algebra.dagger

# Relative tolerance for cross-checking G(T) (from the jax tail's raw
# Fisher-Lee formula) against the reference transporttk.smatrix.get_smatrix
# result (which applies unitarize.check_and_fix when the S-matrix's own
# unitarity error exceeds its threshold, ~100*delta_smatrix -- see
# kappa_branch's docstring for why this check exists).
_reference_rtol = 1e-6

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False


def applicable(ht):
    """Whether get_kappa_jax's exact-gradient shortcut is valid for this
    branch object (as produced by transporttk.kappa.generate_HT) --
    structural checks only, not the reference coupling T (callers check
    T!=0 themselves once T is known).

    Requires has_spin=True on both the sample and the probe: the Nambu
    reordering _branch_arrays uses (sctk.reorder.block2nambu_matrix, the
    same convention ht.get_eh_sector/didv_BdG rely on) hardcodes 4
    degrees of freedom per site (2 spin x 2 nambu). A spinless-Nambu
    system (has_eh=True, has_spin=False, directly buildable via
    add_swave on a has_spin=False geometry) doesn't match that layout;
    reject it here rather than silently reordering into nonsense."""
    if not jax_available: return False
    if not isinstance(ht, LocalProbe): return False
    if not ht.has_eh: return False
    if not (getattr(ht.H, "has_spin", False) and getattr(ht.lead, "has_spin", False)):
        return False
    if _both_leads_superconducting(ht): return False
    return True


def _branch_arrays(ht, energy):
    """Numpy-side prep: selfenergies (unchanged library code, computed
    once, since they don't depend on ht.T) plus the constant matrix blocks
    the jax-side T-dependent tail needs. Assumes ht.T is already the
    reference coupling to build the coupling block at (kappa_branch sets
    it before calling this)."""
    delta = ht.delta
    if delta > _delta_smatrix: delta = _delta_smatrix
    selfl = ht.get_selfenergy(energy, delta=delta, lead=0, pristine=True)
    selfr = ht.get_selfenergy(energy, delta=delta, lead=1, pristine=True)
    T0 = ht.T
    hlist = ht.get_central_gmatrix(selfl=selfl, selfr=selfr, energy=energy)
    A = np.array(algebra.todense(hlist[0][0]))
    D = np.array(algebra.todense(hlist[1][1]))
    off0 = np.array(algebra.todense(hlist[0][1]))
    C = -off0/T0  # so that hlist[0][1](t) = -C*t reproduces off0 at t=T0
    gammal = 1j*(selfl - dagger(selfl))
    gammar = 1j*(selfr - dagger(selfr))
    gl = np.array(algebra.sqrtm(gammal))
    gr = np.array(algebra.sqrtm(gammar))
    n = A.shape[0]
    from ..sctk.reorder import block2nambu_matrix
    R = np.array(block2nambu_matrix(np.zeros((n, n))).todense())
    return A, D, C, gl, gr, R, delta, selfl, selfr


if jax_available:
    def _logG(logt, A, D, C, gl, gr, R):
        """log of didv_BdG's conductance formula (ree.shape[0]-Ree+Reh),
        restricted to the T-dependent tail -- see this module's docstring
        for why selfenergies (baked into A, D, gl, gr, all T-independent
        constants here) don't need to be inside this traced function."""
        n = A.shape[0]
        half = n//2
        iden_n = jnp.eye(n, dtype=jnp.complex128)
        t = jnp.exp(logt).astype(jnp.complex128)
        off = -C*t
        M = jnp.block([[A, off], [jnp.conj(off).T, D]])
        Minv = jnp.linalg.inv(M)
        g11 = Minv[:n, :n]
        s00 = -iden_n + 1j*gl@g11@gl  # reflection block, Fisher-Lee
        Rh = jnp.conj(R).T
        rr = R@s00@Rh  # reorder into (electron,hole) block form
        ree = rr[0:half, 0:half]
        reh = rr[0:half, half:n]
        Ree = jnp.trace(jnp.conj(ree).T@ree)
        Reh = jnp.trace(jnp.conj(reh).T@reh)
        G = (half - Ree + Reh).real
        return jnp.log(G)

    _grad_logG = jax.jit(jax.grad(_logG, argnums=0))
    _logG_jit = jax.jit(_logG)
else:
    _grad_logG = None
    _logG_jit = None


def _reference_G(ht, energy, T, delta, selfl, selfr):
    """G(T) from the library's own get_smatrix+didv_BdG formula, which
    (unlike _logG's raw Fisher-Lee tail) applies unitarize.check_and_fix
    whenever the S-matrix's own unitarity error exceeds threshold. Reuses
    the already-solved selfl/selfr via LocalProbe's selfenergy cache
    (keyed exactly as get_smatrix's own ht.get_selfenergy(energy,
    delta=delta,lead=...,pristine=True) calls would key it -- see
    LocalProbe.get_selfenergy) so this doesn't re-run the expensive
    Sancho-Rubio/sample-GF solve."""
    from .smatrix import get_smatrix
    prev_flag, prev_cache = ht.reuse_selfenergy, ht._selfenergy_cache
    ht.reuse_selfenergy = True
    ht._selfenergy_cache = {
        (energy, 0, delta, None): selfl,
        (energy, 1, delta, None): selfr,
    }
    try:
        s = get_smatrix(ht, energy=energy, check=True)
    finally:
        ht.reuse_selfenergy = prev_flag
        ht._selfenergy_cache = prev_cache
    r = ht.get_reflection_normal_lead(s)
    ree = ht.get_eh_sector(r, i=0, j=0)
    reh = ht.get_eh_sector(r, i=0, j=1)
    Ree = np.trace(dagger(ree)@ree)
    Reh = np.trace(dagger(reh)@reh)
    return (ree.shape[0] - Ree + Reh).real


def kappa_branch(ht, energy=0.0, T=None):
    """d(log G)/d(log t) at t=T (default ht.T), computed exactly via
    jax.grad instead of transporttk.kappa.get_power's secant fit. Raises
    if jax is unavailable, `applicable(ht)` is False, or the jax tail's
    raw (uncorrected) G(T) disagrees with the reference get_smatrix+
    check_and_fix result by more than `_reference_rtol` -- signalling
    that check_and_fix actually intervened, so the jax tail would be
    differentiating a formula the reference path doesn't return unmodified.
    Callers should use get_kappa_ratio_jax, which checks applicability
    first and never raises."""
    if not jax_available:
        raise RuntimeError("jax is not available")
    if T is None: T = ht.T
    if T == 0: raise ValueError("T must be nonzero")
    ht.T = T  # so the coupling block get_central_gmatrix builds matches T
    A, D, C, gl, gr, R, delta, selfl, selfr = _branch_arrays(ht, energy)
    args = [jnp.array(x, dtype=jnp.complex128) for x in (A, D, C, gl, gr, R)]
    logt = jnp.log(jnp.array(T, dtype=jnp.float64))

    G_jax = float(jnp.exp(_logG_jit(logt, *args)))
    G_ref = _reference_G(ht, energy, T, delta, selfl, selfr)
    if not np.isclose(G_jax, G_ref, rtol=_reference_rtol, atol=1e-10):
        raise RuntimeError(
            f"kappa_jax G(T)={G_jax} disagrees with reference get_smatrix "
            f"G(T)={G_ref} beyond rtol={_reference_rtol} -- likely "
            "unitarize.check_and_fix intervening on the reference path; "
            "falling back to the numeric secant.")

    return float(_grad_logG(logt, *args))


def get_kappa_ratio_jax(ht_sc, ht_normal, energy=0.0, T=1e-2, **kwargs):
    """Exact-gradient analog of transporttk.kappa.get_kappa_ratio's ks1/ks2
    for two already-built SC/normal branch objects (see
    transporttk.kappa.generate_HT). Returns None (never raises) whenever
    the fast path doesn't apply or fails to converge, so callers can
    unconditionally fall back to the numeric secant path for both
    branches on None."""
    if T == 0: return None
    if not (applicable(ht_sc) and applicable(ht_normal)): return None
    try:
        k1 = kappa_branch(ht_sc, energy=energy, T=T)
        k2 = kappa_branch(ht_normal, energy=energy, T=T)
        ratio = k1/k2
    except Exception:
        return None
    if not np.isfinite(ratio): return None
    return ratio
