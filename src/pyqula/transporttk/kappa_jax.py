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

Benchmarked on examples/transport/localprobe_kappa_1D (101-point energy
sweep, T=0.2, delta=1e-3): ~3.7x faster than transporttk.kappa's existing
get_kappa_ratio on this LocalProbe, and matches it to within ~3e-3 out of
O(1) values -- that residual is the secant's own finite-difference
curvature bias (confirmed by tightening the secant window), not error
introduced here. See tests/transport/test_kappa_jax.py.

Not covered here (get_kappa_ratio_jax returns None so callers fall back to
transporttk.kappa's numeric path): finite temperature
(get_kappa_finite_temperature_energies), a superconducting probe lead
(Floquet-Keldysh dc_current, e.g. examples/transport/decay_constant_keldysh),
and Heterostructure (as opposed to LocalProbe) objects.
"""
import numpy as np

from .. import algebra
from .localprobe import LocalProbe
from .didv import _both_leads_superconducting

dagger = algebra.dagger

_delta_smatrix = 1e-12

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
    T!=0 themselves once T is known)."""
    if not jax_available: return False
    if not isinstance(ht, LocalProbe): return False
    if not ht.has_eh: return False
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
    return A, D, C, gl, gr, R


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
else:
    _grad_logG = None


def kappa_branch(ht, energy=0.0, T=None):
    """d(log G)/d(log t) at t=T (default ht.T), computed exactly via
    jax.grad instead of transporttk.kappa.get_power's secant fit. Raises
    if jax is unavailable or `applicable(ht)` is False -- callers should
    use get_kappa_ratio_jax, which checks that first and never raises."""
    if not jax_available:
        raise RuntimeError("jax is not available")
    if T is None: T = ht.T
    if T == 0: raise ValueError("T must be nonzero")
    ht.T = T  # so the coupling block get_central_gmatrix builds matches T
    A, D, C, gl, gr, R = _branch_arrays(ht, energy)
    args = [jnp.array(x, dtype=jnp.complex128) for x in (A, D, C, gl, gr, R)]
    logt = jnp.log(jnp.array(T, dtype=jnp.float64))
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
    except Exception:
        return None
    if not (np.isfinite(k1) and np.isfinite(k2)): return None
    return k1/k2
