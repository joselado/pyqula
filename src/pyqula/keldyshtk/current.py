import warnings

import numpy as np
from scipy.integrate import quad

from .. import algebra
from ..algebra import dagger
from ..transporttk.smatrix import enlarge_hlist
from ..transporttk.fermidirac import fermidirac
from .floquet import floquet_hamiltonian

# Floquet-Keldysh DC current between two (possibly superconducting) leads,
# following San-Jose, Cayao, Prada, Aguado, New J. Phys. 15, 075019 (2013)
# (arXiv:1301.4408), Appendix A. See also src/pyqula/keldysh.py.
#
# Only heterostructures with NO explicit central region (heterostructures.
# build(h1,h2), the two leads directly weak-linked via set_coupling) are
# supported. Testing found a confirmed, unresolved systematic error (a few
# percent, growing at low transparency) for junctions with an explicit
# central Hamiltonian (a single dense central site or several block-diagonal
# central sites) whenever that region is not structurally identical to a
# lead; _check_supported rejects that case until it's root-caused.


def lesser_from_retarded(sigma_r, energy, temperature=0.):
    """Sigma^<(energy) = i*f(energy)*A(energy) = -f(energy)*[Sigma^r-Sigma^r^dagger],
    with f the Fermi function (equilibrium leads, chemical potential 0)."""
    f = fermidirac(np.array([energy]), temp=temperature)[0]
    return -f*(sigma_r - dagger(sigma_r))


def _check_supported(ht):
    if getattr(ht, "dimensionality", 1) != 1:
        raise NotImplementedError(
            "keldysh.dc_current only supports 1D leads")
    if not ht.has_eh:
        raise NotImplementedError(
            "keldysh.dc_current needs a Nambu (BdG) heterostructure; "
            "call h.turn_nambu() on both leads first (even with zero pairing)")
    if not (ht.block_diagonal and len(ht.central_intra) == 0):
        raise NotImplementedError(
            "keldysh.dc_current only supports heterostructures with no "
            "explicit central region (heterostructures.build(h1,h2), i.e. "
            "the two leads directly weak-linked via set_coupling); a "
            "confirmed, unresolved systematic error was found during "
            "testing for junctions with an explicit central Hamiltonian, so "
            "that case is rejected until it is root-caused")


def _prepare_system(ht):
    """Build the 2-block chain (one extra unit cell of each lead, directly
    weak-linked) and the electron/hole projectors for the right lead, whose
    unit cell is the target of the AC-carrying bond."""
    _check_supported(ht)
    hlist = enlarge_hlist(ht).central_intra
    proje = algebra.todense(ht.Hr.get_operator("electron").get_matrix())
    projh = algebra.todense(ht.Hr.get_operator("hole").get_matrix())
    dim = algebra.todense(hlist[0][0]).shape[0]
    if proje.shape[0] != dim:
        raise ValueError("dimension mismatch between the lead Hamiltonian "
                          "and the heterostructure's unit cell")
    return hlist, proje, projh, dim


def _cached_selfenergy(ht, e, lead, delta, cache):
    """Static lead self-energies only depend on (lead, energy); memoize them
    since the same energies recur across sideband/quadrature/adaptive-nmax
    evaluations within a single dc_current call, and green_renormalization
    (the underlying Sancho-Rubio iteration) is not cheap. `numba=True`
    routes it through the compiled Sancho-Rubio kernel (greentk.rg.
    green_renormalization_jit) instead of the plain-Python default used
    elsewhere in the library -- this call site alone recomputes lead
    selfenergies tens of thousands of times per dc_current call, where the
    per-call Python overhead dominates; the tolerance is the same as the
    Python path (see green_renormalization_jit), so this only changes
    speed, never the result."""
    key = (lead, round(e, 10))
    out = cache.get(key)
    if out is None:
        out = algebra.todense(ht.get_selfenergy(e, lead=lead, delta=delta,
                                                 pristine=True, numba=True))
        cache[key] = out
    return out


def _floquet_green_functions(ht, voltage, quasienergy, nmax, delta,
                              temperature, cache):
    """Build the retarded and lesser Floquet Green's functions of the
    2-block S region, together with the left lead's lesser/advanced
    self-energies at every sideband (needed by the current trace)."""
    hlist, proje, projh, dim = _prepare_system(ht)
    nb = len(hlist)
    ns = 2*nmax+1
    Hbig = floquet_hamiltonian(hlist, (0, 1), voltage, nmax, proje, projh)
    ntot = Hbig.shape[0]

    sigL = np.zeros((ntot, ntot), dtype=np.complex128)
    sigR = np.zeros((ntot, ntot), dtype=np.complex128)
    sigLess = np.zeros((ntot, ntot), dtype=np.complex128)
    sigL_less = {}
    sigL_a = {}
    for isb in range(ns):
        n = isb-nmax
        e = quasienergy+n*voltage
        sl = _cached_selfenergy(ht, e, 0, delta, cache)
        sr = _cached_selfenergy(ht, e, 1, delta, cache)
        off0 = (0*ns+isb)*dim
        offR = ((nb-1)*ns+isb)*dim
        sigL[off0:off0+dim, off0:off0+dim] += sl
        sigR[offR:offR+dim, offR:offR+dim] += sr
        sl_less = lesser_from_retarded(sl, e, temperature=temperature)
        sr_less = lesser_from_retarded(sr, e, temperature=temperature)
        sigLess[off0:off0+dim, off0:off0+dim] += sl_less
        sigLess[offR:offR+dim, offR:offR+dim] += sr_less
        sigL_less[n] = sl_less
        sigL_a[n] = dagger(sl)

    iden = np.eye(ntot, dtype=np.complex128)*(quasienergy+1j*delta)
    Gr = np.linalg.inv(iden - Hbig - sigL - sigR)
    Ga = dagger(Gr)
    Gless = Gr@sigLess@Ga
    return Gr, Gless, sigL_less, sigL_a, dim, ns


def current_integrand(ht, voltage, quasienergy, nmax, tauz,
                       delta=1e-6, temperature=0., cache=None):
    """Integrand Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} of the paper's
    Eq. for I_dc, at a fixed quasienergy. `tauz` is the electron/hole
    grading operator matching the left lead's unit-cell dimension."""
    if cache is None:
        cache = {}
    Gr, Gless, sigL_less, sigL_a, dim, ns = _floquet_green_functions(
        ht, voltage, quasienergy, nmax, delta, temperature, cache)
    total = 0.0+0.0j
    for isb in range(ns):
        n = isb-nmax
        off0 = (0*ns+isb)*dim
        Gr00 = Gr[off0:off0+dim, off0:off0+dim]
        Gless00 = Gless[off0:off0+dim, off0:off0+dim]
        M = Gr00@sigL_less[n] + Gless00@sigL_a[n]
        total += np.trace(M@tauz)
    return total.real


def dc_current(ht, voltage, nmax=6, nmax_max=40, tol=1e-3, temperature=0.,
               delta=None):
    """Time-averaged (DC) current through a two-terminal junction (two
    leads, no explicit central region) under a bias `voltage`, computed
    with the Floquet-Keldysh formalism of San-Jose, Cayao, Prada, Aguado,
    NJP 15, 075019 (2013).

    The number of Floquet sidebands is increased adaptively (as in the
    paper) until the result changes by less than `tol`, capped at
    `nmax_max` to guarantee termination (a warning is issued if the cap is
    hit before convergence)."""
    if voltage == 0.:
        return 0.0
    _check_supported(ht)
    if delta is None:
        delta = ht.delta
    tauz = algebra.todense(ht.Hl.get_operator("tauz").get_matrix())
    cache = {}

    def integral(nmax):
        f = lambda e: current_integrand(ht, voltage, e, nmax, tauz,
                                         delta=delta, temperature=temperature,
                                         cache=cache)
        val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=1e-3)
        return val

    prev = integral(nmax)
    converged = False
    while nmax < nmax_max:
        nmax += 2
        cur = integral(nmax)
        denom = max(abs(cur), abs(prev), 1e-12)
        converged = abs(cur-prev)/denom < tol
        prev = cur
        if converged:
            break
    if not converged:
        warnings.warn(
            f"keldysh.dc_current: sidebands did not converge to tol={tol} "
            f"by nmax_max={nmax_max} at voltage={voltage}; result may be "
            "inaccurate, try a larger nmax_max")
    return prev


def iv_curve(ht, voltages, **kwargs):
    """Convenience wrapper: dc_current evaluated over an array of voltages,
    in parallel (see parallel.pcall)."""
    from ..parallel import pcall
    return np.array(pcall(lambda v: dc_current(ht, v, **kwargs), voltages))
