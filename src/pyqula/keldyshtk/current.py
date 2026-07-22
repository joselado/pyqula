import numpy as np
from scipy.integrate import quad

from ..algebra import dagger
from ..transporttk.smatrix import enlarge_hlist
from ..transporttk.fermidirac import fermidirac
from .floquet import floquet_hamiltonian, todense

# Floquet-Keldysh DC current between two (possibly superconducting) leads,
# following San-Jose, Cayao, Prada, Aguado, New J. Phys. 15, 075019 (2013)
# (arXiv:1301.4408), Appendix A. See also src/pyqula/keldysh.py.


def lesser_from_retarded(sigma_r, energy, temperature=0.):
    """Sigma^<(energy) = i*f(energy)*A(energy) = -f(energy)*[Sigma^r-Sigma^r^dagger],
    with f the Fermi function (equilibrium leads, chemical potential 0)."""
    f = fermidirac(np.array([energy]), temp=temperature)[0]
    return -f*(sigma_r - dagger(sigma_r))


def _prepare_system(ht, link=None):
    """Build the block-tridiagonal chain (leads' own unit cell + any
    explicit central sites) and identify the electron/hole projectors for
    the bond that carries the bias-induced AC phase."""
    if not ht.has_eh:
        raise NotImplementedError(
            "keldysh.dc_current needs a Nambu (BdG) heterostructure; "
            "call h.turn_nambu() on both leads first (even with zero pairing)")
    if not ht.block_diagonal:
        raise NotImplementedError(
            "keldysh.dc_current only supports heterostructures built in "
            "block-diagonal form (heterostructures.build with zero or two-or"
            "-more explicit central sites); a single dense central site is "
            "not yet supported")
    ht2 = enlarge_hlist(ht)
    hlist = ht2.central_intra
    nb = len(hlist)
    if nb < 2:
        raise ValueError("need at least two blocks to define a weak link")
    if link is None:
        link = (0, 1)
    i0, i1 = link
    if i1 != i0+1:
        raise ValueError("link must connect adjacent blocks")
    if i1 == nb-1:
        h_target = ht.Hr
    elif i0 == 0:
        h_target = ht.Hl
    else:
        raise NotImplementedError(
            "the AC-carrying bond must currently be adjacent to one of the "
            "two leads (link=(0,1) or link=(nb-2,nb-1))")
    proje = todense(h_target.get_operator("electron"))
    projh = todense(h_target.get_operator("hole"))
    dim = todense(hlist[0][0]).shape[0]
    if proje.shape[0] != dim:
        raise ValueError("dimension mismatch between the lead Hamiltonian "
                          "and the heterostructure's unit cell")
    return hlist, link, proje, projh, dim, nb


def _floquet_green_functions(ht, voltage, quasienergy, nmax, link, delta,
                              temperature):
    """Build the retarded and lesser Floquet Green's functions of the
    S region (leads' extra unit cell + any explicit central sites),
    together with the left lead's lesser/advanced self-energies at every
    sideband (needed by the current trace)."""
    hlist, link, proje, projh, dim, nb = _prepare_system(ht, link=link)
    ns = 2*nmax+1
    Hbig = floquet_hamiltonian(hlist, link, voltage, nmax, proje, projh)
    ntot = Hbig.shape[0]

    sigL = np.zeros((ntot, ntot), dtype=np.complex128)
    sigR = np.zeros((ntot, ntot), dtype=np.complex128)
    sigLess = np.zeros((ntot, ntot), dtype=np.complex128)
    sigL_less = {}
    sigL_a = {}
    for isb in range(ns):
        n = isb-nmax
        e = quasienergy+n*voltage
        sl = todense(ht.get_selfenergy(e, lead=0, delta=delta, pristine=True))
        sr = todense(ht.get_selfenergy(e, lead=1, delta=delta, pristine=True))
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
    return Gr, Gless, sigL_less, sigL_a, dim, ns, nb


def current_integrand(ht, voltage, quasienergy, nmax, tauz, link=None,
                       delta=1e-6, temperature=0.):
    """Integrand Re Tr{[G^r Sigma_L^< + G^< Sigma_L^a] tauz} of the paper's
    Eq. for I_dc, at a fixed quasienergy. `tauz` is the electron/hole
    grading operator matching the left lead's unit-cell dimension."""
    Gr, Gless, sigL_less, sigL_a, dim, ns, nb = _floquet_green_functions(
        ht, voltage, quasienergy, nmax, link, delta, temperature)
    total = 0.0+0.0j
    for isb in range(ns):
        n = isb-nmax
        off0 = (0*ns+isb)*dim
        Gr00 = Gr[off0:off0+dim, off0:off0+dim]
        Gless00 = Gless[off0:off0+dim, off0:off0+dim]
        M = Gr00@sigL_less[n] + Gless00@sigL_a[n]
        total += np.trace(M@tauz)
    return total.real


def _get_tauz(ht):
    proje = todense(ht.Hl.get_operator("electron"))
    projh = todense(ht.Hl.get_operator("hole"))
    return proje-projh


def dc_current(ht, voltage, nmax=6, nmax_max=40, tol=1e-3, temperature=0.,
               delta=None, link=None):
    """Time-averaged (DC) current through a two-terminal junction under a
    bias `voltage`, computed with the Floquet-Keldysh formalism of
    San-Jose, Cayao, Prada, Aguado, NJP 15, 075019 (2013). Works for any
    combination of normal/superconducting leads built with
    heterostructures.build (with an implicit or explicit weak-link bond,
    see keldyshtk.current._prepare_system).

    The number of Floquet sidebands is increased adaptively (as in the
    paper) until the result changes by less than `tol`, capped at
    `nmax_max` to guarantee termination."""
    if voltage == 0.:
        return 0.0
    if delta is None:
        delta = ht.delta
    tauz = _get_tauz(ht)

    def integral(nmax):
        f = lambda e: current_integrand(ht, voltage, e, nmax, tauz,
                                         link=link, delta=delta,
                                         temperature=temperature)
        val, _ = quad(f, 0., abs(voltage), limit=50, epsrel=1e-3)
        return val

    prev = integral(nmax)
    while nmax < nmax_max:
        nmax_new = nmax+2
        cur = integral(nmax_new)
        denom = max(abs(cur), abs(prev), 1e-12)
        if abs(cur-prev)/denom < tol:
            prev = cur
            break
        nmax = nmax_new
        prev = cur
    return prev*np.sign(voltage)


def iv_curve(ht, voltages, **kwargs):
    """Convenience wrapper: dc_current evaluated over a list of voltages."""
    return np.array([dc_current(ht, v, **kwargs) for v in voltages])
