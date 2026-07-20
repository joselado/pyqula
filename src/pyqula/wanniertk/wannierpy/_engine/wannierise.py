"""Pure-Python port of the Wannierisation engine (src/wannierise.F90):
minimizes the gauge-dependent part of the spread functional Omega
(Marzari & Vanderbilt, PRB 56, 12847 (1997) -- "MV"; Souza, Marzari &
Vanderbilt, PRB 65, 035109 (2001) -- "SMV") over the num_wann x num_wann
unitary rotation U(k), via conjugate-gradient with a parabolic line search.

Ported: the core CG minimization, ``guiding_centres``/``wann_phases``
(branch-cut phase fixing -- without it, ``csheet``/``sheet`` stay at their
Fortran-side initial values, ``csheet = 1``/``sheet = 0``, which is exact
whenever ``guiding_centres = False``, also the Fortran default), and
selective localization with centre constraints (``slwf_num``/
``slwf_constrain``/``slwf_lambda``, "SLWF+C", Vitale et al., PRB 90, 165125
(2018)) -- see ``wann_domega``'s selective_loc branch, hand-transcribed
term-by-term from the Fortran rather than vectorized/simplified, since that
branch has several terms that algebraically cancel and re-deriving that by
hand risked introducing exactly the kind of subtle sign error this
docstring is warning a future reader about. Also ported: preconditioning
(``precond``, ``_precond_direction`` -- real-space-filtered gradient via
``ws_vectors.wigner_seitz_vectors``) and fixed (rather than line-searched)
step length (``fixed_step``). Also ported: symmetry-adapted mode
(``lsitesymmetry``, ``sym`` parameter) -- see ``_engine/sitesym.py``. Unlike
``disentangle.py``, this needs no restructuring of the main loop: the
anti-Hermitian gradient/search-direction/rotation-generator arrays
(``cdodq``, ``cdq``, and its matrix exponential) get symmetrized in place
at three points (:func:`sitesym.symmetrize_gradient` mode 1 right after
``wann_domega``, mode 2 right after the search direction is built, and
:func:`sitesym.symmetrize_rotation` on the matrix exponential before it's
applied) -- everything else (spread evaluation, guiding centres, the CG
bookkeeping, convergence check) reads/writes the full ``U``/``M`` arrays,
which stay valid across the whole Brillouin zone throughout (the
symmetrized generators are zero outside each star's representative
k-point, by construction, so the existing full-BZ dot products/sums used
for ``gcnorm1``/``doda0`` already come out correct without extra star-size
weighting -- each representative's post-symmetrization value already
carries its whole star's accumulated contribution).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import numpy as np

from . import sitesym
from .ws_vectors import wigner_seitz_vectors


@dataclass
class SpreadTerms:
    om_i: float
    om_od: float
    om_d: float
    om_tot: float
    rave: np.ndarray  # (3, num_wann)
    r2ave: np.ndarray  # (num_wann,)


EPS6 = 1.0e-6


def wann_phases(M: np.ndarray, bk: np.ndarray, bka: np.ndarray, neigh: np.ndarray, rguide: np.ndarray,
                 irguide: int):
    """Pick a consistent choice of branch cut for the spread definition
    using guiding centres (MV Sec. IV / wannierise.F90's ``wann_phases``).

    ``rguide`` is the running guiding-centre estimate, Cartesian, shape (3,
    num_wann) -- the caller seeds it with the projection centres before the
    first call. ``irguide == 0`` means "this is that first call": the fit
    below still runs (to validate/report), but per Fortran's own semantics
    (see the docstring on ``irguide``: "zero if first call"), ``rguide``
    itself is *not* overwritten on that call -- the projection-site seed is
    trusted as-is until the next call.

    Returns (csheet, sheet, rguide) -- rguide updated in place and returned
    for clarity.
    """
    num_wann = rguide.shape[1]
    nnh = bka.shape[1]
    num_kpts, nntot = neigh.shape[0], bk.shape[1]

    rguide = rguide.copy()
    for iw in range(num_wann):
        csum = np.array([np.sum(M[iw, iw, neigh[:, na] - 1, np.arange(num_kpts)]) for na in range(nnh)])

        xx = np.zeros(nnh)
        smat = np.zeros((3, 3))
        svec = np.zeros(3)
        for na in range(nnh):
            if na < 3:
                xx[na] = -np.imag(np.log(csum[na]))
            else:
                xx0 = float(bka[:, na] @ rguide[:, iw])
                csumt = np.exp(1j * xx0)
                xx[na] = xx0 - np.imag(np.log(csum[na] * csumt))
            smat += np.outer(bka[:, na], bka[:, na])
            svec += bka[:, na] * xx[na]
            if na >= 2:
                det = np.linalg.det(smat)
                if abs(det) > EPS6 and irguide != 0:
                    rguide[:, iw] = np.linalg.solve(smat, svec)

    sheet = np.einsum("ibk,iw->wbk", bk, rguide)  # (num_wann, nntot, num_kpts)
    csheet = np.exp(1j * sheet)
    return csheet, sheet, rguide


def _ln_tmp(csheet: np.ndarray, sheet: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Im[log(csheet * M_nn(k,b))] - sheet, for each (n, nn, k)."""
    num_wann, _, nntot, num_kpts = M.shape
    diag = np.einsum("nnbk->nbk", M)  # M[n, n, nn, k] -> (num_wann, nntot, num_kpts)
    return np.imag(np.log(csheet * diag)) - sheet


def wann_omega(csheet: np.ndarray, sheet: np.ndarray, M: np.ndarray, bk: np.ndarray, wb: np.ndarray,
                om_i_cached: float | None, selective_loc: bool = False, slwf_num: int | None = None,
                slwf_constrain: bool = False, lambda_loc: float = 0.0,
                ccentres_cart: np.ndarray | None = None) -> SpreadTerms:
    num_wann, _, nntot, num_kpts = M.shape
    ln_tmp = _ln_tmp(csheet, sheet, M)  # (num_wann, nntot, num_kpts)

    rave = -np.einsum("b,ibk,wbk->iw", wb, bk, ln_tmp) / num_kpts  # (3, num_wann)
    rave2 = np.sum(rave ** 2, axis=0)  # (num_wann,)

    mnn2 = np.abs(np.einsum("nnbk->nbk", M)) ** 2  # (num_wann, nntot, num_kpts)
    r2ave = np.einsum("b,nbk->n", wb, 1.0 - mnn2 + ln_tmp ** 2) / num_kpts

    if selective_loc:
        # Selective localization with optional centre constraints (Vitale et al.,
        # PRB 90, 165125 (2018), "SLWF+C") -- only wannierise.F90's wann_omega
        # selective_loc branch is ported; om_i/om_od aren't computed in this mode
        # by Fortran either (see wann_main's commented-out omega_invariant/
        # omega_tilde assignment for selective_loc), so they're left at 0 here too
        # -- only om_tot/rave/r2ave (and hence wann_centres/wann_spreads/
        # spread_total) are meaningful for this mode.
        s = slwf_num
        per_n = mnn2[:s].copy()
        if slwf_constrain:
            per_n = per_n - lambda_loc * ln_tmp[:s] ** 2
        summ = per_n.sum(axis=0)  # (nntot, num_kpts)
        om_iod = float(np.einsum("b,bk->", wb, s - summ) / num_kpts)

        brn = np.einsum("ibk,iw->wbk", bk, rave)[:s]
        om_d = float(np.einsum("b,wbk->", wb, (1.0 - lambda_loc) * (ln_tmp[:s] + brn) ** 2) / num_kpts)

        om_nu = 0.0
        if slwf_constrain:
            r0 = np.einsum("ibk,ni->nbk", bk, ccentres_cart[:s])  # bk . ccentres_cart(n)
            om_nu = float(np.einsum("b,nbk->", wb, 2.0 * ln_tmp[:s] * lambda_loc * r0) / num_kpts)
            om_nu += float(lambda_loc * np.sum(ccentres_cart[:s] ** 2))

        om_tot = om_iod + om_d + om_nu
        return SpreadTerms(0.0, 0.0, om_d, om_tot, rave, r2ave)

    abs2 = np.abs(M) ** 2
    sum_all = float(np.einsum("b,mnbk->", wb, abs2) / num_kpts)  # all (m, n) pairs
    sum_diag = float(np.einsum("b,nbk->", wb, mnn2) / num_kpts)  # m == n only
    om_od = sum_all - sum_diag

    if om_i_cached is not None:
        om_i = om_i_cached
    else:
        om_i = float(num_wann * np.sum(wb)) - sum_all

    brn = np.einsum("ibk,iw->wbk", bk, rave)  # bk . rave(n), shape (num_wann, nntot, num_kpts)
    om_d = float(np.einsum("b,wbk->", wb, (ln_tmp + brn) ** 2) / num_kpts)

    om_tot = om_i + om_d + om_od
    return SpreadTerms(om_i, om_od, om_d, om_tot, rave, r2ave)


def wann_domega(csheet: np.ndarray, sheet: np.ndarray, M: np.ndarray, bk: np.ndarray, wb: np.ndarray,
                 selective_loc: bool = False, slwf_num: int | None = None, slwf_constrain: bool = False,
                 lambda_loc: float = 0.0, ccentres_cart: np.ndarray | None = None):
    num_wann, _, nntot, num_kpts = M.shape
    ln_tmp2 = wb[None, :, None] * _ln_tmp(csheet, sheet, M)  # (num_wann, nntot, num_kpts), wb baked in

    rave = -np.einsum("ibk,wbk->iw", bk, ln_tmp2) / num_kpts  # (3, num_wann)
    rnkb = np.einsum("ibk,iw->wbk", bk, rave)  # (num_wann, nntot, num_kpts)

    diag = np.einsum("nnbk->nbk", M)  # (num_wann, nntot, num_kpts)
    crt = M / diag[None, :, :, :]  # crt[m, n, b, k] = M[m, n, b, k] / M[n, n, b, k]
    cr = M * diag.conj()[None, :, :, :]  # cr[m, n, b, k] = M[m, n, b, k] * conj(M[n, n, b, k])

    cdodq = np.zeros((num_wann, num_wann, num_kpts), dtype=complex)

    if not selective_loc:
        for nn in range(nntot):
            cr_b, crt_b = cr[:, :, nn, :], crt[:, :, nn, :]
            ln_b, rnkb_b = ln_tmp2[:, nn, :], rnkb[:, nn, :]
            cdodq += wb[nn] * 0.5 * (cr_b - np.transpose(cr_b, (1, 0, 2)).conj())
            term1 = crt_b * ln_b[None, :, :] + np.transpose(crt_b, (1, 0, 2)).conj() * ln_b[:, None, :]
            cdodq -= term1 * (-0.5j)
            term2 = crt_b * rnkb_b[None, :, :] + np.transpose(crt_b, (1, 0, 2)).conj() * rnkb_b[:, None, :]
            cdodq -= wb[nn] * term2 * (-0.5j)
    else:
        # Selective localization (SLWF+C, PRB 90, 165125): m, n < slwf_num are
        # "objective" (selectively localized) functions. Transcribed element-by-
        # element straight from wann_domega's selective_loc branch (not
        # vectorized/simplified -- some terms below algebraically cancel, e.g.
        # the two slwf_constrain ln_tmp terms in the m,n both-objective case;
        # kept anyway to stay a direct, auditable copy of the Fortran, and
        # because getting the cancellation "by hand" is exactly the kind of
        # step that's easy to get subtly wrong).
        s = slwf_num
        X = -0.5j
        r0kb = np.einsum("ibk,ni->nbk", bk, ccentres_cart[:s]) if slwf_constrain else None  # (s, nntot, num_kpts)
        for nn in range(nntot):
            wbnn = wb[nn]
            for m in range(num_wann):
                for n in range(num_wann):
                    crmn, crnm = cr[m, n, nn, :], cr[n, m, nn, :]
                    crtmn, crtnm = crt[m, n, nn, :], crt[n, m, nn, :]
                    ln_m, ln_n = ln_tmp2[m, nn, :], ln_tmp2[n, nn, :]
                    rnkb_m, rnkb_n = rnkb[m, nn, :], rnkb[n, nn, :]

                    if m < s and n < s:
                        val = wbnn * 0.5 * (crmn - crnm.conj())
                        val = val - (crtmn * ln_n + (crtnm * ln_m).conj()) * X
                        val = val - (crtmn * rnkb_n + (crtnm * rnkb_m).conj()) * X
                        if slwf_constrain:
                            r0_m, r0_n = r0kb[m, nn, :], r0kb[n, nn, :]
                            val = val + lambda_loc * (crtmn * ln_n + (crtnm * ln_m).conj()) * X
                            val = val + wbnn * lambda_loc * (crtmn * rnkb_n + (crtnm * rnkb_m).conj()) * X
                            val = val - lambda_loc * (crtmn * ln_n + crtnm.conj() * ln_m) * X
                            val = val - wbnn * lambda_loc * (r0_n * crtmn + r0_m * crtnm.conj()) * X
                    elif m < s:  # n >= s
                        val = -wbnn * 0.5 * crnm.conj() - (crtnm * (ln_m + wbnn * rnkb_m)).conj() * X
                        if slwf_constrain:
                            r0_m = r0kb[m, nn, :]
                            val = val + lambda_loc * (crtnm * (ln_m + wbnn * rnkb_m)).conj() * X
                            val = val - lambda_loc * crtnm.conj() * ln_m * X
                            val = val - wbnn * lambda_loc * r0_m * crtnm.conj() * X
                    elif n < s:  # m >= s
                        val = wbnn * crmn * 0.5 - crtmn * (ln_n + wbnn * rnkb_n) * X
                        if slwf_constrain:
                            r0_n = r0kb[n, nn, :]
                            val = val + lambda_loc * crtmn * (ln_n + wbnn * rnkb_n) * X
                            val = val - lambda_loc * crtmn * ln_n * X
                            val = val - wbnn * lambda_loc * r0_n * crtmn * X
                    else:
                        continue  # cdodq(m, n) unchanged (both non-objective)

                    cdodq[m, n, :] += val

    cdodq *= 4.0 / num_kpts
    return rave, cdodq


def _unitary_exp(cdq: np.ndarray) -> np.ndarray:
    """exp(cdq) for an anti-Hermitian cdq(num_wann, num_wann, num_kpts), via
    the Hermitian eigendecomposition of i*cdq (exact and manifestly
    unitary, matching ZHEEV-based internal_new_u_and_m)."""
    H = 1j * np.moveaxis(cdq, -1, 0)  # (num_kpts, num_wann, num_wann), Hermitian
    evals, V = np.linalg.eigh(H)
    out = (V * np.exp(-1j * evals)[:, None, :]) @ np.moveaxis(V.conj(), -1, -2)
    return np.moveaxis(out, 0, -1)


def _rotate_u_and_m(U: np.ndarray, M: np.ndarray, cdq: np.ndarray, nnlist: np.ndarray, sym=None):
    expq = _unitary_exp(cdq)
    if sym is not None:
        # cdq (hence expq) is only meaningful at each star's representative
        # k-point under lsitesymmetry (see wann_main's symmetrize_gradient
        # calls); propagate expq(Rk) = D(R,k) expq(k) D(R,k)^dagger to the
        # rest of the Brillouin zone before using it to rotate U/M below --
        # mirrors internal_new_u_and_m's sitesym_symmetrize_rotation(cdq)
        # call, applied to the same post-exponentiation array.
        expq = sitesym.symmetrize_rotation(expq, sym)
    num_kpts = U.shape[2]
    U_new = np.einsum("mik,ink->mnk", U, expq)
    M_new = np.empty_like(M)
    nntot = M.shape[2]
    for k in range(num_kpts):
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            M_new[:, :, nn, k] = expq[:, :, k].conj().T @ M[:, :, nn, k] @ expq[:, :, k2]
    return U_new, M_new


def _precond_direction(cdodq: np.ndarray, kpt_latt: np.ndarray, irvec: np.ndarray, ndegen: np.ndarray,
                        real_lattice: np.ndarray, om_tot: float, num_wann: int) -> np.ndarray:
    """Real-space-filtered preconditioned gradient (``internal_search_direction``'s
    ``precond`` branch): Fourier transform cdodq(k) -> cdodq_r(R), damp by
    ``1/(1 + |R|^2/alpha)`` (alpha ~ current spread scale), transform back.
    Uses the direct O(num_kpts * nrpts) sum (Fortran's ``optimisation < 3``
    path) rather than the GEMM/``k_to_r`` reformulation (``optimisation >=
    3``, the Fortran default) -- both compute the same Fourier transform,
    the latter is purely a performance reformulation, not a different
    result."""
    num_kpts = cdodq.shape[2]
    rdotk = 2.0 * np.pi * (kpt_latt.T @ irvec.T)  # (num_kpts, nrpts)
    phase_to_r = np.exp(-1j * rdotk)
    cdodq_r = np.einsum("mnk,kr->mnr", cdodq, phase_to_r) / num_kpts

    # rvec_cart(r) = real_lattice @ irvec(r) -- literal transcription of Fortran's
    # matmul(real_lattice, irvec); real_lattice's rows are the lattice vectors.
    rvec_cart = irvec @ real_lattice.T  # (nrpts, 3)
    alpha_precond = 10.0 * om_tot / num_wann
    filt = 1.0 / (1.0 + np.sum(rvec_cart ** 2, axis=1) / alpha_precond)
    cdodq_r = cdodq_r * filt[None, None, :]

    phase_to_k = np.exp(1j * rdotk)
    return np.einsum("mnr,kr->mnk", cdodq_r / ndegen[None, None, :], phase_to_k)


def wann_main(
    u_matrix_init: np.ndarray, M_matrix_init: np.ndarray, nnlist: np.ndarray, bk: np.ndarray, wb: np.ndarray,
    num_iter: int, num_cg_steps: int, conv_tol: float, conv_window: int, trial_step: float,
    guiding_centres: bool = False, bka: np.ndarray | None = None, neigh: np.ndarray | None = None,
    proj_site_cart: np.ndarray | None = None, num_no_guide_iter: int = 0, num_guide_cycles: int = 1,
    selective_loc: bool = False, slwf_num: int | None = None, slwf_constrain: bool = False,
    lambda_loc: float = 0.0, ccentres_cart: np.ndarray | None = None, fixed_step: float | None = None,
    precond: bool = False, kpt_latt: np.ndarray | None = None, real_lattice: np.ndarray | None = None,
    mp_grid: np.ndarray | None = None, sym=None,
):
    """``bka``/``neigh`` (from ``kmesh_get``) and ``proj_site_cart`` (the
    parsed projections' sites, converted to Cartesian) are only needed when
    ``guiding_centres=True``. ``slwf_num``/``ccentres_cart`` (Cartesian,
    shape (slwf_num, 3)) are only needed when ``selective_loc=True``.
    ``kpt_latt``/``real_lattice``/``mp_grid`` are only needed when
    ``precond=True``. ``sym`` (a ``sitesym.SymmetryData``) enables
    ``lsitesymmetry`` -- ``u_matrix_init``/``M_matrix_init`` must already be
    symmetry-consistent across each star (``overlap_project``/
    ``disentangle.dis_main`` both arrange this when given the same ``sym``)."""
    num_wann, _, num_kpts = u_matrix_init.shape
    wbtot = float(np.sum(wb))
    csheet = np.ones((num_wann, len(wb), num_kpts), dtype=complex)
    sheet = np.zeros((num_wann, len(wb), num_kpts))

    if precond:
        if kpt_latt is None or real_lattice is None or mp_grid is None:
            raise ValueError("wann_main: precond=True requires kpt_latt/real_lattice/mp_grid")
        irvec, ndegen, _ = wigner_seitz_vectors(mp_grid, real_lattice)

    U = u_matrix_init.copy()
    M = M_matrix_init.copy()

    omega = functools.partial(wann_omega, selective_loc=selective_loc, slwf_num=slwf_num,
                               slwf_constrain=slwf_constrain, lambda_loc=lambda_loc, ccentres_cart=ccentres_cart)
    domega = functools.partial(wann_domega, selective_loc=selective_loc, slwf_num=slwf_num,
                                slwf_constrain=slwf_constrain, lambda_loc=lambda_loc, ccentres_cart=ccentres_cart)

    irguide = 0
    if guiding_centres:
        if bka is None or neigh is None or proj_site_cart is None:
            raise ValueError("wann_main: guiding_centres=True requires bka/neigh/proj_site_cart")
        rguide = proj_site_cart.copy()
        if num_no_guide_iter <= 0:
            csheet, sheet, rguide = wann_phases(M, bk, bka, neigh, rguide, irguide)
            irguide = 1
    else:
        rguide = None

    om_i_cached = None
    spread = omega(csheet, sheet, M, bk, wb, om_i_cached)
    om_i_cached = spread.om_i

    history: list[float] = []
    gcnorm0 = 0.0
    cdq_prev = np.zeros((num_wann, num_wann, num_kpts), dtype=complex)
    ncg = 0
    converged = False

    for iteration in range(1, num_iter + 1):
        if guiding_centres and iteration > num_no_guide_iter and iteration % num_guide_cycles == 0:
            csheet, sheet, rguide = wann_phases(M, bk, bka, neigh, rguide, irguide)
            irguide = 1

        rave, cdodq = domega(csheet, sheet, M, bk, wb)
        if sym is not None:
            cdodq = sitesym.symmetrize_gradient(1, cdodq, sym)

        if precond:
            cdodq_precond = _precond_direction(cdodq, kpt_latt, irvec, ndegen, real_lattice,
                                                spread.om_tot, num_wann)
            gcnorm1 = float(np.sum((cdodq_precond.conj() * cdodq).real))
        else:
            gcnorm1 = float(np.sum((cdodq.conj() * cdodq).real))
        if iteration == 1 or ncg >= num_cg_steps:
            gcfac = 0.0
            ncg = 0
        elif gcnorm0 > np.finfo(float).eps:
            gcfac = gcnorm1 / gcnorm0
            if gcfac > 3.0:
                gcfac = 0.0
                ncg = 0
            else:
                ncg += 1
        else:
            gcfac = 0.0
            ncg = 0
        gcnorm0 = gcnorm1

        cdq = (cdodq_precond if precond else cdodq) + cdq_prev * gcfac

        doda0 = -float(np.sum((cdodq.conj() * cdq).real)) / (4.0 * wbtot)
        if doda0 > 0.0:
            if ncg > 0:
                cdq = cdodq.copy()
                ncg = 0
                gcfac = 0.0
                doda0 = -float(np.sum((cdodq.conj() * cdq).real)) / (4.0 * wbtot)
                if doda0 > 0.0:
                    cdq = -cdq
                    doda0 = -doda0
            else:
                cdq = -cdq
                doda0 = -doda0

        # Matches wann_main's own call order: sitesym_symmetrize_gradient(2, cdq)
        # runs *after* internal_search_direction (including its uphill-reset
        # logic above) returns, using whatever doda0 that logic already settled
        # on -- doda0 itself is never recomputed from the symmetrized cdq.
        if sym is not None:
            cdq = sitesym.symmetrize_gradient(2, cdq, sym)
        cdq_prev = cdq.copy()

        if fixed_step is not None:
            cdq_final = cdq * (fixed_step / (4.0 * wbtot))
            U_new, M_new = _rotate_u_and_m(U, M, cdq_final, nnlist, sym=sym)
            old_spread = spread
            spread = omega(csheet, sheet, M_new, bk, wb, om_i_cached)
            U, M = U_new, M_new
        else:
            U0, M0 = U, M
            cdq_trial = cdq * (trial_step / (4.0 * wbtot))
            U_trial, M_trial = _rotate_u_and_m(U0, M0, cdq_trial, nnlist, sym=sym)
            trial_spread = omega(csheet, sheet, M_trial, bk, wb, om_i_cached)

            fac = trial_spread.om_tot - spread.om_tot
            if abs(fac) > np.finfo(float).tiny:
                fac = 1.0 / fac
                shift = 1.0
            else:
                fac = 1.0e6
                shift = fac * trial_spread.om_tot - fac * spread.om_tot
            eqb = fac * doda0
            eqa = shift - eqb * trial_step
            lquad = abs(eqa / (fac * spread.om_tot)) > np.finfo(float).eps
            if lquad:
                alphamin = -0.5 * eqb / eqa * trial_step ** 2
            else:
                alphamin = trial_step
            if doda0 * alphamin > 0.0:
                lquad = False
                alphamin = trial_step

            if lquad:
                cdq_final = cdq * (alphamin / (4.0 * wbtot))
                U_new, M_new = _rotate_u_and_m(U0, M0, cdq_final, nnlist, sym=sym)
                old_spread = spread
                spread = omega(csheet, sheet, M_new, bk, wb, om_i_cached)
                U, M = U_new, M_new
            else:
                old_spread = spread
                spread = trial_spread
                U, M = U_trial, M_trial

        delta_omega = spread.om_tot - old_spread.om_tot
        history.append(delta_omega)
        if len(history) > conv_window:
            history.pop(0)
        if len(history) >= conv_window and all(abs(h) < conv_tol for h in history):
            converged = True
            break

    wann_centres = spread.rave
    wann_spreads = spread.r2ave - np.sum(spread.rave ** 2, axis=0)
    return U, wann_centres, wann_spreads, spread.om_tot, spread.om_i, spread.om_d + spread.om_od, converged
