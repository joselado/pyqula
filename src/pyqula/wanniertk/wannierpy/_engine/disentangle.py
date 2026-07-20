"""Pure-Python port of the disentanglement engine (src/disentangle.F90):
extracts an optimally-connected num_wann-dimensional subspace at each
k-point by minimizing Omega_I (Souza, Marzari & Vanderbilt, PRB 65, 035109
(2001) -- "SMV" in comments below), given an outer energy window and
optional frozen (inner window) states.

Mirrors the Fortran call graph in ``dis_main`` one-to-one:
``dis_windows`` -> ``dis_project`` -> (``dis_proj_froz`` if there are frozen
states) -> ``internal_slim_m`` -> ``dis_extract``.

Symmetry-adapted mode (``lsitesymmetry``, ``sym`` parameter throughout this
module) is ported too -- see ``_engine/sitesym.py`` -- but only when there
are no frozen states (Fortran itself doesn't support combining the two:
"frozen window not implemented yet in symmetry-adapted mode").

Not yet ported: the gamma-point variant (``dis_extract_gamma`` etc.),
``devel_flag`` escape hatches (``compspace``, energy-window "spheres").

All array indices here are 0-indexed Python/NumPy convention; the Fortran
source (extensively cross-referenced in comments) is 1-indexed throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import sitesym

EPS5 = 1.0e-5
EPS8 = 1.0e-8


@dataclass
class Windows:
    ndimwin: np.ndarray  # (num_kpts,) int -- # bands in outer window
    nfirstwin: np.ndarray  # (num_kpts,) int, 0-indexed -- first band of outer window
    ndimfroz: np.ndarray  # (num_kpts,) int -- # frozen (inner-window) bands
    lfrozen: np.ndarray  # (num_bands, num_kpts) bool, window-relative rows
    indxfroz: np.ndarray  # (num_bands, num_kpts) int, window-relative, first ndimfroz(k) valid
    indxnfroz: np.ndarray  # (num_bands, num_kpts) int, window-relative, first ndimwin(k)-ndimfroz(k) valid
    eigval_opt: np.ndarray  # (num_bands, num_kpts), slimmed to window (rows >= ndimwin are 0)
    linner: bool


def dis_windows(eigval: np.ndarray, num_wann: int, dis_win_min: float, dis_win_max: float,
                 frozen_states: bool, dis_froz_min: float, dis_froz_max: float) -> Windows:
    num_bands, num_kpts = eigval.shape
    ndimwin = np.empty(num_kpts, dtype=np.int64)
    nfirstwin = np.empty(num_kpts, dtype=np.int64)
    ndimfroz = np.zeros(num_kpts, dtype=np.int64)
    lfrozen = np.zeros((num_bands, num_kpts), dtype=bool)
    indxfroz = np.zeros((num_bands, num_kpts), dtype=np.int64)
    indxnfroz = np.zeros((num_bands, num_kpts), dtype=np.int64)
    eigval_opt = np.zeros((num_bands, num_kpts))

    for k in range(num_kpts):
        e = eigval[:, k]
        if e[0] > dis_win_max or e[-1] < dis_win_min:
            raise ValueError(f"dis_windows: outer energy window contains no eigenvalues at k-point {k}")
        in_win = (e >= dis_win_min) & (e <= dis_win_max)
        imin = int(np.argmax(in_win))  # first True (assumes ascending eigenvalues, matches Fortran)
        imax = int(np.nonzero(e <= dis_win_max)[0][-1])
        ndimwin[k] = imax - imin + 1
        nfirstwin[k] = imin
        if ndimwin[k] < num_wann:
            raise ValueError(f"dis_windows: energy window has fewer states than num_wann at k-point {k}")

        kifroz_min, kifroz_max = 0, -1  # 1-indexed within-window, as in Fortran; 0 means "not found yet"
        if frozen_states:
            for i in range(imin, imax + 1):
                if e[i] < dis_froz_min or e[i] > dis_froz_max:
                    if kifroz_min != 0:
                        break  # frozen states are contiguous in energy (ascending eigenvalues)
                    continue
                if kifroz_min == 0:
                    kifroz_min = kifroz_max = i - imin + 1
                else:
                    kifroz_max += 1
        nfroz = kifroz_max - kifroz_min + 1
        ndimfroz[k] = nfroz
        if nfroz > num_wann:
            raise ValueError(f"dis_windows: more frozen states than num_wann at k-point {k}")
        if nfroz > 0:
            indxfroz[:nfroz, k] = np.arange(kifroz_min, kifroz_max + 1) - 1  # 0-indexed, window-relative
            lfrozen[indxfroz[:nfroz, k], k] = True
        indxnfroz[:ndimwin[k] - nfroz, k] = np.nonzero(~lfrozen[:ndimwin[k], k])[0]

        eigval_opt[:ndimwin[k], k] = e[imin:imax + 1]

    return Windows(ndimwin, nfirstwin, ndimfroz, lfrozen, indxfroz, indxnfroz, eigval_opt,
                    linner=bool(np.any(ndimfroz > 0)))


def dis_project(A_matrix: np.ndarray, windows: Windows, num_wann: int) -> np.ndarray:
    """Initial guess for u_matrix_opt via SVD of the window-sliced
    projection overlaps (dis_project's non-square Lowdin transformation,
    SMV Sec. III.D): U = Z[:, :num_wann] @ Vh."""
    num_bands, num_kpts = A_matrix.shape[0], A_matrix.shape[2]
    u_matrix_opt = np.zeros((num_bands, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        n0, nd = windows.nfirstwin[k], windows.ndimwin[k]
        A_win = A_matrix[n0:n0 + nd, :num_wann, k]
        Z, _, Vh = np.linalg.svd(A_win, full_matrices=True)  # Z: (nd, nd), Vh: (num_wann, num_wann)
        u_matrix_opt[:nd, :, k] = Z[:, :num_wann] @ Vh

        unitarity = u_matrix_opt[:nd, :, k].conj().T @ u_matrix_opt[:nd, :, k]
        if not np.allclose(unitarity, np.eye(num_wann), atol=EPS5):
            raise ValueError(f"dis_project: initial U_opt not unitary at k-point {k}")
    return u_matrix_opt


def dis_proj_froz(u_matrix_opt: np.ndarray, windows: Windows, num_wann: int) -> np.ndarray:
    """Replace the non-frozen columns of u_matrix_opt with the leading
    eigenvectors of Q_froz P_s Q_froz (SMV Eq. 27), then set the frozen
    columns to the corresponding unit vectors. Only called when
    ``windows.linner`` (some k-point has frozen states)."""
    num_bands, num_kpts = u_matrix_opt.shape[0], u_matrix_opt.shape[2]
    out = u_matrix_opt.copy()

    for k in range(num_kpts):
        nd, nfroz = windows.ndimwin[k], windows.ndimfroz[k]
        if num_wann > nfroz:
            u = out[:nd, :, k]
            p_s = u @ u.conj().T
            not_frozen = ~windows.lfrozen[:nd, k]
            q_froz = np.diag(not_frozen.astype(complex))
            qpq = q_froz @ p_s @ q_froz
            if not np.allclose(qpq, qpq.conj().T, atol=EPS8):
                raise ValueError(f"dis_proj_froz: Q P Q not Hermitian at k-point {k}")

            w, cz = np.linalg.eigh(qpq)  # ascending eigenvalues, matches ZHPEVX 'I' interval convention
            if np.any((w < -EPS8) | (w > 1.0 + EPS8)):
                raise ValueError(f"dis_proj_froz: eigenvalues of Q P Q outside [0,1] at k-point {k}")

            need = num_wann - nfroz
            # Leading `need` eigenvectors = the `need` largest eigenvalues (ascending order -> tail).
            zero_count = int(np.sum(w[nd - need:] < EPS8))
            if zero_count == 0:
                chosen = cz[:, nd - need:]
            else:
                # Orth-fix (SMV degenerate-zero-eigenvalue edge case, on by default upstream):
                # some of the needed eigenvectors have (numerically) zero eigenvalue and may be
                # degenerate with the frozen subspace; pick by orthogonality to the frozen states
                # (u_matrix_opt's first `nfroz` columns) among the non-"good" (non-top) eigenvectors
                # instead of blindly trusting eigenvalue rank -- see disentangle.F90's ortho-fix.
                good = nd - need + zero_count  # first `good` columns (ascending) are unambiguous
                good_idx = list(range(nd - 1, good - 1, -1))
                chosen_idx = list(good_idx)
                for _ in range(zero_count):
                    for cand in range(nd - 1, -1, -1):
                        if cand in chosen_idx:
                            continue
                        overlaps = out[:nd, :nfroz, k].conj().T @ cz[:, cand]
                        if np.all(np.abs(overlaps) <= EPS8):
                            chosen_idx.append(cand)
                            break
                    else:
                        raise ValueError(f"dis_proj_froz: ortho-fix failed to find enough vectors at k-point {k}")
                # chosen_idx holds the `good` eigenvector indices then the `zero_count` fix-ups, in
                # the order they fill u_matrix_opt's columns nfroz..num_wann-1 (ascending column).
                chosen = cz[:, chosen_idx[::-1]]

            out[:nd, nfroz:num_wann, k] = chosen

        if nfroz > 0:
            out[:, :nfroz, k] = 0.0
            for i in range(nfroz):
                out[windows.indxfroz[i, k], i, k] = 1.0

    return out


def check_orthonormal(u_matrix_opt: np.ndarray, windows: Windows, num_wann: int) -> None:
    num_kpts = u_matrix_opt.shape[2]
    for k in range(num_kpts):
        nd = windows.ndimwin[k]
        u = u_matrix_opt[:nd, :, k]
        gram = u.conj().T @ u
        if not np.allclose(gram, np.eye(num_wann), atol=EPS8):
            raise ValueError(f"dis_main: trial orbitals for disentanglement are not orthonormal at k-point {k}")


def slim_m(M_matrix_orig: np.ndarray, windows: Windows, nnlist: np.ndarray) -> np.ndarray:
    """Reindex M_matrix_orig(k, b) rows/cols from absolute band index to
    window-relative index (0..ndimwin(k)-1), zero elsewhere."""
    num_bands, _, nntot, num_kpts = M_matrix_orig.shape
    out = np.zeros_like(M_matrix_orig)
    for k in range(num_kpts):
        n0, nd = windows.nfirstwin[k], windows.ndimwin[k]
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            n0_2, nd_2 = windows.nfirstwin[k2], windows.ndimwin[k2]
            out[:nd, :nd_2, nn, k] = M_matrix_orig[n0:n0 + nd, n0_2:n0_2 + nd_2, nn, k]
    return out


def _zmatrix(k: int, u_matrix_opt: np.ndarray, M_slim: np.ndarray, windows: Windows,
             nnlist: np.ndarray, wb: np.ndarray, num_wann: int) -> np.ndarray:
    """Z-matrix (SMV Eq. 21) restricted to the non-frozen subspace at k-point k."""
    ndk = windows.ndimwin[k] - windows.ndimfroz[k]
    idx = windows.indxnfroz[:ndk, k]
    nntot = M_slim.shape[2]
    z = np.zeros((ndk, ndk), dtype=complex)
    for nn in range(nntot):
        k2 = int(nnlist[k, nn]) - 1
        nd2 = windows.ndimwin[k2]
        cbw = M_slim[:windows.ndimwin[k], :nd2, nn, k] @ u_matrix_opt[:nd2, :, k2]  # (ndimwin(k), num_wann)
        block = cbw[idx, :]  # (ndk, num_wann)
        z += wb[nn] * (block @ block.conj().T)
    return z


def _womegai(u_matrix_opt: np.ndarray, M_slim: np.ndarray, windows: Windows, nnlist: np.ndarray,
             wb: np.ndarray, wbtot: float, num_wann: int) -> float:
    """Current-subspace Omega_I estimate (SMV Eq. 13), summed over the full
    k-mesh -- identical formula whether or not symmetry was used to obtain
    ``u_matrix_opt``, so shared by :func:`dis_extract` and
    :func:`_dis_extract_symmetric`."""
    num_kpts = u_matrix_opt.shape[2]
    womegai = 0.0
    for k in range(num_kpts):
        nd = windows.ndimwin[k]
        wk = 0.0
        for nn in range(len(wb)):
            k2 = int(nnlist[k, nn]) - 1
            nd2 = windows.ndimwin[k2]
            cwb = u_matrix_opt[:nd, :, k].conj().T @ M_slim[:nd, :nd2, nn, k]
            cww = cwb @ u_matrix_opt[:nd2, :, k2]
            wk += wb[nn] * np.sum(np.abs(cww) ** 2)
        womegai += num_wann * wbtot - wk
    return womegai / num_kpts


def _dis_extract_symmetric(u_matrix_opt_init: np.ndarray, M_slim: np.ndarray, windows: Windows,
                            nnlist: np.ndarray, wb: np.ndarray, wbtot: float, num_wann: int,
                            dis_num_iter: int, dis_mix_ratio: float, dis_conv_tol: float,
                            dis_conv_window: int, sym: "sitesym.SymmetryData", lwindow: np.ndarray):
    """Symmetry-adapted counterpart of :func:`dis_extract` (``lsitesymmetry``):
    only representative k-points (``sym.ir2ik``) are ever diagonalized (via
    :func:`sitesym.dis_extract_symmetry` in place of a direct ``eigh``); the
    rest of the Brillouin zone is filled in by propagating through the star
    (:func:`sitesym.symmetrize_u_matrix`). Frozen states are not supported
    in this mode (enforced by the caller, ``dis_main``), so every
    k-point's window-relative Z-matrix is full-sized (``ndimwin(k) x
    ndimwin(k)``, no frozen subtraction) -- unlike the ragged
    frozen-subtracted per-k dict :func:`dis_extract` uses, Z here is kept as
    one zero-padded ``(num_bands, num_bands, num_kpts)`` stack so it can be
    handed to :func:`sitesym.symmetrize_zmatrix` directly.

    ``sym`` must already have its ``d_matrix_band`` window-sliced (see
    :func:`sitesym.slim_d_matrix_band`, done once in ``dis_main``)."""
    num_bands, _, num_kpts = u_matrix_opt_init.shape
    u_matrix_opt = u_matrix_opt_init.copy()

    def zmatrix_stack(u):
        Z = np.zeros((num_bands, num_bands, num_kpts), dtype=complex)
        for k in range(num_kpts):
            nd = windows.ndimwin[k]
            Z[:nd, :nd, k] = _zmatrix(k, u, M_slim, windows, nnlist, wb, num_wann)
        return Z

    z_in = sitesym.symmetrize_zmatrix(zmatrix_stack(u_matrix_opt), sym, lwindow)
    history = []
    converged = False

    for iteration in range(1, dis_num_iter + 1):
        if iteration > 1:
            for ir in range(sym.nkptirr):
                ik = sym.ir2ik[ir]
                nd = windows.ndimwin[ik]
                mixed = dis_mix_ratio * z_out[:nd, :nd, ik] + (1.0 - dis_mix_ratio) * z_in[:nd, :nd, ik]
                z_in[:nd, :nd, ik] = 0.5 * (mixed + mixed.conj().T)  # enforce hermiticity, as Fortran does

        womegai1 = 0.0
        for ir in range(sym.nkptirr):
            ik = sym.ir2ik[ir]
            nd = windows.ndimwin[ik]
            ngk = int(np.count_nonzero(sym.kptsym[:, ir] == ik))
            star_size = sym.nsymmetry / ngk  # every k-point in the star contributes the same wk

            u_new, lam = sitesym.dis_extract_symmetry(
                ik, nd, z_in[:nd, :nd, ik], u_matrix_opt[:, :, ik], sym
            )
            u_matrix_opt[:, :, ik] = u_new
            wk = num_wann * wbtot - np.sum(np.real(np.diag(lam)))
            womegai1 += star_size * wk
        womegai1 /= num_kpts

        u_matrix_opt = sitesym.symmetrize_u_matrix(u_matrix_opt, sym, lwindow)

        womegai = _womegai(u_matrix_opt, M_slim, windows, nnlist, wb, wbtot, num_wann)
        delta = womegai1 / womegai - 1.0

        z_out = sitesym.symmetrize_zmatrix(zmatrix_stack(u_matrix_opt), sym, lwindow)

        history.append(delta)
        if len(history) > dis_conv_window:
            history.pop(0)
        if iteration >= dis_conv_window and all(abs(h) < dis_conv_tol for h in history):
            converged = True
            break

    return u_matrix_opt, converged


def dis_extract(u_matrix_opt_init: np.ndarray, M_slim: np.ndarray, windows: Windows, nnlist: np.ndarray,
                 wb: np.ndarray, wbtot: float, num_wann: int, dis_num_iter: int, dis_mix_ratio: float,
                 dis_conv_tol: float, dis_conv_window: int, sym=None, lwindow: np.ndarray | None = None):
    """The disentanglement iteration (SMV Eqs. 12-21): alternately
    diagonalize the Z-matrix to refine u_matrix_opt, and rebuild the
    Z-matrix from the refined subspace, mixing between iterations.

    ``sym`` (a ``sitesym.SymmetryData`` with ``d_matrix_band`` already
    window-sliced) dispatches to :func:`_dis_extract_symmetric` instead --
    kept as a separate function rather than interleaved conditionals here
    because the symmetric path needs full zero-padded Z-matrix arrays (to
    interoperate with ``sitesym``'s symmetrization routines) instead of
    this function's ragged frozen-subtracted per-k dict."""
    if sym is not None:
        return _dis_extract_symmetric(
            u_matrix_opt_init, M_slim, windows, nnlist, wb, wbtot, num_wann,
            dis_num_iter, dis_mix_ratio, dis_conv_tol, dis_conv_window, sym, lwindow,
        )

    num_bands, _, num_kpts = u_matrix_opt_init.shape
    u_matrix_opt = u_matrix_opt_init.copy()

    active = np.nonzero(num_wann > windows.ndimfroz)[0]
    z_in = {k: _zmatrix(k, u_matrix_opt, M_slim, windows, nnlist, wb, num_wann) for k in active}
    history = []
    converged = False

    for iteration in range(1, dis_num_iter + 1):
        if iteration > 1:
            for k in active:
                z_out = z_out_by_k[k]
                z_in[k] = dis_mix_ratio * z_out + (1.0 - dis_mix_ratio) * z_in[k]
                z_in[k] = 0.5 * (z_in[k] + z_in[k].conj().T)  # enforce hermiticity, as Fortran does

        wkomegai1 = np.full(num_kpts, num_wann * wbtot)
        for k in range(num_kpts):
            nfroz = windows.ndimfroz[k]
            if nfroz == 0:
                continue
            froz_idx = windows.indxfroz[:nfroz, k]
            rsum = 0.0
            for nn in range(len(wb)):
                k2 = int(nnlist[k, nn]) - 1
                nd2 = windows.ndimwin[k2]
                # U_opt(:, :nfroz, k) columns are one-hot at row indxfroz[i, k] (set by
                # dis_proj_froz and never touched again by dis_extract), so
                # U_opt(:, :nfroz, k)^H @ M reduces to selecting M's rows at froz_idx.
                m_block = M_slim[froz_idx, :nd2, nn, k]
                cww = m_block @ u_matrix_opt[:nd2, :, k2]  # (nfroz, num_wann)
                rsum += wb[nn] * np.sum(np.abs(cww) ** 2)
            wkomegai1[k] -= rsum

        for k in active:
            nd, nfroz = windows.ndimwin[k], windows.ndimfroz[k]
            ndk = nd - nfroz
            w, cz = np.linalg.eigh(z_in[k])  # ascending
            need = num_wann - nfroz
            top_w = w[ndk - need:]
            top_cz = cz[:, ndk - need:]
            wkomegai1[k] -= np.sum(top_w)
            idx = windows.indxnfroz[:ndk, k]
            u_matrix_opt[:nd, nfroz:num_wann, k] = 0.0
            u_matrix_opt[idx, nfroz:num_wann, k] = top_cz

        womegai1 = np.sum(wkomegai1) / num_kpts

        womegai = _womegai(u_matrix_opt, M_slim, windows, nnlist, wb, wbtot, num_wann)

        delta = womegai1 / womegai - 1.0

        z_out_by_k = {k: _zmatrix(k, u_matrix_opt, M_slim, windows, nnlist, wb, num_wann) for k in active}

        history.append(delta)
        if len(history) > dis_conv_window:
            history.pop(0)
        if iteration >= dis_conv_window and all(abs(h) < dis_conv_tol for h in history):
            converged = True
            break

    return u_matrix_opt, converged


def _rotate_m(u: np.ndarray, M: np.ndarray, windows: Windows, nnlist: np.ndarray, num_wann: int,
              windowed: bool) -> np.ndarray:
    """M_new(k, b) = u(k)^dagger @ M(k, b) @ u(k2). ``windowed=True`` for
    the (ndimwin -> num_wann) gauge change (u = u_matrix_opt, row dimension
    varies per k-point); ``windowed=False`` for a further
    (num_wann -> num_wann) rotation (u = u_matrix)."""
    num_kpts = M.shape[3]
    nntot = M.shape[2]
    out = np.empty((num_wann, num_wann, nntot, num_kpts), dtype=complex)
    for k in range(num_kpts):
        nd = windows.ndimwin[k] if windowed else num_wann
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            nd2 = windows.ndimwin[k2] if windowed else num_wann
            out[:, :, nn, k] = u[:nd, :, k].conj().T @ M[:nd, :nd2, nn, k] @ u[:nd2, :, k2]
    return out


def internal_find_u(u_matrix_opt: np.ndarray, A_matrix: np.ndarray, windows: Windows, num_wann: int,
                     sym=None) -> np.ndarray:
    """Initial guess for the num_wann x num_wann rotation ``u_matrix``
    (SMV Sec. III.D, square case): Lowdin-orthogonalize the overlap between
    the optimal-subspace states and the (window-sliced) trial projections.

    With ``sym`` (a ``sitesym.SymmetryData``, *not* window-sliced -- there's
    no window left once ``u_matrix`` is in the num_wann-dimensional gauge),
    only representative k-points are computed directly; the rest of the
    Brillouin zone is filled in by propagating through the star, using
    ``d_matrix_wann`` on both sides (``sitesym_replace_d_matrix_band``'s
    trick in the Fortran source -- once both bra and ket are num_wann-sized,
    the "band" and "wann" representations coincide)."""
    num_kpts = u_matrix_opt.shape[2]
    u_matrix = np.zeros((num_wann, num_wann, num_kpts), dtype=complex)
    k_range = sym.ir2ik if sym is not None else range(num_kpts)
    for k in k_range:
        nd = windows.ndimwin[k]
        caa = u_matrix_opt[:nd, :, k].conj().T @ A_matrix[:nd, :num_wann, k]
        Z, _, Vh = np.linalg.svd(caa, full_matrices=True)
        u_matrix[:, :, k] = Z @ Vh
    if sym is not None:
        sym_wann = replace(sym, d_matrix_band=sym.d_matrix_wann.copy())
        u_matrix = sitesym.symmetrize_u_matrix(u_matrix, sym_wann, lwindow=None)
    return u_matrix


def dis_main(A_matrix: np.ndarray, M_matrix_orig: np.ndarray, eigval: np.ndarray, nnlist: np.ndarray,
             wb: np.ndarray, num_wann: int, dis_win_min: float, dis_win_max: float, frozen_states: bool,
             dis_froz_min: float, dis_froz_max: float, dis_num_iter: int, dis_mix_ratio: float,
             dis_conv_tol: float, dis_conv_window: int, sym=None):
    """Full disentanglement pipeline, mirroring ``dis_main`` in
    disentangle.F90. Returns (u_matrix_opt, u_matrix, lwindow, M_matrix,
    converged) -- ``u_matrix``/``M_matrix`` are the initial-guess rotation
    and num_wann-gauge overlap matrices Wannierisation (phase 3) takes as
    its starting point.

    ``sym`` (a ``sitesym.SymmetryData``, ``d_matrix_band`` sized to the full
    band range -- window-slicing happens here) enables ``lsitesymmetry``.
    Not supported together with ``frozen_states``, matching the Fortran
    source ("frozen window not implemented yet in symmetry-adapted mode")."""
    if sym is not None and frozen_states:
        raise NotImplementedError(
            "site symmetry (lsitesymmetry) does not support frozen states "
            "(dis_froz_min/dis_froz_max) -- not implemented upstream either"
        )

    num_bands, num_kpts = eigval.shape
    wbtot = float(np.sum(wb))

    windows = dis_windows(eigval, num_wann, dis_win_min, dis_win_max, frozen_states, dis_froz_min, dis_froz_max)

    lwindow = np.zeros((num_bands, num_kpts), dtype=bool)
    for k in range(num_kpts):
        n0, nd = windows.nfirstwin[k], windows.ndimwin[k]
        lwindow[n0:n0 + nd, k] = True

    u_matrix_opt = dis_project(A_matrix, windows, num_wann)
    sym_win = None
    if sym is not None:
        sym_win = replace(sym, d_matrix_band=sitesym.slim_d_matrix_band(sym.d_matrix_band, lwindow, sym.ir2ik))
        u_matrix_opt = sitesym.symmetrize_u_matrix(u_matrix_opt, sym_win, lwindow)
    if windows.linner:
        u_matrix_opt = dis_proj_froz(u_matrix_opt, windows, num_wann)
    check_orthonormal(u_matrix_opt, windows, num_wann)

    M_slim = slim_m(M_matrix_orig, windows, nnlist)

    u_matrix_opt, converged = dis_extract(
        u_matrix_opt, M_slim, windows, nnlist, wb, wbtot, num_wann,
        dis_num_iter, dis_mix_ratio, dis_conv_tol, dis_conv_window,
        sym=sym_win, lwindow=lwindow,
    )

    # Rotate M into the num_wann-dimensional gauge defined by the optimal subspace...
    M_wann_gauge = _rotate_m(u_matrix_opt, M_slim, windows, nnlist, num_wann, windowed=True)
    # ...then seed the initial num_wann x num_wann rotation and rotate M into it too.
    u_matrix = internal_find_u(u_matrix_opt, A_matrix, windows, num_wann, sym=sym)
    M_matrix = _rotate_m(u_matrix, M_wann_gauge, windows, nnlist, num_wann, windowed=False)

    for k in range(num_kpts):
        nd = windows.ndimwin[k]
        if nd < num_bands:
            u_matrix_opt[nd:, :, k] = 0.0

    return u_matrix_opt, u_matrix, lwindow, M_matrix, converged
