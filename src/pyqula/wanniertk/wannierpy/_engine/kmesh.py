"""Pure-Python port of ``kmesh_get`` (src/kmesh.F90): determines the
b-vectors (nearest-neighbour k-point shells) and their finite-difference
weights needed for the discretized spread functional (Marzari & Vanderbilt,
PRB 56, 12847 (1997), Appendix B).

Unlike the Fortran routine, this takes every input as a plain argument and
returns every output as a plain value -- no ``.win`` file, no module-global
state. All array outputs use the same 1-indexed convention as the Fortran
routine (and hence ``wannierpy``'s existing ``io_helpers.read_mmn``) for
``nnlist``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NSUPCELL = 5
EPS5 = 1.0e-5
EPS6 = 1.0e-6
EPS7 = 1.0e-7
EPS8 = 1.0e-8
_ETA = 99999999.0


@dataclass
class KmeshResult:
    nntot: int
    nnlist: np.ndarray  # (num_kpts, nntot), 1-indexed k-point labels
    nncell: np.ndarray  # (3, num_kpts, nntot)
    wb: np.ndarray  # (nntot,)
    bk: np.ndarray  # (3, nntot, num_kpts)
    bka: np.ndarray  # (3, nntot // 2) -- distinct b-directions (mod inversion)
    neigh: np.ndarray  # (num_kpts, nntot // 2), 1-indexed into the nntot axis


def _sorted_supercell_translations(recip_lattice: np.ndarray):
    """Integer (l, m, n) translations of the reciprocal lattice within a
    ``(2*NSUPCELL+1)**3`` supercell, sorted by ascending distance from the
    origin -- replicates ``kmesh_supercell_sort``.

    Tie-break: ``internal_maxloc`` always extracts the *lowest* original
    enumeration index among values tied for the current maximum, filling
    the sorted array from the top (largest distance) down. Read back in
    ascending order (as the rest of kmesh_get does), a tied group therefore
    comes out in *descending* original-index order -- not ascending, as a
    plain stable sort on distance would give.
    """
    r = range(-NSUPCELL, NSUPCELL + 1)
    triples = [(l, m, n) for l in r for m in r for n in r if not (l == 0 and m == 0 and n == 0)]
    lmn = np.array([(0, 0, 0)] + triples, dtype=np.float64)
    cart = lmn @ recip_lattice
    dist = np.linalg.norm(cart, axis=1)
    order = np.lexsort((-np.arange(len(dist)), dist))
    return lmn[order], cart[order]


def _find_shells(dist_from_k0: np.ndarray, search_shells: int, kmesh_tol: float):
    """Greedy shell extraction, equivalent to kmesh_get's shell-search sweep:
    repeatedly take the smallest remaining distance beyond the last shell as
    the new shell distance, and count how many samples fall within
    ``kmesh_tol`` of it. Order-independent given well-separated shells (see
    module notes in the PR/plan) -- matches ``internal_maxloc``'s running
    min/counter sweep without needing to replicate its iteration order.
    """
    flat = dist_from_k0.ravel()
    dnn0 = 0.0
    dnn = np.empty(search_shells)
    multi = np.empty(search_shells, dtype=np.int64)
    for i in range(search_shells):
        candidates = flat[flat > dnn0 + kmesh_tol]
        if candidates.size == 0:
            dnn[i] = _ETA
            multi[i] = 0
            dnn0 = _ETA
            continue
        d1 = candidates.min()
        multi[i] = np.count_nonzero(np.abs(candidates - d1) < kmesh_tol)
        dnn[i] = d1
        dnn0 = d1
    return dnn, multi


def _shell_bvectors(vkpp_all: np.ndarray, kpt_cart: np.ndarray, nkp: int, shell_dist: float,
                     multi: int, kmesh_tol: float):
    """b-vectors (and their (translation, k-point) origin) for k-point
    ``nkp`` in a shell at distance ``shell_dist`` with multiplicity
    ``multi``, in the same (translation ascending, k-point ascending) scan
    order as the Fortran routine's early-exit search -- see module notes:
    given the periodicity of a regular mesh, an unrestricted band search
    finds exactly ``multi`` matches, so no early exit is needed for
    correctness, only for the *order* in which they're returned.
    """
    dist = np.linalg.norm(vkpp_all - kpt_cart[nkp], axis=-1)  # (Ntrans, num_kpts)
    lo, hi = shell_dist * (1.0 - kmesh_tol), shell_dist * (1.0 + kmesh_tol)
    mask = (dist >= lo) & (dist <= hi)
    matches = np.argwhere(mask)  # row-major: translation ascending, then k-point ascending
    if matches.shape[0] != multi:
        raise ValueError(
            f"kmesh_get: expected {multi} neighbours in shell at distance {shell_dist} "
            f"for k-point {nkp}, found {matches.shape[0]} (kmesh_tol may be too tight/loose "
            "or the k-mesh is not a regular periodic grid)"
        )
    return matches  # (multi, 2): columns are (translation index, k-point index)


def _solve_shell_weights(bvectors_by_shell: list[np.ndarray], singv_eps: float = EPS5):
    """Least-squares B1 weights (Mostofi et al./Marzari-Vanderbilt Eq. B1)
    for a fixed list of shells' b-vectors -- equivalent to
    ``kmesh_shell_automatic``/``kmesh_shell_fixed``'s SVD solve, using
    ``lstsq``/``svd`` directly instead of manually reassembling the
    pseudoinverse from U/S/V. ``singv_eps`` differs between the two Fortran
    callers (``eps5`` vs ``eps7``)."""
    target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    amat = np.zeros((6, len(bvectors_by_shell)))
    for s, bvecs in enumerate(bvectors_by_shell):
        amat[0, s] = np.sum(bvecs[:, 0] * bvecs[:, 0])
        amat[1, s] = np.sum(bvecs[:, 1] * bvecs[:, 1])
        amat[2, s] = np.sum(bvecs[:, 2] * bvecs[:, 2])
        amat[3, s] = np.sum(bvecs[:, 0] * bvecs[:, 1])
        amat[4, s] = np.sum(bvecs[:, 1] * bvecs[:, 2])
        amat[5, s] = np.sum(bvecs[:, 2] * bvecs[:, 0])
    singv = np.linalg.svd(amat, compute_uv=False)
    if np.any(np.abs(singv) < singv_eps):
        return None, None
    bweight, *_ = np.linalg.lstsq(amat, target, rcond=None)
    return bweight, singv


def _shells_fixed(shell_list_1indexed: list[int], dnn: np.ndarray, multi: np.ndarray, vkpp_all: np.ndarray,
                   kpt_cart: np.ndarray, kmesh_tol: float, skip_b1_tests: bool):
    """Port of ``kmesh_shell_fixed``: solve B1 weights for a user-specified,
    fixed set of shells (no incremental add/parallel-rejection like
    ``kmesh_shell_automatic``)."""
    chosen_shells = [s - 1 for s in shell_list_1indexed]  # to 0-indexed
    bvectors_by_shell = [
        np.array([vkpp_all[t, k2] - kpt_cart[0] for t, k2 in
                   _shell_bvectors(vkpp_all, kpt_cart, 0, dnn[s], multi[s], kmesh_tol)])
        for s in chosen_shells
    ]
    bweight, singv = _solve_shell_weights(bvectors_by_shell, singv_eps=EPS7)
    if bweight is None:
        raise ValueError("kmesh_get: kmesh_shell_fixed found a very small singular value in the B1 fit")
    if not skip_b1_tests and not _b1_satisfied(bvectors_by_shell, bweight, kmesh_tol):
        raise ValueError("kmesh_get: kmesh_shell_fixed: B1 condition not satisfied for the given shell_list")
    return chosen_shells, bweight


def _b1_satisfied(bvectors_by_shell: list[np.ndarray], bweight: np.ndarray, kmesh_tol: float) -> bool:
    acc = np.zeros((3, 3))
    for w, bvecs in zip(bweight, bvectors_by_shell):
        acc += w * (bvecs.T @ bvecs)
    return bool(np.all(np.abs(acc - np.eye(3)) < kmesh_tol))


def _choose_shells(dnn: np.ndarray, multi: np.ndarray, vkpp_all: np.ndarray, kpt_cart: np.ndarray,
                    search_shells: int, kmesh_tol: float):
    """Port of ``kmesh_shell_automatic``: incrementally add shells (skipping
    ones parallel to an already-chosen shell's b-vectors), refitting the B1
    weights each time, until the B1 completeness relation is satisfied."""
    shell_list: list[int] = []
    bvectors_by_shell: list[np.ndarray] = []
    bweight = None
    for shell in range(search_shells):
        if multi[shell] == 0:
            continue
        matches = _shell_bvectors(vkpp_all, kpt_cart, 0, dnn[shell], multi[shell], kmesh_tol)
        bvecs = np.array([
            vkpp_all[t, k2] - kpt_cart[0] for t, k2 in matches
        ])

        is_parallel = False
        for existing in bvectors_by_shell:
            cos = (bvecs @ existing.T) / (
                np.linalg.norm(bvecs, axis=1)[:, None] * np.linalg.norm(existing, axis=1)[None, :]
            )
            if np.any(np.abs(np.abs(cos) - 1.0) < EPS6):
                is_parallel = True
                break
        if is_parallel:
            continue

        shell_list.append(shell)
        bvectors_by_shell.append(bvecs)
        new_bweight, singv = _solve_shell_weights(bvectors_by_shell)
        if new_bweight is None:
            # SVD found a near-zero singular value: reject this shell.
            shell_list.pop()
            bvectors_by_shell.pop()
            continue

        bweight = new_bweight
        if _b1_satisfied(bvectors_by_shell, bweight, kmesh_tol):
            return shell_list, bweight

    raise ValueError(
        f"kmesh_get: unable to satisfy the B1 completeness relation within the first "
        f"{search_shells} shells (try increasing search_shells)"
    )


def kmesh_get(
    kpt_frac: np.ndarray,
    recip_lattice: np.ndarray,
    *,
    search_shells: int = 36,
    kmesh_tol: float = 1.0e-6,
    num_shells: int = 0,
    shell_list: list | None = None,
    skip_b1_tests: bool = False,
    gamma_only: bool = False,
) -> KmeshResult:
    """Determine b-vector shells/weights for a regular (Monkhorst-Pack-like)
    k-point mesh.

    Parameters
    ----------
    kpt_frac : (3, num_kpts) array
        K-points in fractional (reciprocal-lattice) coordinates.
    recip_lattice : (3, 3) array
        Reciprocal lattice vectors as rows (``2*pi/[real_lattice units]``).
    num_shells / shell_list :
        ``num_shells == 0`` selects wannier90's default automatic shell
        search (``kmesh_shell_automatic``); ``num_shells > 0`` with
        ``shell_list`` given fixes the shell set explicitly
        (``kmesh_shell_fixed``, 1-indexed raw shell numbers, i.e. the same
        numbering ``iprint>=4``'s "complete list of b-vectors" would show).
        ``devel_flag=kmesh_degen`` (b-vectors from a file) is not ported.
    """
    kpt_cart = kpt_frac.T @ recip_lattice  # (num_kpts, 3)
    num_kpts = kpt_cart.shape[0]

    lmn_sorted, trans_cart = _sorted_supercell_translations(recip_lattice)
    vkpp_all = trans_cart[:, None, :] + kpt_cart[None, :, :]  # (Ntrans, num_kpts, 3)

    dist_from_k0 = np.linalg.norm(vkpp_all - kpt_cart[0], axis=-1)  # (Ntrans, num_kpts)
    dnn, multi = _find_shells(dist_from_k0, search_shells, kmesh_tol)

    if num_shells != 0:
        if not shell_list:
            raise ValueError("kmesh_get: num_shells > 0 requires shell_list")
        chosen_shells, bweight = _shells_fixed(
            shell_list, dnn, multi, vkpp_all, kpt_cart, kmesh_tol, skip_b1_tests
        )
    else:
        chosen_shells, bweight = _choose_shells(dnn, multi, vkpp_all, kpt_cart, search_shells, kmesh_tol)

    nntot = int(sum(multi[s] for s in chosen_shells))
    nnlist = np.zeros((num_kpts, nntot), dtype=np.int64)
    nncell = np.zeros((3, num_kpts, nntot), dtype=np.int64)
    bk = np.zeros((3, nntot, num_kpts))
    wb = np.zeros(nntot)

    nnx = 0
    for s, shell in enumerate(chosen_shells):
        for _ in range(multi[shell]):
            wb[nnx] = bweight[s]
            nnx += 1
    for nkp in range(num_kpts):
        nnx = 0
        for s, shell in enumerate(chosen_shells):
            matches = _shell_bvectors(vkpp_all, kpt_cart, nkp, dnn[shell], multi[shell], kmesh_tol)
            for t, k2 in matches:
                nnlist[nkp, nnx] = k2 + 1
                nncell[:, nkp, nnx] = lmn_sorted[t]
                bk[:, nnx, nkp] = vkpp_all[t, k2] - kpt_cart[nkp]
                nnx += 1

    if not skip_b1_tests:
        acc = np.einsum("n,in,jn->ij", wb, bk[:, :, 0], bk[:, :, 0])
        if np.any(np.abs(acc - np.eye(3)) > kmesh_tol):
            raise ValueError("kmesh_get: Eq. (B1) not satisfied for the assembled neighbour list")

    nnh = nntot // 2
    bka = np.zeros((3, nnh))
    na = 0
    for nn in range(nntot):
        v = bk[:, nn, 0]
        found = any(np.sum((bka[:, i] + v) ** 2) < EPS8 for i in range(na))
        if not found:
            bka[:, na] = v
            na += 1
    if na != nnh:
        raise ValueError("kmesh_get: did not find the expected number of b-directions")

    neigh = np.zeros((num_kpts, nnh), dtype=np.int64)
    for nkp in range(num_kpts):
        for na_i in range(nnh):
            match = None
            for nn in range(nntot):
                if np.sum((bka[:, na_i] - bk[:, nn, nkp]) ** 2) < EPS8:
                    match = nn
            if match is None:
                raise ValueError("kmesh_get: failed to find neighbours for a k-point")
            neigh[nkp, na_i] = match + 1

    if gamma_only:
        if num_kpts != 1:
            raise ValueError("kmesh_get: gamma_only requires num_kpts == 1")
        new_nntot = nnh
        new_nnlist = np.zeros((1, new_nntot), dtype=np.int64)
        new_nncell = np.zeros((3, 1, new_nntot), dtype=np.int64)
        new_bk = np.zeros((3, new_nntot, 1))
        new_wb = np.zeros(new_nntot)
        na = 0
        for nn in range(nntot):
            v = bk[:, nn, 0]
            found = any(np.sum((new_bk[:, i, 0] + v) ** 2) < EPS8 for i in range(na))
            if not found:
                new_bk[:, na, 0] = v
                new_wb[na] = 2.0 * wb[nn]
                new_nnlist[0, na] = nnlist[0, nn]
                new_nncell[:, 0, na] = nncell[:, 0, nn]
                na += 1
        if na != nnh:
            raise ValueError("kmesh_get: did not find the expected number of b-vectors in gamma_only mode")
        nntot, nnlist, nncell, bk, wb = new_nntot, new_nnlist, new_nncell, new_bk, new_wb
        neigh = np.arange(1, nnh + 1).reshape(1, nnh)

    return KmeshResult(nntot=nntot, nnlist=nnlist, nncell=nncell, wb=wb, bk=bk, bka=bka, neigh=neigh)
