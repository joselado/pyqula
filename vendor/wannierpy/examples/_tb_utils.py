"""Shared plumbing for the tight-binding examples in this folder: build the
``M_matrix``/``A_matrix``/``eigenvalues`` arrays :func:`wannier90.run` needs
directly from a hard-coded Bloch Hamiltonian function, instead of reading
them from ``.mmn``/``.amn``/``.eig`` files the way a real DFT interface
would.

Every example's ``H(k)`` is written in the "periodic gauge" (tight-binding
hoppings enter only via lattice-vector phase factors ``exp(i 2*pi k.R)``
with ``R`` an *integer* lattice vector -- no atomic sub-cell position phase
factors), so ``H(k + G) == H(k)`` exactly for any reciprocal lattice vector
``G``. That means evaluating ``H`` at whichever mesh point ``kmesh_get``'s
neighbour table (``nnlist``) picks out already gives the correct periodic
image -- no need for ``nncell``'s shift vectors at all, unlike when
matching neighbours in a file written by an external DFT code. The overlap
between neighbouring cell-periodic Bloch states is then just a matrix
product of eigenvector matrices, ``M(k,b) = C(k)^dagger @ C(k+b)``.

Trial projections -- why this needs more than "project onto the orbital
basis": every example here keeps ``num_wann == num_bands`` (no
disentanglement -- every orbital becomes one Wannier function). Whenever
that holds, using the orbital basis itself as the trial projection
(``g_n = orbital n``, giving ``A(k) = C(k)^dagger`` -- the "obvious"
choice) makes ``overlap_project``'s Lowdin step exactly invert the same
diagonalization ``M_matrix`` was built from: writing ``U(k) = C(k)^dagger
@ W(k)`` for whatever unitary ``W(k)`` Lowdin picks, the rotated overlap
``U(k)^dagger M(k,b) U(k+b)`` algebraically reduces to ``W(k)^dagger
W(k+b)`` (using ``C(k) C(k)^dagger = I``, exact completeness of a full,
untruncated eigenbasis) -- and if ``W(k)`` doesn't vary with k (true for
*any* fixed, k-independent trial matrix, no matter how it's chosen: fixed
in, fixed out of Lowdin too), that's identically the identity matrix for
every k, b. Omega is then exactly zero and every centre lands on the cell
origin, with no CG iteration ever doing anything -- a real mathematical
fact about "full manifold + fixed trial" (not a bug, and not fixable by
picking a "better" fixed matrix), but a poor demo.

What actually breaks it is a trial projection that varies with k in a way
that mixes orbitals -- exactly what a real, finite-width atomic orbital
gives: it overlaps *neighbouring* orbitals (not just periodic images of
itself) with a distance-dependent weight, Fourier-transforming to a
genuinely k-dependent, non-diagonal trial matrix. ``gaussian_trial_matrix``
builds this directly from real-space Gaussian trial orbitals of given
per-orbital widths, positions and centres -- using *different* widths per
trial orbital is what matters (equal widths for every orbital collapses
back to the same degenerate case here, by a similar cancellation).

Even so, keeping ``num_wann == num_bands`` (as every example here does) is
still a good demo: it just means the *converged* answer is the (correctly)
exact, zero-spread atomic basis -- what's worth showing is the CG
minimisation actually getting there from a non-trivial starting point.
:func:`initial_spread` computes the pre-``wann_main`` spread (straight out
of ``overlap_project``'s Lowdin step) so each example can report it
alongside the converged one from :func:`wannier90.run`.

None of this is specific to wannierpy's pure-Python backend -- the same
``M_matrix``/``A_matrix``/``eigenvalues`` arrays work identically with
``backend="fortran"``.
"""
from __future__ import annotations

import itertools

import numpy as np


def monkhorst_pack(mp_grid) -> np.ndarray:
    """Uniform Gamma-centred fractional k-point mesh, shape (3, prod(mp_grid))
    -- the ``kpt_latt`` convention :func:`wannier90.setup` expects."""
    n1, n2, n3 = (int(n) for n in mp_grid)
    pts = [
        [i / n1, j / n2, k / n3]
        for i in range(n1) for j in range(n2) for k in range(n3)
    ]
    return np.array(pts, dtype=np.float64).T


def gaussian_trial_matrix(k_frac: np.ndarray, orbital_positions_frac: np.ndarray,
                           trial_positions_frac: np.ndarray, trial_widths, periodic_dims,
                           max_shell: int = 4) -> np.ndarray:
    """``T(k)[i, n] = sum_R exp(-i 2*pi k.R) * exp(-|tau_i + R - r_n|^2 / (2 sigma_n^2))``
    -- the Bloch transform of the overlap between tight-binding orbital
    ``i`` (fractional position ``tau_i``, repeated every cell ``R``) and a
    real-space Gaussian trial orbital ``n`` (fractional centre ``r_n``,
    width ``sigma_n`` lattice vectors), ``R`` ranging over integer cell
    shifts along ``periodic_dims`` out to +-``max_shell``. See module
    docstring for why this (as opposed to a simpler diagonal-only or fixed
    trial matrix) is what actually gives ``overlap_project``/``wann_main``
    real work to do.

    Distances are computed directly in fractional coordinates (i.e.
    ``sigma`` is "a fraction of the lattice vector along that axis"), fine
    for the orthogonal cells used in these examples.
    """
    tau = np.asarray(orbital_positions_frac, dtype=np.float64)  # (num_orbitals, 3)
    r = np.asarray(trial_positions_frac, dtype=np.float64)  # (num_wann, 3)
    sigma = np.asarray(trial_widths, dtype=np.float64)  # (num_wann,)
    num_orbitals, num_wann = tau.shape[0], r.shape[0]

    T = np.zeros((num_orbitals, num_wann), dtype=complex)
    for offsets in itertools.product(range(-max_shell, max_shell + 1), repeat=len(periodic_dims)):
        R = np.zeros(3)
        for axis, shift in zip(periodic_dims, offsets):
            R[axis] = shift
        phase = np.exp(-1j * 2 * np.pi * float(k_frac @ R))
        d = (tau[:, None, :] + R[None, None, :]) - r[None, :, :]  # (num_orbitals, num_wann, 3)
        weight = np.exp(-np.sum(d ** 2, axis=-1) / (2 * sigma[None, :] ** 2))
        T += phase * weight
    return T


def build_overlaps(hamiltonian_k, num_orbitals: int, kpt_latt: np.ndarray, nnlist: np.ndarray,
                    orbital_positions_frac: np.ndarray | None = None,
                    trial_positions_frac: np.ndarray | None = None,
                    trial_widths=None, periodic_dims=None, band_indices=None,
                    trial_vectors: np.ndarray | None = None):
    """Diagonalize ``hamiltonian_k`` at every k-point and build the overlap
    data :func:`wannier90.run` needs.

    Parameters
    ----------
    hamiltonian_k : callable(k_frac) -> (num_orbitals, num_orbitals) complex ndarray
        The Bloch Hamiltonian (periodic-gauge convention, see module
        docstring), evaluated at one fractional k-point at a time -- can be
        a hard-coded model, or an external one (e.g. ``h.get_hk_gen()``
        from a `pyqula <https://github.com/joselado/pyqula>`_ Hamiltonian).
    num_orbitals : int
        Tight-binding orbitals per unit cell -- the dimension
        ``hamiltonian_k`` returns. Equal to ``num_bands``; equal to
        ``num_wann`` too (no disentanglement: every orbital becomes one
        Wannier function) unless ``band_indices`` selects a subset.
    kpt_latt : (3, num_kpts) ndarray
        Fractional k-points, exactly what was passed to ``wannier90.setup``.
    nnlist : (num_kpts, nntot) ndarray, 1-indexed
        Neighbour table from that same call's ``SetupResult``.
    orbital_positions_frac : (num_orbitals, 3) ndarray, optional
        Fractional intra-cell position of each orbital. Needed for
        ``trial_positions_frac`` (distances are measured from these), and
        also regauges the eigenvectors (``C(k) -> D(k) C(k)`` with
        ``D(k) = diag(exp(i 2*pi k . tau_m))``) so the reported Wannier
        centres are directly comparable to real atomic positions rather
        than always landing on the cell origin.
    trial_positions_frac, trial_widths : (num_wann, 3) / (num_wann,), optional
        Real-space Gaussian trial orbitals -- see
        :func:`gaussian_trial_matrix` and the module docstring. Omit all
        three ``trial_*``/``periodic_dims`` arguments for the trivial
        "trial = exact orbital/band basis" choice (identically zero spread
        whenever ``num_wann == num_bands``, see module docstring -- not
        useful except to demonstrate that fact).
    periodic_dims : sequence of int, optional
        Which axes (0, 1, 2) are actually periodic -- required together
        with the ``trial_*`` arguments. E.g. ``[0]`` for a 1D chain along
        x, ``[0, 1]`` for a 2D sheet in the xy-plane.
    band_indices : sequence of int, optional
        Pre-select a subset of the ``num_orbitals`` bands at every k-point
        (0-indexed into ``eigh``'s ascending output, e.g. ``[0]`` for just
        the lowest band) *before* building any overlaps -- the manual
        analogue of Wannier90's ``exclude_bands``, useful when you already
        know which bands you want and don't need the disentanglement
        machinery to find a subspace for you. ``num_wann`` (and
        ``M_matrix``'s/``eigenvalues``'s size) becomes ``len(band_indices)``
        instead of ``num_orbitals`` -- still ``num_wann == num_bands`` as
        far as :func:`wannier90.run` is concerned, since the excluded bands
        were never handed to it in the first place. Requires an explicit
        trial (``trial_vectors`` or ``trial_widths``): unlike the full,
        untruncated manifold (see above), a genuine subset of bands is
        *not* degenerate for a fixed trial -- ``C(k)`` is then a
        rectangular isometry, not a square unitary, so the "``W(k)``
        collapses to a constant" argument above no longer applies -- but
        there's also no single obvious default to fall back to.
    trial_vectors : (num_orbitals, num_wann) ndarray, optional
        A simple *fixed* (k-independent) trial matrix, as an alternative to
        the k-dependent Gaussian envelope (``trial_widths``) -- fine (not
        degenerate) whenever ``band_indices`` selects a genuine subset of
        ``num_orbitals`` (see above), but falls back into the same
        "identically zero spread" case as a fixed ``trial_widths``-free
        choice if ``band_indices`` is left at the full manifold.

    Returns
    -------
    M_matrix : (num_selected, num_selected, nntot, num_kpts) complex
    A_matrix : (num_selected, num_wann, num_kpts) complex
    eigenvalues : (num_selected, num_kpts) real
    """
    num_kpts = kpt_latt.shape[1]
    nntot = nnlist.shape[1]

    C_full = np.empty((num_orbitals, num_orbitals, num_kpts), dtype=complex)
    eig_full = np.empty((num_orbitals, num_kpts))
    for k in range(num_kpts):
        H = np.asarray(hamiltonian_k(kpt_latt[:, k]), dtype=complex)
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise ValueError(f"hamiltonian_k(k={kpt_latt[:, k]}) is not Hermitian")
        w, v = np.linalg.eigh(H)
        eig_full[:, k] = w
        C_full[:, :, k] = v

    if orbital_positions_frac is not None:
        tau = np.asarray(orbital_positions_frac, dtype=np.float64)  # (num_orbitals, 3)
        phase = np.exp(1j * 2 * np.pi * (tau @ kpt_latt))  # (num_orbitals, num_kpts)
        C_full *= phase[:, None, :]  # regauge row m (orbital m) by its own position phase at each k

    if band_indices is not None:
        C = C_full[:, list(band_indices), :]
        eigenvalues = eig_full[list(band_indices), :]
    else:
        C = C_full
        eigenvalues = eig_full
    num_selected = C.shape[1]

    if band_indices is not None and trial_widths is None and trial_vectors is None:
        raise ValueError(
            "build_overlaps: band_indices selects a genuine subset of bands, which has no "
            "single obvious default trial -- pass trial_vectors (fixed) or trial_widths "
            "(k-dependent Gaussian envelope) explicitly"
        )

    num_wann = num_selected if trial_widths is None else len(trial_widths)
    if trial_vectors is not None:
        num_wann = np.asarray(trial_vectors).shape[1]
    M_matrix = np.empty((num_selected, num_selected, nntot, num_kpts), dtype=complex)
    A_matrix = np.empty((num_selected, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        if trial_vectors is not None:
            A_matrix[:, :, k] = C[:, :, k].conj().T @ trial_vectors
        elif trial_widths is not None:
            A_matrix[:, :, k] = C[:, :, k].conj().T @ gaussian_trial_matrix(
                kpt_latt[:, k], orbital_positions_frac, trial_positions_frac, trial_widths, periodic_dims
            )
        else:
            A_matrix[:, :, k] = C[:, :, k].conj().T
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            M_matrix[:, :, nn, k] = C[:, :, k].conj().T @ C[:, :, k2]

    return M_matrix, A_matrix, eigenvalues


def initial_spread(A_matrix: np.ndarray, M_matrix: np.ndarray, nnlist: np.ndarray) -> float:
    """Omega_total straight out of the initial (Lowdin-orthogonalized
    trial-projection) gauge, before any CG minimisation -- see module
    docstring. Only meaningful for the no-disentanglement path
    (``num_wann == num_bands``, true for every example here)."""
    from wannier90._engine.overlap import overlap_project
    from wannier90._engine.wannierise import wann_omega

    U0, M0 = overlap_project(A_matrix, M_matrix, nnlist)
    num_wann, _, nntot, num_kpts = M0.shape
    csheet = np.ones((num_wann, nntot, num_kpts), dtype=complex)
    sheet = np.zeros((num_wann, nntot, num_kpts))
    bk_dummy = np.zeros((3, nntot, num_kpts))  # only Omega_D needs bk; report om_i/om_od only
    wb_dummy = np.ones(nntot)
    spread = wann_omega(csheet, sheet, M0, bk_dummy, wb_dummy, None)
    return spread.om_i + spread.om_od  # skip om_d -- it needs real bk/wb, irrelevant for this check


def report(run_result, label: str, initial_omega: float | None = None) -> None:
    """Pretty-print a :class:`wannier90.RunResult`, optionally alongside
    the pre-minimisation spread from :func:`initial_spread`."""
    print(f"\n=== {label}: converged Wannier functions ===")
    if initial_omega is not None:
        print(f"  Omega_total before CG minimisation (initial trial gauge) ~ {initial_omega:.6f} Ang^2")
    print(f"  Omega_total (gauge-dependent spread)     = {run_result.spread_total:.6f} Ang^2")
    print(f"  Omega_I (gauge-invariant part)            = {run_result.spread_invariant:.6f} Ang^2")
    print(f"  Omega_D + Omega_OD (minimised part)       = {run_result.spread_tilde:.6f} Ang^2")
    print("  Wannier centres (Ang) and spreads (Ang^2):")
    num_wann = run_result.wann_centres.shape[1]
    for n in range(num_wann):
        cx, cy, cz = run_result.wann_centres[:, n]
        print(f"    WF {n + 1}: ({cx:8.4f}, {cy:8.4f}, {cz:8.4f})   spread = {run_result.wann_spreads[n]:.6f}")
