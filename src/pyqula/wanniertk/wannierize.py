"""Wannierize a subset of the bands of a pyqula Hamiltonian using
wannierpy (github.com/joselado/wannierpy)'s pure-Python Wannier90 port
(the ``wannier90`` package, ``backend="python"``).

wannierpy is not a pyqula dependency -- ``import wannier90`` is done
lazily inside :func:`get_wannier_hamiltonian`, matching the numba/jax
optional-backend pattern used elsewhere in this codebase. Install the
copy vendored in this repo with ``pip install -e vendor/wannierpy``.

Only the "fixed band subset, no disentanglement" case is implemented
(``num_wann == len(band_indices)``, matching wannierpy's own
``examples/pyqula_ladder.py`` demo): pick ``num_bands`` bands (by default
the lowest ``num_bands``, or an explicit ``band_indices``) at every
k-point on a Monkhorst-Pack mesh, Wannierize exactly that subspace, and
Fourier-transform the resulting smooth-gauge Bloch Hamiltonian back into
real-space hoppings for a new pyqula Hamiltonian. Disentanglement (a
frozen/outer energy window instead of a fixed band count) is not
implemented yet.

Superconducting (Nambu/BdG, ``h.has_eh=True``) Hamiltonians are also
supported, with electron-hole (particle-hole) symmetry *enforced* on the
result -- see ``_particle_hole_operator``/``_enforce_particle_hole_symmetry``
below. This is a pyqula-side, post-hoc symmetrization of the reconstructed
Bloch Hamiltonian, *not* routed through wannierpy's own built-in symmetry
enforcement (``lsitesymmetry``/``.dmn``, ``_engine/sitesym.py``): that
engine (R. Sakuma, PRB 87, 235109 (2013)) only ever combines eigenvectors
via unitary representation matrices ``D(R,k)`` -- no conjugation appears
anywhere in its formulas -- so it can represent point-group symmetries but
not an antiunitary one like particle-hole conjugation (``C = tau_x sigma_y
K``, which maps k -> -k *and* conjugates). Routing electron-hole symmetry
through that engine would need extending it with real antiunitary support
(a conjugation flag threaded through ``symmetrize_ukirr``/
``symmetrize_u_matrix``/etc. in the vendored ``sitesym.py``) -- a
substantial change to third-party code, left for a future iteration; the
post-hoc approach here is exact and self-contained in the meantime.

Gauge note: pyqula's own ``get_hk_gen()`` uses the "periodic gauge" --
Bloch phases enter only via integer lattice-vector directions
(``exp(i 2*pi R.k)``, see ``htk/bloch.py``), never via intra-cell atomic
positions. Wannier90's centre/spread formulas assume the opposite
convention (phases include the orbital's intra-cell position), so the
eigenvectors used to build the overlap matrices below are regauged by
``exp(i 2*pi k.tau_orbital)`` before handing them to wannier90 -- exactly
what wannierpy's own ``pyqula_ladder.py`` example does (see its
``orbital_positions_frac`` argument) -- so the reported Wannier centres
are physically meaningful (relative to the real orbital positions) rather
than always landing on the cell origin. This regauging only affects which
smooth gauge U(k) wannier90's CG search converges to and how the centres
are interpreted; the Hamiltonian reconstruction below (``H_W(k) =
U(k)^dagger @ diag(eigenvalues) @ U(k)``, Fourier-transformed with
pyqula's own bare convention) is self-consistent regardless, since U(k)
only ever acts within the abstract num_wann-dimensional band manifold,
never referencing the orbital basis again.
"""
import itertools

import numpy as np


def _import_wannier90():
    try:
        import wannier90
    except ImportError as e:
        raise ImportError(
            "get_wannier_hamiltonian requires the 'wannier90' package (wannierpy's "
            "pure-Python backend) -- not a pyqula dependency. Install the copy vendored "
            "in this repo: pip install -e vendor/wannierpy (from the pyqula repo root), "
            "or pip install wannierpy if a released version is available."
        ) from e
    return wannier90


def _mp_grid(h, nk):
    """(3,) int32 array: ``nk`` sample points along each of ``h``'s
    periodic directions, a single (Gamma) point along the rest."""
    if np.isscalar(nk):
        nk = [int(nk)] * h.dimensionality
    else:
        nk = list(nk)
    if len(nk) != h.dimensionality:
        raise ValueError(f"nk has {len(nk)} entries, expected {h.dimensionality} "
                          f"(h.dimensionality)")
    grid = [1, 1, 1]
    for i, n in enumerate(nk):
        grid[i] = int(n)
    return np.array(grid, dtype=np.int32)


def _monkhorst_pack(mp_grid):
    """Uniform Gamma-centred fractional k-mesh, shape (3, prod(mp_grid))
    -- same convention as wannierpy's own examples/_tb_utils.py."""
    n1, n2, n3 = (int(n) for n in mp_grid)
    pts = [[i / n1, j / n2, k / n3]
           for i in range(n1) for j in range(n2) for k in range(n3)]
    return np.array(pts, dtype=np.float64).T


def _real_lattice(h):
    """(3,3) array, rows = direct lattice vectors -- matches both
    pyqula's own convention (geometrytk/fractional.py) and wannierpy's
    (see io_helpers.reciprocal_lattice's docstring)."""
    return np.array([h.geometry.a1, h.geometry.a2, h.geometry.a3], dtype=np.float64)


def _orbital_positions_frac(h, num_orbitals):
    """(num_orbitals,3) fractional positions, one row per row/column of
    ``h.intra`` -- not one row per geometry site, since a spinful
    Hamiltonian has two (spin) or four (spinful Nambu/BdG) basis rows per
    site. Matches pyqula's own basis conventions: spin-doubling
    (``increase_hilbert.m2spin_sparse``) interleaves site i's up/down at
    basis indices 2*i/2*i+1; the further electron-hole doubling
    (``sctk/reorder.py::block2nambu``) regroups each site into one
    contiguous block of 4 (electron up/down, hole down/up) at indices
    4*i..4*i+3. Either way, all of a site's basis rows share that site's
    position, so site positions are simply repeated in place."""
    frac_r = np.array(h.geometry.frac_r, dtype=np.float64)
    n_sites = frac_r.shape[0]
    if num_orbitals == n_sites:
        return frac_r
    if h.has_eh and h.has_spin and num_orbitals == 4 * n_sites:
        return np.repeat(frac_r, 4, axis=0)
    if h.has_spin and not h.has_eh and num_orbitals == 2 * n_sites:
        return np.repeat(frac_r, 2, axis=0)
    raise NotImplementedError(
        "get_wannier_hamiltonian only supports spinless Hamiltonians, spinful "
        "ones, or spinful electron-hole (Nambu/BdG) ones; got "
        f"num_orbitals={num_orbitals} for {n_sites} geometry sites "
        f"(has_spin={h.has_spin}, has_eh={h.has_eh})")


def _particle_hole_operator(h, num_orbitals):
    """Unitary ``C`` (num_orbitals,num_orbitals) implementing pyqula's
    fixed, k-independent electron-hole (particle-hole) symmetry of a
    Nambu/BdG Hamiltonian: ``C @ conj(m) @ C^-1 == -m`` holds exactly for
    ``h.intra`` and every real-space hopping matrix ``m`` (this is a
    stronger, purely-local identity than the usual C H(k)* C^-1 = -H(-k)
    -- it holds matrix-by-matrix, before any Fourier sum over k, because
    pyqula's ``superconductivity.build_eh``/``turn_nambu`` build the hole
    block of *every* hopping matrix independently via the same fixed
    spin-time-reversal map, with no extra R-dependence).

    Derivation: before ``sctk.reorder.block2nambu``'s per-site
    regrouping, the natural operator swapping the electron/hole blocks
    and applying spin time-reversal (``superconductivity.time_reversal``,
    i.e. ``sigma_y`` per site) within each is ``i*tau_y (x) sigma_y``
    (``[[0, sigma_y],[-sigma_y, 0]]`` in block form); reordered into
    pyqula's actual per-site-block-of-4 basis via the same
    ``block2nambu`` transform applied to Hamiltonians, and verified
    numerically against ``h.intra``/``h.hopping`` for representative
    chains with Rashba SOC, exchange and s-wave pairing (see PR
    discussion) -- not (yet) derived from first principles inside this
    module."""
    from ..hamiltonians import sy
    from ..sctk.reorder import block2nambu
    import scipy.sparse as sp
    n_sites = len(h.geometry.frac_r)
    if num_orbitals != 4 * n_sites or not (h.has_eh and h.has_spin):
        raise NotImplementedError(
            "_particle_hole_operator only supports spinful Nambu/BdG "
            f"Hamiltonians (has_eh=True, has_spin=True); got num_orbitals="
            f"{num_orbitals} for {n_sites} sites (has_spin={h.has_spin}, "
            f"has_eh={h.has_eh})")
    isy = np.array(sp.block_diag([sy] * n_sites).todense())
    zero = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
    c_unreordered = np.block([[zero, isy], [-isy, zero]])
    return block2nambu(c_unreordered)


def _default_or_validated_eh_band_indices(num_bands, band_indices, num_orbitals):
    """For Nambu/BdG Hamiltonians, a band selection can only be made
    electron-hole symmetric if it is closed under the index pairing
    ``n -> num_orbitals-1-n`` implied by :func:`_particle_hole_operator`
    (ascending-``eigh``-sorted bands at any k: band n's exact
    particle-hole partner is band ``num_orbitals-1-n``, since the full
    spectrum at every k is its own negation under that reindexing --
    picking one forces picking the other). Without an explicit
    ``band_indices``, default to the ``num_bands`` indices centred on the
    gap (the low-energy quasiparticle/quasihole bands nearest the Fermi
    level) -- a symmetric window around ``(num_orbitals-1)/2`` is
    automatically closed under that map."""
    if band_indices is None:
        if num_bands is None:
            raise ValueError(
                "get_wannier_hamiltonian: pass num_bands or an explicit "
                "band_indices list")
        if num_bands % 2 != 0:
            raise ValueError(
                "get_wannier_hamiltonian: a Nambu/BdG Hamiltonian (has_eh=True) "
                f"needs an even num_bands to form electron-hole pairs, got {num_bands}")
        lo = (num_orbitals - num_bands) // 2
        band_indices = list(range(lo, lo + num_bands))
    else:
        band_indices = list(band_indices)
    pairs_to = lambda n: num_orbitals - 1 - n
    if sorted(pairs_to(n) for n in band_indices) != sorted(band_indices):
        raise ValueError(
            "get_wannier_hamiltonian: band_indices must be closed under the "
            f"electron-hole pairing n -> {num_orbitals - 1}-n to enforce particle-hole "
            f"symmetry (pick a band, pick its exact partner too); got {band_indices}")
    return band_indices


def _wannier_particle_hole_permutation(band_indices, num_orbitals):
    """(num_wann,num_wann) permutation matrix mapping each selected
    band's local (Wannier-basis) index to its electron-hole partner's --
    the induced action of :func:`_particle_hole_operator` restricted to
    a ``band_indices`` selection already validated as pairing-closed by
    :func:`_default_or_validated_eh_band_indices`."""
    index_of = {n: i for i, n in enumerate(band_indices)}
    num_wann = len(band_indices)
    perm = np.zeros((num_wann, num_wann), dtype=complex)
    for i, n in enumerate(band_indices):
        j = index_of[num_orbitals - 1 - n]
        perm[j, i] = 1.0
    return perm


def _default_eh_trial_vectors(num_orbitals, particle_hole_perm, particle_hole_operator):
    """Default (num_orbitals,num_wann) trial projection matrix for a
    Nambu/BdG Hamiltonian, with columns closed under electron-hole
    pairing (``trial_vectors[:,j] = particle_hole_operator @
    conj(trial_vectors[:,i])`` whenever ``particle_hole_perm[j,i]==1``)
    instead of the spinless/spinful default's arbitrary first-``num_wann``
    orbitals. Seeding the CG minimization with an already electron-hole-
    symmetric projection makes its (otherwise symmetry-unaware) converged
    gauge land closer to satisfying the pairing itself, which reduces how
    much :func:`_enforce_particle_hole_symmetry`'s post-hoc correction has
    to perturb the result away from the original selected-band spectrum."""
    num_wann = particle_hole_perm.shape[0]
    partner = np.argmax(np.abs(particle_hole_perm), axis=0)
    trial_vectors = np.zeros((num_orbitals, num_wann), dtype=complex)
    assigned = np.zeros(num_wann, dtype=bool)
    seed_orbital = 0
    for i in range(num_wann):
        if assigned[i]:
            continue
        j = int(partner[i])
        trial_vectors[seed_orbital, i] = 1.0
        trial_vectors[:, j] = particle_hole_operator @ trial_vectors[:, i].conj()
        assigned[i] = True
        assigned[j] = True
        seed_orbital += 1
    return trial_vectors


def _negative_k_index(kpt_latt, atol=1e-8):
    """For each column of ``kpt_latt`` (fractional mesh k-points), the
    column index of ``-k`` (mod 1) on the same mesh -- a Gamma-centred
    Monkhorst-Pack mesh is always closed under negation, which the
    electron-hole symmetrization below relies on."""
    num_kpts = kpt_latt.shape[1]
    neg = np.mod(-kpt_latt, 1.0)
    idx = np.empty(num_kpts, dtype=int)
    for k in range(num_kpts):
        d2 = np.sum(np.mod(np.abs(kpt_latt - neg[:, k:k + 1]), 1.0) ** 2, axis=0)
        j = int(np.argmin(d2))
        if d2[j] > atol:
            raise ValueError("get_wannier_hamiltonian: k-mesh is not closed under k -> -k, "
                              "needed to enforce electron-hole symmetry")
        idx[k] = j
    return idx


def _enforce_particle_hole_symmetry(H_k_mesh, kpt_latt, C_wan):
    """Symmetrize a Wannier-gauge Bloch Hamiltonian mesh so the
    reconstructed real-space Hamiltonian has *exact* electron-hole
    symmetry: ``H_sym(k) := 1/2 [H(k) - C_wan H(-k)* C_wan^-1]`` at every
    mesh k (both terms Hermitian, so is the average); Fourier-transforming
    this satisfies ``C_wan @ conj(h_R) @ C_wan^-1 == -h_R`` for every
    real-space hopping matrix h_R, mirroring
    :func:`_particle_hole_operator`'s identity for the original ``h``.
    The unconstrained CG spread minimization run per mesh k-point has no
    knowledge of this symmetry, so ``H_k_mesh`` only satisfies it
    approximately beforehand (up to the independent gauge choice at k vs
    -k); this trades the exact-spectral-reproduction guarantee documented
    on :func:`_bloch_hamiltonian_from_gauge` for exact electron-hole
    symmetry instead -- the two coincide in the well-converged limit."""
    neg_idx = _negative_k_index(kpt_latt)
    C_inv = C_wan.conj().T
    num_kpts = H_k_mesh.shape[2]
    H_sym = np.empty_like(H_k_mesh)
    for k in range(num_kpts):
        mirrored = -C_wan @ H_k_mesh[:, :, neg_idx[k]].conj() @ C_inv
        H_sym[:, :, k] = 0.5 * (H_k_mesh[:, :, k] + mirrored)
    return H_sym


def _smooth_degenerate_gauge(C_full, eig_full, nnlist, tol=1e-5):
    """Re-rotate each mesh k-point's (near-)degenerate eigenvector
    clusters so they vary smoothly across the mesh, instead of the
    arbitrary orthonormal basis ``np.linalg.eigh`` happens to return
    within a degenerate subspace (any rotation within it is an equally
    valid set of eigenvectors, but an uncorrelated, k-by-k-independent
    choice is *not* smooth).

    Why this matters even for a *fixed* band-index selection: the
    Marzari-Vanderbilt overlap matrices ``M(k,k+b) = C(k)^dagger C(k+b)``
    (see :func:`_build_overlaps`) are gauge-covariant -- built from any
    smoothly-varying eigenbasis of the same selected subspace, they carry
    the same physical information, and the CG spread minimization is
    designed to optimize away whatever smooth gauge choice was made. But
    at an isolated band touching/crossing, an *unaligned* per-k
    diagonalization can rotate wildly between neighboring mesh points
    with no relation to each other, which the finite-difference M-matrix
    construction has no way to distinguish from a genuine, physical
    band-character discontinuity. Left uncorrected, this corrupts the CG
    optimization specifically around the touching point, showing up as a
    couple of pathologically large Wannier spreads and wild real-space
    interpolation ringing far from that point, even though the selected
    subspace itself is perfectly smooth (its total projector is
    continuous through the touching point).

    Fixed by an orthogonal-Procrustes alignment: process mesh k-points in
    breadth-first order over the ``nnlist`` neighbor graph (so every
    k-point except the root has an already-processed neighbor to align
    against, regardless of mesh dimensionality), and within each
    near-degenerate cluster (eigenvalues closer than ``tol``), rotate the
    cluster's eigenvectors by the unitary that maximizes their overlap
    with the same cluster at that neighbor -- the standard closed-form
    solution (``rot = Vh^dagger @ U^dagger`` from the overlap's SVD).
    Non-degenerate bands need no correction: a lone eigenvector's
    arbitrary overall phase doesn't create the same ill-conditioning (the
    CG's own per-k gauge freedom already absorbs it)."""
    num_orbitals, _, num_kpts = C_full.shape
    parent_of = {0: None}
    order = [0]
    qi = 0
    while qi < len(order):
        k = order[qi]; qi += 1
        for nn in range(nnlist.shape[1]):
            k2 = int(nnlist[k, nn]) - 1
            if k2 not in parent_of:
                parent_of[k2] = k
                order.append(k2)
    if len(order) != num_kpts:
        raise ValueError("get_wannier_hamiltonian: k-mesh neighbor graph (nnlist) is not "
                          "connected, cannot align degenerate gauges across it")

    C_aligned = C_full.copy()
    for k in order[1:]:
        kp = parent_of[k]
        e = eig_full[:, k]
        start = 0
        while start < num_orbitals:
            end = start + 1
            while end < num_orbitals and e[end] - e[end - 1] < tol:
                end += 1
            if end - start > 1: # a (near-)degenerate cluster [start,end)
                overlap = C_aligned[:, start:end, kp].conj().T @ C_full[:, start:end, k]
                U, _, Vh = np.linalg.svd(overlap)
                rot = Vh.conj().T @ U.conj().T
                C_aligned[:, start:end, k] = C_full[:, start:end, k] @ rot
            start = end
    return C_aligned


def _build_overlaps(hamiltonian_k, num_orbitals, kpt_latt, nnlist,
                     orbital_positions_frac, band_indices, trial_vectors):
    """Diagonalize ``hamiltonian_k`` on the wannierization mesh and build
    the M/A/eigenvalue arrays ``wannier90.run`` needs, restricted to a
    fixed band subset with a fixed trial projection matrix -- the "no
    disentanglement, pre-selected bands" path. A pyqula-local port of the
    relevant part of wannierpy's examples/_tb_utils.py::build_overlaps
    (that module is example code, not part of wannierpy's installable
    package, so it isn't imported directly)."""
    num_kpts = kpt_latt.shape[1]
    nntot = nnlist.shape[1]
    num_selected = len(band_indices)
    num_wann = trial_vectors.shape[1]

    C_full = np.empty((num_orbitals, num_orbitals, num_kpts), dtype=complex)
    eig_full = np.empty((num_orbitals, num_kpts))
    for k in range(num_kpts):
        Hk = np.asarray(hamiltonian_k(kpt_latt[:, k]), dtype=complex)
        if not np.allclose(Hk, Hk.conj().T, atol=1e-8):
            raise ValueError(f"hamiltonian_k(k={kpt_latt[:, k]}) is not Hermitian")
        w, v = np.linalg.eigh(Hk)
        eig_full[:, k] = w
        C_full[:, :, k] = v

    C_full = _smooth_degenerate_gauge(C_full, eig_full, nnlist)

    tau = np.asarray(orbital_positions_frac, dtype=np.float64)  # (num_orbitals,3)
    phase = np.exp(1j * 2 * np.pi * (tau @ kpt_latt))  # (num_orbitals,num_kpts)
    C_full = C_full * phase[:, None, :]  # regauge each orbital row, see module docstring

    C = C_full[:, list(band_indices), :]
    eigenvalues = eig_full[list(band_indices), :]

    M_matrix = np.empty((num_selected, num_selected, nntot, num_kpts), dtype=complex)
    A_matrix = np.empty((num_selected, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        A_matrix[:, :, k] = C[:, :, k].conj().T @ trial_vectors
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            M_matrix[:, :, nn, k] = C[:, :, k].conj().T @ C[:, :, k2]

    return M_matrix, A_matrix, eigenvalues


def _bloch_hamiltonian_from_gauge(U_matrix, eigenvalues):
    """H_W(k) = U(k)^dagger @ diag(eigenvalues(k)) @ U(k) -- the
    selected-band Hamiltonian rotated into the smooth Wannier gauge.
    ``U_matrix[m,n,k]``: m = index into the selected/original band
    subspace, n = Wannier index (overlap.py's convention, confirmed by
    its own unitarity check). Exact by construction: eigenvalues of a
    unitary similarity transform are unchanged, so this reproduces the
    original selected-band spectrum exactly at every mesh k-point."""
    return np.einsum("mik,mk,mjk->ijk", U_matrix.conj(), eigenvalues, U_matrix)


def _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=1e-6):
    """Inverse-Fourier-transform a (num_wann,num_wann,num_kpts) Bloch
    Hamiltonian sampled on a Monkhorst-Pack mesh into real-space hopping
    matrices, using pyqula's own Bloch convention (H(k) = sum_R m_R
    exp(i 2*pi R.k), see htk/bloch.py's evaluate_bloch_matrix_jit) -- so
    plugging the result into set_multihopping()/get_hk_gen() exactly
    reproduces H_k_mesh at every mesh k-point (trigonometric
    interpolation elsewhere)."""
    num_kpts = H_k_mesh.shape[2]
    axis_ranges = [np.fft.fftfreq(int(n)).astype(np.float64) * int(n) for n in mp_grid]
    Rs = np.array(list(itertools.product(*axis_ranges)), dtype=np.float64)
    phase = np.exp(-1j * 2 * np.pi * (Rs @ kpt_latt))  # (num_R,num_kpts)
    HR_all = np.einsum("rk,ijk->rij", phase, H_k_mesh) / num_kpts
    hopping = {}
    for idx in range(len(Rs)):
        Rt = tuple(int(round(x)) for x in Rs[idx])
        HR = HR_all[idx]
        if Rt == (0, 0, 0) or np.max(np.abs(HR)) > cutoff:
            hopping[Rt] = HR
    return hopping


def _offmesh_validation_kfracs(mp_grid):
    """Midpoint mesh -- every wannierization mesh point offset by half a
    grid step, in every periodic direction -- exactly where trigonometric
    interpolation error from a poorly-converged CG gauge is largest.
    Used by :func:`_offmesh_reproduction_error` to pick between the joint
    and split-cluster candidates in ``get_wannier_hamiltonian`` on actual
    reconstruction accuracy, since the wannier90 spread functional (a
    real, gauge-dependent number) is not always a reliable stand-in for
    interpolation smoothness -- confirmed empirically: a joint-vs-split
    comparison based on total spread alone picked a candidate with worse
    off-mesh accuracy in at least one tested case."""
    n1, n2, n3 = (int(n) for n in mp_grid)
    pts = [[(i + 0.5) / n1, (j + 0.5) / n2, (k + 0.5) / n3]
           for i in range(n1) for j in range(n2) for k in range(n3)]
    return np.array(pts, dtype=np.float64).T


def _offmesh_reproduction_error(H_k_mesh, kpt_latt, mp_grid, hamiltonian_k,
                                 band_indices, val_kfracs, cutoff=1e-6):
    """Max eigenvalue deviation, at the :func:`_offmesh_validation_kfracs`
    points, between ``hamiltonian_k``'s true selected-band spectrum and
    the Fourier-interpolated reconstruction from ``H_k_mesh`` -- the
    direct accuracy measure used to choose between Wannierization
    candidates (see ``_offmesh_validation_kfracs``)."""
    hopping = _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=cutoff)
    num_val = val_kfracs.shape[1]
    maxerr = 0.0
    for k in range(num_val):
        kfrac = val_kfracs[:, k]
        Hk = sum(m * np.exp(1j * 2 * np.pi * np.dot(R, kfrac)) for R, m in hopping.items())
        e_w = np.sort(np.linalg.eigvalsh(Hk))
        e_full = np.sort(np.linalg.eigvalsh(hamiltonian_k(kfrac))[list(band_indices)])
        maxerr = max(maxerr, float(np.max(np.abs(e_full - e_w))))
    return maxerr


def _split_gapped_clusters(hamiltonian_k, kpt_latt, band_indices, rel_tol=0.1):
    """Partition ``band_indices`` (sorted) into maximal runs that stay
    gapped from each other across the *whole* wannierization mesh: split
    right after index i if ``min_k(eig[i+1](k) - eig[i](k))`` exceeds
    ``rel_tol`` times the total energy span covered by ``band_indices``.

    Why this matters even though it doesn't change what's mathematically
    achievable: a composite group of ``J`` isolated bands in 1D is
    *always* exactly as localizable as the direct sum of its gapped
    pieces (any complex vector bundle over a circle is trivial, so no
    topological obstruction ever blocks this) -- but wannierpy's
    unconstrained, single-shot CG spread minimization has no way to
    "know" a joint selection decomposes like this, and can get stuck far
    from that readily-achievable solution when asked to solve the joint
    problem directly (empirically: a couple of pathologically large
    Wannier spreads and wildly-oscillating off-mesh interpolation,
    confirmed to persist across thousands of extra CG iterations,
    preconditioning, and multiple random trial-vector seeds -- verified
    fixed by Wannierizing each gapped piece independently and recombining
    block-diagonally, see :func:`get_wannier_hamiltonian`'s
    ``auto_split_clusters``). In 2D/3D this in-1D-always-true triviality
    argument doesn't hold (a gapped group's topological invariant is
    additive over any smooth splitting, so a jointly-trivial selection
    can split into individually-obstructed pieces) -- see
    ``auto_split_clusters``'s docstring for why it defaults to off."""
    bi = sorted(band_indices)
    num_kpts = kpt_latt.shape[1]
    eig = np.empty((len(bi), num_kpts))
    for k in range(num_kpts):
        w = np.linalg.eigvalsh(hamiltonian_k(kpt_latt[:, k]))
        eig[:, k] = w[bi]
    span = np.max(eig) - np.min(eig)
    clusters = [[bi[0]]]
    if span > 0:
        for i in range(1, len(bi)):
            min_gap = np.min(eig[i, :] - eig[i - 1, :])
            if min_gap > rel_tol * span:
                clusters.append([])
            clusters[-1].append(bi[i])
    else:
        clusters[0] = bi
    return clusters


def _wannierize_one_group(wannier90, seedname, mp_grid, kpt_latt, real_lattice,
        atom_symbols, atoms_cart, num_orbitals, hamiltonian_k,
        orbital_positions_frac, band_indices, trial_vectors, keywords):
    """Run one full setup/overlaps/CG-run/reconstruction cycle for a
    single band group -- the inner loop body shared by the single-group
    and auto-split-into-clusters paths of :func:`get_wannier_hamiltonian`."""
    setup_result = wannier90.setup(
        seedname, mp_grid, kpt_latt, real_lattice, num_orbitals,
        atom_symbols, atoms_cart, win_keywords=keywords, backend="python",
    )
    M_matrix, A_matrix, eigenvalues = _build_overlaps(
        hamiltonian_k, num_orbitals, kpt_latt, setup_result.nnlist,
        orbital_positions_frac, band_indices, trial_vectors,
    )
    run_result = wannier90.run(
        seedname, setup_result, mp_grid, kpt_latt, real_lattice,
        atom_symbols, atoms_cart, M_matrix, A_matrix, eigenvalues, backend="python",
    )
    H_k_mesh = _bloch_hamiltonian_from_gauge(run_result.U_matrix, eigenvalues)
    return setup_result, run_result, H_k_mesh


def get_wannier_hamiltonian(h, num_bands=None, band_indices=None, nk=12,
        trial_vectors=None, num_iter=200, conv_tol=1e-10, conv_window=3,
        cutoff=1e-6, seedname="pyqula_wannier", win_keywords=None,
        auto_split_clusters=False, cluster_rel_tol=0.1):
    """Wannierize a fixed subset of ``h``'s bands and return a new pyqula
    Hamiltonian whose real-space hoppings exactly reproduce that band
    subspace on the wannierization mesh (and interpolate smoothly
    elsewhere).

    Parameters
    ----------
    h : Hamiltonian
        Must be periodic (``h.dimensionality>=1``).
    num_bands : int, optional
        Wannierize the lowest ``num_bands`` bands (0-indexed into
        ``eigh``'s ascending output) at every k-point. Required unless
        ``band_indices`` is given. For a Nambu/BdG Hamiltonian
        (``h.has_eh=True``) this instead selects the ``num_bands`` bands
        centred on the gap (the low-energy quasiparticle/quasihole
        bands nearest the Fermi level) -- see ``band_indices`` below for
        why, and note ``num_bands`` must then be even.
    band_indices : sequence of int, optional
        Explicit 0-indexed band selection, overriding ``num_bands``
        (``num_wann = len(band_indices)``). For a Nambu/BdG Hamiltonian
        this must be closed under the electron-hole pairing
        ``n -> num_orbitals-1-n`` (picking a band forces picking its
        exact particle-hole partner, see
        ``_default_or_validated_eh_band_indices``) -- otherwise a
        ``ValueError`` is raised.
    nk : int or sequence of int, optional
        k-points per periodic direction for the Monkhorst-Pack
        wannierization mesh (default 12). A sequence must have
        ``h.dimensionality`` entries.
    trial_vectors : (num_orbitals, num_wann) complex ndarray, optional
        Fixed (k-independent) trial projection matrix seeding the CG
        minimization -- default: project onto the first ``num_wann``
        orbitals (``numpy.eye(num_orbitals)[:, :num_wann]``).
    num_iter, conv_tol, conv_window : optional
        Wannier90 CG minimization parameters, passed through
        ``win_keywords``.
    cutoff : float, optional
        Real-space hopping matrices with max element below this are
        dropped (except the intracell (0,0,0) term, always kept).
    seedname : str, optional
        Passed to ``wannier90.setup``/``run`` (only used for logging by
        the pure-Python backend).
    win_keywords : dict, optional
        Extra/overriding Wannier90 ``.win`` keywords.
    auto_split_clusters : bool, optional
        Default False -- by default all of ``band_indices`` is always
        Wannierized as a single joint group, exactly as requested. If
        True and the selected bands decompose into two or more groups
        that stay gapped from each other across the whole wannierization
        mesh (see :func:`_split_gapped_clusters`), also try Wannierizing
        each group independently and recombining block-diagonally, as an
        alternative to the single joint CG run -- this works around a
        real weakness of the unconstrained single-shot spread
        minimization getting stuck in a bad local minimum (see
        ``_split_gapped_clusters``'s docstring), and the split
        alternative is only kept if it reproduces the selected spectrum
        more accurately off-mesh than the joint run (see
        ``_offmesh_reproduction_error``).

        This is **not safe for topologically nontrivial band groups**,
        and is why it defaults to off: a composite group's topological
        invariant (Chern number and the like) is additive over any
        smooth energy-gap splitting, so a *jointly trivial* selection can
        still split into pieces that are *individually* obstructed (no
        exponentially localized Wannier functions exist for them at
        all -- e.g. two energy-gapped sub-bands with opposite nonzero
        Chern number, canceling only when treated jointly). The
        off-mesh accuracy check above is an eigenvalue-reproduction
        test, not a topological one, and is not guaranteed to catch this
        (a genuine obstruction shows up reliably as Wannier spread
        diverging with mesh density -- see ``_split_gapped_clusters``'s
        PR discussion -- which is a much more expensive check than the
        sparse validation grid used here). Only turn this on for band
        groups you know decompose into topologically trivial pieces
        (guaranteed always true in 1D, not in 2D/3D).
    cluster_rel_tol : float, optional
        Default 0.1. Split threshold for ``auto_split_clusters``: a gap
        between adjacent selected bands must exceed this fraction of the
        selected bands' total energy span, at every mesh k-point, to
        count as a cluster boundary.

    Returns
    -------
    Hamiltonian
        A new, multicell pyqula Hamiltonian with ``num_wann`` orbitals
        per cell, positioned at the computed Wannier centres. Also
        carries ``wannier_band_indices``, ``wannier_clusters`` (the
        ``auto_split_clusters`` decomposition actually used -- a single
        one-element list when splitting didn't trigger), ``wannier_centres``,
        ``wannier_spreads``, ``wannier_spread_total``,
        ``wannier_setup_result``, ``wannier_run_result`` for diagnostics
        (each a single wannierpy result, or a list with one entry per
        cluster when ``auto_split_clusters`` engaged), plus
        ``wannier_particle_hole_operator`` (the unitary ``C_wan``
        with ``C_wan @ conj(h_R) @ C_wan^-1 == -h_R`` for every real-space
        hopping ``h_R`` of the result) when ``h.has_eh``. Enforcing this
        exact electron-hole symmetry means the returned Hamiltonian's
        spectrum only *approximately* (not exactly) reproduces ``h``'s
        selected bands on the wannierization mesh -- see
        ``_enforce_particle_hole_symmetry``'s docstring for the tradeoff;
        the approximation improves with better CG convergence/denser nk.
    """
    if h.dimensionality < 1:
        raise NotImplementedError(
            "get_wannier_hamiltonian needs a periodic Hamiltonian (h.dimensionality>=1)")
    wannier90 = _import_wannier90()

    num_orbitals = h.intra.shape[0]
    if h.has_eh: # Nambu/BdG: band selection must be electron-hole-pair-closed
        band_indices = _default_or_validated_eh_band_indices(num_bands, band_indices, num_orbitals)
    elif band_indices is None:
        if num_bands is None:
            raise ValueError(
                "get_wannier_hamiltonian: pass num_bands (wannierize the lowest num_bands "
                "bands) or an explicit band_indices list")
        band_indices = list(range(num_bands))
    else:
        band_indices = list(band_indices)
    num_wann = len(band_indices)
    if num_wann < 1 or num_wann > num_orbitals:
        raise ValueError(
            f"num_bands/band_indices selects {num_wann} bands, only {num_orbitals} available")

    particle_hole_operator = None
    particle_hole_perm = None
    if h.has_eh:
        particle_hole_operator = _particle_hole_operator(h, num_orbitals)
        particle_hole_perm = _wannier_particle_hole_permutation(band_indices, num_orbitals)

    trial_vectors_given = trial_vectors is not None
    if trial_vectors is None:
        if h.has_eh: # seed with electron-hole-paired trial orbitals, see docstring
            trial_vectors = _default_eh_trial_vectors(
                num_orbitals, particle_hole_perm, particle_hole_operator)
        else:
            trial_vectors = np.eye(num_orbitals, dtype=complex)[:, :num_wann]
    trial_vectors = np.asarray(trial_vectors, dtype=complex)
    if trial_vectors.shape != (num_orbitals, num_wann):
        raise ValueError(f"trial_vectors must have shape ({num_orbitals},{num_wann}), "
                          f"got {trial_vectors.shape}")

    mp_grid = _mp_grid(h, nk)
    kpt_latt = _monkhorst_pack(mp_grid)
    real_lattice = _real_lattice(h)
    if not getattr(h.geometry, "has_fractional", False):
        h.geometry.get_fractional()
    orbital_positions_frac = _orbital_positions_frac(h, num_orbitals)
    atoms_cart = orbital_positions_frac @ real_lattice
    atom_symbols = ["X"] * num_orbitals

    hk_gen = h.get_hk_gen()
    dim = h.dimensionality
    def hamiltonian_k(k_frac):
        k = np.zeros(3)
        k[:dim] = k_frac[:dim]
        return hk_gen(k)

    keywords = {"num_wann": num_wann, "num_iter": num_iter,
                "conv_tol": conv_tol, "conv_window": conv_window}
    if win_keywords:
        keywords.update(win_keywords)

    setup_result, run_result, H_k_mesh = _wannierize_one_group(
        wannier90, seedname, mp_grid, kpt_latt, real_lattice, atom_symbols,
        atoms_cart, num_orbitals, hamiltonian_k, orbital_positions_frac,
        band_indices, trial_vectors, keywords,
    )
    clusters_used = [band_indices]
    wann_centres = run_result.wann_centres.T
    wann_spreads = run_result.wann_spreads
    setup_results, run_results = setup_result, run_result

    if auto_split_clusters and not h.has_eh and not trial_vectors_given:
        clusters = _split_gapped_clusters(hamiltonian_k, kpt_latt, band_indices,
                                           rel_tol=cluster_rel_tol)
        if len(clusters) > 1:
            # gapped-cluster split: Wannierize each independently, recombine
            # block-diagonally -- only kept if it reproduces the selected
            # spectrum more accurately off-mesh than the joint solution
            # (splitting isn't always better, see _split_gapped_clusters's
            # docstring: it rescues cases where the joint CG gets stuck, it
            # doesn't reliably improve on ones that already converge fine;
            # compared by actual reconstruction accuracy, not total spread
            # -- the latter is gauge-dependent and not always a reliable
            # stand-in, see _offmesh_validation_kfracs)
            num_kpts = kpt_latt.shape[1]
            H_k_mesh_split = np.zeros((num_wann, num_wann, num_kpts), dtype=complex)
            centres_parts, spreads_parts = [], []
            setup_results_split, run_results_split = [], []
            offset = 0
            for cluster in clusters:
                nwc = len(cluster)
                tvc = np.eye(num_orbitals, dtype=complex)[:, :nwc]
                kwc = dict(keywords); kwc["num_wann"] = nwc
                sres, rres, Hk_c = _wannierize_one_group(
                    wannier90, seedname, mp_grid, kpt_latt, real_lattice, atom_symbols,
                    atoms_cart, num_orbitals, hamiltonian_k, orbital_positions_frac,
                    cluster, tvc, kwc,
                )
                H_k_mesh_split[offset:offset + nwc, offset:offset + nwc, :] = Hk_c
                centres_parts.append(rres.wann_centres.T)
                spreads_parts.append(rres.wann_spreads)
                setup_results_split.append(sres)
                run_results_split.append(rres)
                offset += nwc
            val_kfracs = _offmesh_validation_kfracs(mp_grid)
            err_joint = _offmesh_reproduction_error(
                H_k_mesh, kpt_latt, mp_grid, hamiltonian_k, band_indices, val_kfracs, cutoff)
            err_split = _offmesh_reproduction_error(
                H_k_mesh_split, kpt_latt, mp_grid, hamiltonian_k, band_indices, val_kfracs, cutoff)
            if err_split < err_joint:
                clusters_used = clusters
                wann_centres = np.concatenate(centres_parts, axis=0)
                wann_spreads = np.concatenate(spreads_parts)
                H_k_mesh = H_k_mesh_split
                setup_results, run_results = setup_results_split, run_results_split

    if h.has_eh:
        H_k_mesh = _enforce_particle_hole_symmetry(H_k_mesh, kpt_latt, particle_hole_perm)

    hopping = _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=cutoff)

    from .. import geometry as geometry_module
    from ..hamiltonians import Hamiltonian
    from ..multihopping import MultiHopping

    g2 = geometry_module.Geometry()
    g2.dimensionality = h.dimensionality
    g2.a1 = np.array(h.geometry.a1, dtype=float)
    g2.a2 = np.array(h.geometry.a2, dtype=float)
    g2.a3 = np.array(h.geometry.a3, dtype=float)
    g2.r = wann_centres
    g2.r2xyz()
    g2.get_fractional()

    h2 = Hamiltonian(g2)
    h2.has_spin = False
    h2.is_sparse = False
    h2.is_multicell = True
    h2.set_multihopping(MultiHopping(hopping))

    h2.wannier_band_indices = band_indices
    h2.wannier_clusters = clusters_used
    h2.wannier_centres = wann_centres
    h2.wannier_spreads = wann_spreads
    h2.wannier_spread_total = float(np.sum(wann_spreads))
    h2.wannier_setup_result = setup_results
    h2.wannier_run_result = run_results
    if particle_hole_perm is not None:
        h2.wannier_particle_hole_operator = particle_hole_perm
    return h2
