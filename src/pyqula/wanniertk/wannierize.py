"""Wannierize a subset of the bands of a pyqula Hamiltonian using
wannierpy (github.com/joselado/wannierpy)'s pure-Python Wannier90 port,
bundled in this repo at ``pyqula.wanniertk.wannierpy``.

Only the "fixed band subset, no disentanglement" case is implemented
(``num_wann == len(band_indices)``, matching wannierpy's own
``examples/pyqula_ladder.py`` demo): pick the contiguous band range
``bands=[a,b]`` (0-indexed into ``eigh``'s ascending output, both ends
inclusive) at every k-point on a Monkhorst-Pack mesh, Wannierize that
whole subspace jointly, and
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

Point-group symmetries (``symmetries=`` argument, see
``symmetrytk/pointgroup.py`` and ``_resolve_symmetries``/
``_enforce_point_group_symmetry`` below) are handled the same post-hoc
way, but the underlying algebra works out very differently since a
point-group operation is unitary, not antiunitary like electron-hole
conjugation: once the selected bands are validated as a genuine union of
symmetry-related multiplets, the reconstructed Hamiltonian's spectral
symmetry turns out to already be automatic for *any* CG gauge -- see
``_enforce_point_group_symmetry``'s docstring for the derivation. So
``symmetries=`` is mainly a validation gate plus numerical cleanup, not a
correction like the electron-hole case below; it also does not make the
Wannier functions/centres themselves symmetric (that needs CG-internal
constraints, i.e. routing through ``sitesym.py`` for real, not attempted
here).

That post-hoc approach only works when the CG's converged gauge happens
to be electron-hole *covariant* -- i.e. the operator it actually induces,
``C_wan(k) = W(k)^dagger @ particle_hole_operator @ conj(W(-k))``, is the
*same* matrix at every mesh k (not merely unitary at each k individually;
see :func:`_wannier_particle_hole_operator_from_gauge`'s docstring for the
distinction and why both are checked). Nothing about the unconstrained
CG minimization pushes it toward such a gauge, so this holds reliably
only for the full band manifold seeded with identity trial vectors (the
default here -- confirmed empirically: the full manifold's true optimum
is a near-atomic, close-to-identity gauge that inherits covariance from
the original, manifestly covariant orbital basis). Partial (non-full)
band selections generally do *not* converge to a covariant gauge even
with electron-hole-paired trial vectors and full CG convergence
(confirmed on multiple models) -- ``get_wannier_hamiltonian`` detects
this and raises rather than silently returning a Hamiltonian with a
badly wrong spectrum (which is what happened before this check existed).
Making partial BdG selections reliably work would need the same
antiunitary-aware CG extension mentioned above, not just a better trial
vector.

Gauge note: pyqula's own ``get_hk_gen()`` uses the "periodic gauge" --
Bloch phases enter only via integer lattice-vector directions
(``exp(i 2*pi R.k)``, see ``htk/bloch.py``), never via intra-cell atomic
positions. Wannier90's centre/spread formulas assume the opposite
convention (phases include the orbital's intra-cell position), so the
eigenvectors used to build the overlap matrices below are regauged by
``exp(-i 2*pi k.tau_orbital)`` before handing them to wannier90 (the minus
sign: this multiplies eigenvector *coefficients*, which pick up the
opposite phase from the basis vectors they're coefficients of -- see
``_build_overlaps``'s inline derivation) -- exactly the same regauging
wannierpy's own ``pyqula_ladder.py`` example does (see its
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

from . import wannierpy
from ..symmetrytk import pointgroup


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
    (see io_helpers.reciprocal_lattice's docstring).

    Non-periodic rows (index >= h.dimensionality) are replaced with clean,
    mutually orthogonal, artificially large dummy vectors instead of
    trusting whatever ``h.geometry.a2``/``a3`` happen to hold. Some 1D/2D
    geometry constructors already pad these consistently (e.g.
    ``honeycomb_zigzag_ribbon``'s ``a2=[0,100,0]``, per this module's own
    docstring), but others (``triangular_ribbon``, and anything built via
    ``ribbon.py``'s ``geometry_bulk2ribbon``, e.g. ``lieb_ribbon``,
    ``kagome_ribbon``) leave a genuine, lattice-scale, non-orthogonal
    in-plane vector there instead -- harmless for pyqula's own physics
    (``geometrytk/fractional.py``'s ``get_fractional`` never even reads
    ``a2``/``a3`` for ``dim==1``, nor ``a3`` for ``dim==2``: it hardcodes
    the non-periodic axes as untransformed cartesian coordinates), but
    fatal for wannierpy's b-vector shell search (``kmesh_get``), which
    needs a well-conditioned 3x3 cell and raises ``unable to satisfy the
    B1 completeness relation`` on a near-degenerate one. Substituting
    clean padding here fixes every such geometry uniformly, without
    depending on each constructor happening to follow the padding
    convention. Scale matches ``Geometry``'s own default padding
    (``geometry.py``'s ``self.a2 = [0,100,0]``/``self.a3 = [0,0,100]``)
    exactly, rather than scaling with ``a1``: ``kmesh_get``'s shell/
    neighbour-degeneracy tolerance is an absolute distance, so padding
    much larger than this (tried 1000x the norm of ``a1`` first) can
    itself break the neighbour-shell search on some meshes."""
    a1 = np.array(h.geometry.a1, dtype=np.float64)
    a2 = np.array(h.geometry.a2, dtype=np.float64)
    a3 = np.array(h.geometry.a3, dtype=np.float64)
    dim = h.dimensionality
    if dim >= 3:
        return np.array([a1, a2, a3])
    scale = 100.0
    if dim == 2:
        normal = np.cross(a1, a2)
        a3 = normal / np.linalg.norm(normal) * scale
        return np.array([a1, a2, a3])
    # dim == 1: both a2 and a3 are non-periodic padding
    e1 = a1 / np.linalg.norm(a1)
    seed = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = seed - np.dot(seed, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.array([a1, e2 * scale, e3 * scale])


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
    position, so site positions are simply repeated in place.

    Ensures fractional coordinates are fresh itself (not just relying on
    the caller having done so) -- the same "read frac_r before checking
    has_fractional" mistake this module's _particle_hole_operator used to
    make is easy to reintroduce here too if a future caller or reordering
    stops guaranteeing get_fractional() ran immediately beforehand."""
    if not getattr(h.geometry, "has_fractional", False): h.geometry.get_fractional()
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
    # Only the site *count* is needed here (not positions), so use
    # geometry.r rather than frac_r: this function runs before
    # get_wannier_hamiltonian's own get_fractional() call further down,
    # so frac_r may not exist yet, or -- the actual bug this replaced --
    # may hold a stale cached array from an earlier, different-sized
    # geometry (e.g. an intermediate supercell during island construction)
    # if has_fractional was reset to False without clearing frac_r itself.
    n_sites = len(h.geometry.r)
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


def _validate_eh_band_indices(band_indices, num_orbitals):
    """For Nambu/BdG Hamiltonians, a band selection can only be made
    electron-hole symmetric if it is closed under the index pairing
    ``n -> num_orbitals-1-n`` implied by :func:`_particle_hole_operator`
    (ascending-``eigh``-sorted bands at any k: band n's exact
    particle-hole partner is band ``num_orbitals-1-n``, since the full
    spectrum at every k is its own negation under that reindexing --
    picking one forces picking the other). Since ``band_indices`` is a
    contiguous range, this holds exactly when the range is centred on
    ``(num_orbitals-1)/2``."""
    pairs_to = lambda n: num_orbitals - 1 - n
    if sorted(pairs_to(n) for n in band_indices) != sorted(band_indices):
        raise ValueError(
            "get_wannier_hamiltonian: bands must be closed under the "
            f"electron-hole pairing n -> {num_orbitals - 1}-n to enforce particle-hole "
            f"symmetry (pick a band, pick its exact partner too); got {band_indices}")


def _wannier_particle_hole_permutation(band_indices, num_orbitals):
    """(num_wann,num_wann) permutation matrix mapping each selected
    band's local (Wannier-basis) index to its electron-hole partner's --
    the induced action of :func:`_particle_hole_operator` restricted to
    a ``band_indices`` selection already validated as pairing-closed by
    :func:`_default_or_validated_eh_band_indices`.

    Only used to seed :func:`_default_eh_trial_vectors` for partial (non-
    full-manifold) selections -- it is *not* generally the operator
    actually induced on the CG-converged Wannier gauge (that assumption
    is what :func:`_wannier_particle_hole_operator_from_gauge` replaced,
    see its docstring), only a reasonable index-based guess to start the
    CG from."""
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


def _mesh_index_of(kpt_latt, targets, atol, on_miss):
    """For each column of ``targets`` (fractional k-points, any real
    values -- not necessarily pre-reduced to ``[0,1)``), the column index
    of the matching point (mod 1) on mesh ``kpt_latt``, matched by
    minimum-image (periodic, wrap-around) distance -- i.e. a component
    difference of ``0.999999...`` (as roundoff can produce for a
    mathematically exact ``0``/``1`` boundary case, e.g. from
    ``np.linalg.inv`` in :meth:`pointgroup.CompiledSymmetry.k_image`) is
    treated as the ``~0`` it actually is on the torus, not as ``~1``.
    Plain ``np.mod(np.abs(diff), 1.0)`` alone does *not* do this: once
    both operands are already reduced to ``[0,1)`` that outer ``mod`` is
    a no-op, so it silently reports such a roundoff-wrapped near-integer
    as maximally far instead of maximally close -- confirmed to cause
    spurious "mesh is not closed" failures on genuinely closed meshes for
    3D lattices whose symmetry-induced integer matrices are not sparse
    (e.g. ``geometry.pyrochlore_lattice()``).

    ``on_miss(k)`` builds the ``ValueError`` message for column ``k`` if
    no match is found within ``atol``."""
    num_kpts = kpt_latt.shape[1]
    targets = np.mod(targets, 1.0)
    idx = np.empty(num_kpts, dtype=int)
    for k in range(num_kpts):
        diff = np.mod(np.abs(kpt_latt - targets[:, k:k + 1]), 1.0)
        diff = np.minimum(diff, 1.0 - diff)  # minimum-image distance on the torus
        d2 = np.sum(diff ** 2, axis=0)
        j = int(np.argmin(d2))
        if d2[j] > atol:
            raise ValueError(on_miss(k))
        idx[k] = j
    return idx


def _negative_k_index(kpt_latt, atol=1e-8):
    """For each column of ``kpt_latt`` (fractional mesh k-points), the
    column index of ``-k`` (mod 1) on the same mesh -- a Gamma-centred
    Monkhorst-Pack mesh is always closed under negation, which the
    electron-hole symmetrization below relies on."""
    return _mesh_index_of(
        kpt_latt, -kpt_latt, atol,
        lambda k: "get_wannier_hamiltonian: k-mesh is not closed under k -> -k, "
                  "needed to enforce electron-hole symmetry")


def _wannier_particle_hole_operator_from_gauge(W_k_mesh, kpt_latt, particle_hole_operator,
                                                unitarity_atol=1e-4, constancy_atol=1e-3):
    """The *actual* particle-hole operator induced on the converged Wannier
    gauge, ``C_wan(k) = W(k)^dagger @ particle_hole_operator @
    conj(W(-k))``.

    This -- not :func:`_wannier_particle_hole_permutation` -- is what
    :func:`_enforce_particle_hole_symmetry` needs. That helper assumes each
    selected band's partner is at the fixed index ``num_orbitals-1-n``, i.e.
    that the CG-converged Wannier gauge keeps every band's original,
    ascending-eigenvalue-sorted index -- true only when ``num_wann==1`` or
    the CG never mixes bands. In every other case the CG's converged smooth
    gauge mixes band indices freely, and the *true* induced operator
    (computed here) is a completely different matrix from the naive
    index-flip guess.

    ``C_wan(k)`` is unitary at *every* mesh k individually whenever the
    selected subspace is a genuine union of degenerate multiplets of the
    original Hamiltonian (proof: unitarity of ``particle_hole_operator``,
    ``W(k)`` and ``W(-k)`` alone gives this, with no further assumption on
    the gauge) -- so a unitarity failure at any k means the band selection
    slices *through* a degenerate multiplet (picking some, not all, of it),
    which has no well-defined, gauge-covariant induced operator since
    eigh's choice of basis within that multiplet is arbitrary.

    Unitarity alone is *not* enough, though: nothing forces ``C_wan(k)`` to
    be the *same* matrix at every k -- that requires the CG's converged
    gauge to itself be "particle-hole covariant" (``W(-k) = const @
    conj(W(k))`` up to the fixed operator), which the CG has no way to
    know to aim for (confirmed numerically: identity/simple trial-vector
    seeds happen to converge to such a covariant gauge on some models, but
    the production electron-hole-paired trial-vector seed does not on a
    Rashba+Zeeman+s-wave honeycomb BdG model -- ``C_wan(k)`` there varies
    by O(1) across the mesh). Using a single k's value regardless (e.g.
    always the Gamma point) in :func:`_enforce_particle_hole_symmetry`
    would then silently reintroduce large spectral errors on exactly the
    cases this whole mechanism exists to avoid -- checked here instead,
    consistently with the unitarity check above, rather than assumed."""
    neg_idx = _negative_k_index(kpt_latt)
    num_kpts = kpt_latt.shape[1]
    num_wann = W_k_mesh.shape[1]
    C_wan_all = np.empty((num_wann, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        Wk = W_k_mesh[:, :, k]
        Wmk = W_k_mesh[:, :, neg_idx[k]]
        C_wan_all[:, :, k] = Wk.conj().T @ particle_hole_operator @ np.conj(Wmk)
        err = np.max(np.abs(C_wan_all[:, :, k].conj().T @ C_wan_all[:, :, k] - np.eye(num_wann)))
        if err > unitarity_atol:
            raise ValueError(
                "get_wannier_hamiltonian: the selected band group's induced "
                f"electron-hole operator is not unitary at k={kpt_latt[:, k]} (deviation "
                f"{err:.3g}), so exact particle-hole symmetry cannot be enforced -- this band "
                "selection likely slices through a degenerate multiplet (picks some, not all, "
                "of a degenerate cluster), which has no well-defined electron-hole-covariant "
                "subspace; pick a band range that is a union of whole degenerate "
                "multiplets, or increase num_iter/nk if this is a convergence issue")
    C_wan = C_wan_all[:, :, 0]
    variation = np.max(np.abs(C_wan_all - C_wan[:, :, None]))
    if variation > constancy_atol:
        raise ValueError(
            "get_wannier_hamiltonian: the electron-hole operator induced by the converged "
            f"Wannier gauge is not the same matrix at every mesh k-point (max variation "
            f"{variation:.3g}) -- the CG minimization has no constraint keeping the gauge "
            "particle-hole covariant, so exact symmetry enforcement is not possible for this "
            "band selection as-is. This is a gauge-choice issue, not a convergence tail -- "
            "confirmed to persist unchanged across a >10x increase in num_iter on affected "
            "models -- so denser nk/more num_iter will not fix it. Reliably supported today: "
            "the full band manifold (bands=[0, num_orbitals-1]), which defaults to an "
            "identity trial-vector seed chosen to converge to a covariant gauge; a partial "
            "selection may happen to work with hand-picked trial_vectors, but there is no "
            "general recipe for finding a covariant one")
    return C_wan


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


def _resolve_symmetries(h, symmetries):
    """Turn ``get_wannier_hamiltonian``'s ``symmetries`` argument
    (``None``, ``"auto"``, or a list of ``pointgroup.SymmetryOperation``)
    into a group-closed list of ``pointgroup.CompiledSymmetry``, ready for
    :func:`_enforce_point_group_symmetry`.

    ``"auto"`` only ever *narrows* ``h.geometry``'s detected point group
    down to the operations verified against ``h`` itself (see
    ``pointgroup.find_point_group`` -- a Hamiltonian's symmetries are
    always a subset of its geometry's, never a superset); an explicit
    list is verified the same way, but raises instead of silently
    dropping an operation the caller explicitly asked for, since that
    almost always means a mistake in the requested operation rather than
    something to quietly ignore."""
    if symmetries is None:
        return []
    if h.has_eh:
        raise NotImplementedError(
            "get_wannier_hamiltonian: symmetries= is not implemented for Nambu/BdG "
            "Hamiltonians (h.has_eh=True) -- see wanniertk/../symmetrytk/pointgroup.py's "
            "module docstring")
    if symmetries == "auto":
        found = pointgroup.find_point_group(h.geometry, h=h)
        return pointgroup.close_group(h, found)
    ops = list(symmetries)
    if not ops:
        return []
    for op in ops:
        if pointgroup.compile_symmetry(h, op) is None:
            raise ValueError(
                f"get_wannier_hamiltonian: symmetries= includes {op!r}, which is not a "
                "verified symmetry of this Hamiltonian's geometry and H(k) (a Hamiltonian's "
                "symmetries are always a subset of its geometry's) -- check the operation, "
                'or use symmetries="auto" to only ever use genuinely-present ones')
    return pointgroup.close_group(h, ops)


def _symmetry_target_index(compiled, kpt_latt, atol=1e-6):
    """For each mesh index ``k``, the mesh index of
    ``compiled.k_image(kpt_latt[:,k])`` (mod 1) -- generalizes
    :func:`_negative_k_index` (the ``k -> -k`` case used for
    electron-hole symmetry) to an arbitrary point-group operation's
    ``k -> k'`` action. Requires the k-mesh to be closed under this
    action, true for a Gamma-centred Monkhorst-Pack mesh as long as the
    operation doesn't need to relate periodic directions sampled with
    different densities."""
    dim = kpt_latt.shape[0]
    num_kpts = kpt_latt.shape[1]
    images = np.array([compiled.k_image(kpt_latt[:, k])[:dim] for k in range(num_kpts)]).T
    return _mesh_index_of(
        kpt_latt, images, atol,
        lambda k: "get_wannier_hamiltonian: k-mesh is not closed under symmetry "
                  f"{compiled.op.name}'s k -> k' action (needed to enforce it) -- likely an "
                  "anisotropic nk mismatched with a symmetry that mixes periodic directions "
                  "sampled at different densities; use a uniform nk instead")


def _point_group_wannier_operators(compiled, W_k_mesh, kpt_latt, target_idx,
                                    unitarity_atol=1e-4):
    """``(num_wann,num_wann,num_kpts)`` array of the Wannier-gauge
    intertwiners ``D(R,k) = W(Rk)^dagger @ P(R,k) @ W(k)`` induced by one
    compiled symmetry -- the point-group analogue of
    :func:`_wannier_particle_hole_operator_from_gauge`, without the
    complex conjugation that function needs for its antiunitary operator
    (a point-group operation is unitary, so none appears here).

    ``D(R,k)`` is unitary at every mesh k whenever the selected band
    subspace is a genuine union of ``R``-related multiplets (proof:
    unitarity of ``P(R,k)``, ``W(Rk)`` and ``W(k)`` alone gives this) --
    so a unitarity failure means the band selection slices through a
    symmetry-related degenerate multiplet, checked here and raised on,
    exactly mirroring the electron-hole case. Unlike that case, no
    separate "same matrix at every k" check is needed: ``D(R,k)``
    genuinely depends on k by construction, and the family
    ``{D(R,k)}_{R,k}`` automatically satisfies the group cocycle
    condition ``D(R2,R1 k) D(R1,k) = D(R2 R1,k)`` given only the
    per-(R,k) unitarity checked here (each ``P(R,k)`` maps the selected
    subspace at k isomorphically onto the one at ``Rk``, so composing two
    such maps composes their group elements) -- which is exactly what
    makes the group-averaging in :func:`_enforce_point_group_symmetry`
    exactly covariant rather than merely unitary."""
    num_kpts = kpt_latt.shape[1]
    num_wann = W_k_mesh.shape[1]
    D_all = np.empty((num_wann, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        P, _ = compiled.orbital_operator(kpt_latt[:, k])
        kdst = target_idx[k]
        D = W_k_mesh[:, :, kdst].conj().T @ P @ W_k_mesh[:, :, k]
        err = np.max(np.abs(D.conj().T @ D - np.eye(num_wann)))
        if err > unitarity_atol:
            raise ValueError(
                "get_wannier_hamiltonian: the selected band group's induced operator for "
                f"symmetry {compiled.op.name} is not unitary at k={kpt_latt[:, k]} (deviation "
                f"{err:.3g}) -- this band selection likely slices through a symmetry-related "
                "degenerate multiplet (picks some, not all, of it), which has no well-defined "
                "symmetry-covariant subspace for this operation; pick a band range that is a "
                "union of whole such multiplets, drop this symmetry from `symmetries`, or "
                "increase num_iter/nk if this is a convergence issue")
        D_all[:, :, k] = D
    return D_all


def _enforce_point_group_symmetry(H_k_mesh, W_k_mesh, kpt_latt, group):
    """Symmetrize a Wannier-gauge Bloch Hamiltonian mesh over a
    group-closed list of ``pointgroup.CompiledSymmetry`` (see
    :func:`_resolve_symmetries`) via the Reynolds/group-averaging
    operator ``H_sym(k) = (1/|G|) sum_R D(R,k_src) H(k_src) D(R,k_src)^dagger``,
    ``k_src`` the mesh point ``R`` maps to ``k`` -- the point-group
    analogue of :func:`_enforce_particle_hole_symmetry`. A no-op (returns
    ``H_k_mesh`` unchanged) when ``group`` is empty.

    Unlike the electron-hole case, this is *not* generally correcting a
    real violation: for a genuine unitary symmetry (no conjugation, unlike
    electron-hole's antiunitary ``C``), one can show algebraically that
    ``D(R,k) H(k) D(R,k)^dagger == H(k_dst)`` *already* holds exactly for
    *any* per-k gauge ``W(k)`` built from true eigenvectors of the
    selected subspace -- the projector ``W(k) W(k)^dagger`` onto that
    subspace is gauge-independent, and the whole identity reduces to the
    original Hamiltonian's own symmetry once the per-(R,k) unitarity in
    :func:`_point_group_wannier_operators` holds (confirmed empirically:
    a deliberately non-covariant, per-k-random gauge on a genuine
    invariant subspace already satisfies this to machine precision,
    before this function is even called). So this averaging step is
    mainly a numerical-residual cleanup + a strong validation gate (via
    the unitarity check) rather than a correction, in contrast to
    electron-hole enforcement which fixes a genuine, generically large
    gauge mismatch. What this does *not* guarantee: that the *Wannier
    functions themselves* (``W_k_mesh``, hence ``wannier_functions`` and
    ``wannier_centres``) sit at symmetry-related positions -- that needs
    ``D(R,k)`` to be *k-independent* (a real, generally unsatisfied
    constraint on the gauge, analogous to electron-hole's "constancy"
    requirement), which only CG-internal symmetry constraints (the
    vendored but unused ``wannierpy/_engine/sitesym.py`` engine, see the
    module docstring) can reliably achieve -- not implemented here.

    Accumulates by looping over each group element's *source* k (scatter-
    adding each term straight into its destination ``tgt[k_src]``) rather
    than looping over destination k and inverting ``tgt`` first -- same
    sum (``tgt`` is a bijection on the mesh, checked by
    :func:`_symmetry_target_index`), one array instead of two."""
    if not group:
        return H_k_mesh
    num_kpts = kpt_latt.shape[1]
    H_sym = np.zeros_like(H_k_mesh)
    for compiled in group:
        tgt = _symmetry_target_index(compiled, kpt_latt)
        D_all = _point_group_wannier_operators(compiled, W_k_mesh, kpt_latt, tgt)
        for k_src in range(num_kpts):
            D = D_all[:, :, k_src]
            H_sym[:, :, tgt[k_src]] += D @ H_k_mesh[:, :, k_src] @ D.conj().T
    return H_sym / len(group)


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


def _build_overlaps(hamiltonian_k, num_orbitals, kpt_latt, nnlist, nncell,
                     num_periodic, orbital_positions_frac, band_indices, trial_vectors):
    """Diagonalize ``hamiltonian_k`` on the wannierization mesh and build
    the M/A/eigenvalue arrays ``wannierpy.run`` needs, restricted to a
    fixed band subset with a fixed trial projection matrix -- the "no
    disentanglement, pre-selected bands" path. A pyqula-local port of the
    relevant part of wannierpy's examples/_tb_utils.py::build_overlaps
    (that module is example code, not part of wannierpy's installable
    package, so it isn't imported directly).

    Also returns ``C_bare``, the same selected-band eigenvectors *before*
    the position regauging below, in pyqula's own orbital-basis
    convention -- needed by :func:`_wannierize_one_group` to reconstruct
    actual real-space Wannier functions (as opposed to just the
    Wannier-gauge Hamiltonian, which -- per the module docstring --
    doesn't care which convention U(k) is applied to)."""
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

    # bare (un-regauged) selected-band eigenvectors -- pyqula's own
    # orbital-basis convention, kept aside so the actual real-space
    # Wannier functions can later be built in that same convention (see
    # _wannierize_one_group); the M/A matrices below need the
    # position-regauged C instead (see module docstring), so both are
    # computed off the same eigh call but diverge from here on.
    C_bare = C_full[:, list(band_indices), :].copy()

    # Regauging from pyqula's periodic-gauge basis states (|k,b>_perio =
    # (1/sqrt N) sum_R exp(i2*pi*k.R)|R,b>) to Wannier90's convention I
    # (|k,b>_conv1 = (1/sqrt N) sum_R exp(i2*pi*k.(R+tau_b))|R,b> =
    # exp(i2*pi*k.tau_b) |k,b>_perio) needs the *coefficient* transform, not
    # the basis-vector one: writing the same physical state as sum_b c_b
    # |k,b>_perio = sum_b c'_b |k,b>_conv1 and substituting
    # |k,b>_perio = exp(-i2*pi*k.tau_b) |k,b>_conv1 gives
    # c'_b = c_b * exp(-i2*pi*k.tau_b) -- the OPPOSITE sign from the basis
    # vectors' own phase (the usual covariant/contravariant flip for
    # coordinates vs. basis vectors under a phase rescaling). C_full's
    # columns store eigenvector *coefficients* (eigh's output), so this
    # regauging phase must carry the minus sign.
    tau = np.asarray(orbital_positions_frac, dtype=np.float64)  # (num_orbitals,3)
    phase = np.exp(-1j * 2 * np.pi * (tau @ kpt_latt))  # (num_orbitals,num_kpts)
    C_full = C_full * phase[:, None, :]  # regauge each orbital row, see module docstring

    C = C_full[:, list(band_indices), :]
    eigenvalues = eig_full[list(band_indices), :]

    # Wrap-around phase correction. A mesh neighbor k+b that leaves the
    # first BZ is written as k2+G on the mesh (k2 = nnlist-1, G = nncell,
    # an integer reciprocal-lattice shift). In the position-regauged
    # ("convention I") gauge used for the M matrices, eigenvector
    # *coefficients* obey C(k2+G)_alpha = exp(-i 2*pi G.tau_alpha)
    # C(k2)_alpha (same sign as the regauging phase above, by the same
    # coefficient-vs-basis-vector argument -- see it for the derivation),
    # so the true overlap is
    # M(k,b) = C(k)^dagger @ diag(exp(-i 2*pi G.tau)) @ C(k2). The bare
    # C(k2) (missing that phase) is what the old code used, which silently
    # dropped the G-dependent phase whenever a neighbor wrapped -- the
    # bug behind huge, mesh-dependent spreads on ribbon geometries.
    #
    # Only G components along genuinely periodic directions are physical:
    # pyqula pads non-periodic directions with dummy lattice vectors
    # (e.g. a2=[0,100,0]) whose tiny reciprocal vectors make kmesh_get
    # emit spurious b-vectors with nncell!=0 there, while tau along those
    # directions is a raw cartesian coordinate (get_fractional leaves
    # non-periodic axes untransformed), not a meaningful fractional
    # position -- so their tau.G product is meaningless. Masking G to the
    # periodic directions keeps those self-neighbors at M = C^dagger C
    # (as before) while applying the correct phase where it belongs.
    G_mask = np.zeros(3, dtype=np.float64)
    G_mask[:num_periodic] = 1.0
    M_matrix = np.empty((num_selected, num_selected, nntot, num_kpts), dtype=complex)
    A_matrix = np.empty((num_selected, num_wann, num_kpts), dtype=complex)
    for k in range(num_kpts):
        A_matrix[:, :, k] = C[:, :, k].conj().T @ trial_vectors
        for nn in range(nntot):
            k2 = int(nnlist[k, nn]) - 1
            G = nncell[:, k, nn].astype(np.float64) * G_mask
            wrap_phase = np.exp(-1j * 2 * np.pi * (tau @ G))  # (num_orbitals,)
            M_matrix[:, :, nn, k] = C[:, :, k].conj().T @ (wrap_phase[:, None] * C[:, :, k2])

    return M_matrix, A_matrix, eigenvalues, C_bare


def _bloch_hamiltonian_from_gauge(U_matrix, eigenvalues):
    """H_W(k) = U(k)^dagger @ diag(eigenvalues(k)) @ U(k) -- the
    selected-band Hamiltonian rotated into the smooth Wannier gauge.
    ``U_matrix[m,n,k]``: m = index into the selected/original band
    subspace, n = Wannier index (overlap.py's convention, confirmed by
    its own unitarity check). Exact by construction: eigenvalues of a
    unitary similarity transform are unchanged, so this reproduces the
    original selected-band spectrum exactly at every mesh k-point."""
    return np.einsum("mik,mk,mjk->ijk", U_matrix.conj(), eigenvalues, U_matrix)


def _mesh_to_real_space(M_k_mesh, kpt_latt, mp_grid):
    """Inverse-Fourier-transform a (dim1,dim2,num_kpts) array sampled on
    a Monkhorst-Pack mesh into a ``{R: (dim1,dim2) ndarray}`` dict, using
    pyqula's own Bloch convention (H(k) = sum_R m_R exp(i 2*pi R.k), see
    htk/bloch.py's evaluate_bloch_matrix_jit) -- shared by
    :func:`_hopping_from_bloch` (Hamiltonian reconstruction) and
    :func:`get_wannier_hamiltonian`'s own real-space Wannier function
    reconstruction, which Fourier-transform different quantities with
    the same convention so both stay expressed relative to the same
    cell labels."""
    num_kpts = M_k_mesh.shape[2]
    axis_ranges = [np.fft.fftfreq(int(n)).astype(np.float64) * int(n) for n in mp_grid]
    Rs = np.array(list(itertools.product(*axis_ranges)), dtype=np.float64)
    phase = np.exp(-1j * 2 * np.pi * (Rs @ kpt_latt))  # (num_R,num_kpts)
    M_R_all = np.einsum("rk,ijk->rij", phase, M_k_mesh) / num_kpts
    return {tuple(int(round(x)) for x in Rs[idx]): M_R_all[idx] for idx in range(len(Rs))}


def _drop_negligible_cells(R_to_matrix, cutoff):
    """Keep only cells whose matrix has an element above ``cutoff``
    (always keeping the home cell (0,0,0)) -- used to truncate both the
    real-space hoppings and the real-space Wannier function coefficients
    to their numerically significant range."""
    return {R: M for R, M in R_to_matrix.items()
            if R == (0, 0, 0) or np.max(np.abs(M)) > cutoff}


def _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=1e-6):
    """Inverse-Fourier-transform a (num_wann,num_wann,num_kpts) Bloch
    Hamiltonian sampled on a Monkhorst-Pack mesh into real-space hopping
    matrices -- so plugging the result into
    set_multihopping()/get_hk_gen() exactly reproduces H_k_mesh at every
    mesh k-point (trigonometric interpolation elsewhere)."""
    return _drop_negligible_cells(_mesh_to_real_space(H_k_mesh, kpt_latt, mp_grid), cutoff)


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


def _wannierize_one_group(seedname, mp_grid, kpt_latt, real_lattice,
        atom_symbols, atoms_cart, num_orbitals, num_periodic, hamiltonian_k,
        orbital_positions_frac, band_indices, trial_vectors, keywords):
    """Run one full setup/overlaps/CG-run/reconstruction cycle for a
    single band group -- the inner loop body shared by the single-group
    and auto-split-into-clusters paths of :func:`get_wannier_hamiltonian`."""
    setup_result = wannierpy.setup(
        seedname, mp_grid, kpt_latt, real_lattice, num_orbitals,
        atom_symbols, atoms_cart, win_keywords=keywords, backend="python",
    )
    M_matrix, A_matrix, eigenvalues, C_bare = _build_overlaps(
        hamiltonian_k, num_orbitals, kpt_latt, setup_result.nnlist,
        setup_result.nncell, num_periodic,
        orbital_positions_frac, band_indices, trial_vectors,
    )
    run_result = wannierpy.run(
        seedname, setup_result, mp_grid, kpt_latt, real_lattice,
        atom_symbols, atoms_cart, M_matrix, A_matrix, eigenvalues, backend="python",
    )
    H_k_mesh = _bloch_hamiltonian_from_gauge(run_result.U_matrix, eigenvalues)

    # W(k) = C_bare(k) @ U_matrix(k): rotate the *bare* (un-regauged, see
    # _build_overlaps) selected-band eigenvectors into the smooth Wannier
    # gauge -- the real-space Fourier transform of this (done by the
    # caller, alongside H_k_mesh's) gives the actual Wannier function
    # coefficients in pyqula's own orbital basis. Using C_bare rather
    # than the position-regauged C used for the M/A matrices above makes
    # this exact and self-consistent with the returned Hamiltonian: since
    # C_bare are eigenvectors of the *actual* hamiltonian_k matrix used
    # throughout this module, C_bare(k)^dagger @ hamiltonian_k(k) @
    # C_bare(k) == diag(eigenvalues(k)) restricted to the selected bands,
    # so W(k)^dagger @ hamiltonian_k(k) @ W(k) == H_k_mesh(k) exactly at
    # every mesh k-point -- i.e. Fourier-transforming both sides gives
    # <w_n,0| h |w_n',R> == hopping[R][n,n'] for the Hamiltonian
    # get_wannier_hamiltonian returns (approximately, not exactly, for a
    # has_eh Hamiltonian: _enforce_particle_hole_symmetry perturbs
    # H_k_mesh after this identity would otherwise hold, without
    # correspondingly perturbing U_matrix).
    W_k_mesh = np.einsum("omk,mnk->onk", C_bare, run_result.U_matrix)
    return setup_result, run_result, H_k_mesh, W_k_mesh


def get_wannier_hamiltonian(h, bands=None, nk=12,
        trial_vectors=None, num_iter=200, conv_tol=1e-10, conv_window=3,
        cutoff=1e-6, seedname="pyqula_wannier", win_keywords=None,
        auto_split_clusters=False, cluster_rel_tol=0.1, symmetries=None):
    """Wannierize a fixed subset of ``h``'s bands and return a new pyqula
    Hamiltonian whose real-space hoppings exactly reproduce that band
    subspace on the wannierization mesh (and interpolate smoothly
    elsewhere).

    Parameters
    ----------
    h : Hamiltonian
        Must be periodic (``h.dimensionality>=1``).
    bands : sequence of two int
        ``[a,b]``, the first and last band to Wannierize (0-indexed into
        ``eigh``'s ascending output, both ends inclusive) -- every band
        in between is Wannierized jointly as a single group
        (``band_indices = list(range(a,b+1))``, ``num_wann = b-a+1``).
        For a Nambu/BdG Hamiltonian (``h.has_eh=True``) this range must be
        closed under the electron-hole pairing ``n -> num_orbitals-1-n``,
        i.e. centred on the gap (``a+b == num_orbitals-1``) -- otherwise a
        ``ValueError`` is raised.
    nk : int or sequence of int, optional
        k-points per periodic direction for the Monkhorst-Pack
        wannierization mesh (default 12). A sequence must have
        ``h.dimensionality`` entries.
    trial_vectors : (num_orbitals, num_wann) complex ndarray, optional
        Fixed (k-independent) trial projection matrix seeding the CG
        minimization -- default: a random real matrix (fresh, unseeded draw
        each call), used to check that the converged spread/geometry don't
        depend on the particular trial seed (see ``tests/wannier``).
    num_iter, conv_tol, conv_window : optional
        Wannier90 CG minimization parameters, passed through
        ``win_keywords``.
    cutoff : float, optional
        Real-space hopping matrices with max element below this are
        dropped (except the intracell (0,0,0) term, always kept).
    seedname : str, optional
        Passed to ``wannierpy.setup``/``run`` (only used for logging by
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
    symmetries : None, "auto", or list of ``symmetrytk.pointgroup.SymmetryOperation``, optional
        Default None (no symmetry enforcement, current behaviour
        unchanged). ``"auto"`` detects ``h.geometry``'s point group
        (``symmetrytk.pointgroup.find_point_group``, a best-effort,
        dependency-free heuristic -- see its docstring for what it can
        and can't find) and keeps only the operations verified as actual
        symmetries of ``h`` (a Hamiltonian's symmetries are always a
        subset of its geometry's). A list enforces exactly those
        operations instead, after the same per-operation verification
        (raises ``ValueError`` if one doesn't verify -- construct it with
        ``symmetrytk.pointgroup.SymmetryOperation`` and pass a ``center``
        if the relevant rotation axis doesn't pass through the origin).
        Either way the requested/detected operations are closed into a
        full group (``pointgroup.close_group``), and the selected bands
        are validated as a genuine union of symmetry-related multiplets
        under every element (raises ``ValueError`` otherwise, the same
        failure mode ``h.has_eh``'s enforcement has for electron-hole
        pairs -- see ``_point_group_wannier_operators``). Not implemented
        for ``h.has_eh`` Hamiltonians (raises ``NotImplementedError``).

        Note what this does and does not guarantee: for a genuine
        (unitary) point-group symmetry, unlike electron-hole's antiunitary
        one, the reconstructed Hamiltonian's spectral symmetry is already
        automatic once the validation above passes -- *any* CG gauge on a
        valid selection reproduces it exactly (see
        ``_enforce_point_group_symmetry``'s docstring for the algebra), so
        this option mainly acts as that validation gate plus numerical
        cleanup, not a correction. It does *not* make the *Wannier
        functions/centres themselves* (``wannier_functions``,
        ``wannier_centres``) sit at symmetric positions -- that would need
        symmetry constraints inside the CG minimization itself (not
        implemented; see the module docstring).

    Returns
    -------
    WannierHamiltonian
        A new, multicell ``Hamiltonian`` subclass (see
        ``wanniertk.wannierhamiltonian.WannierHamiltonian`` -- every
        ordinary Hamiltonian method works unchanged) with ``num_wann``
        orbitals per cell, positioned at the computed Wannier centres.
        Also carries ``wannier_functions`` -- ``{R: (num_orbitals,
        num_wann) ndarray}``, the real-space Wannier function
        coefficients in ``h``'s own orbital basis (column n, row o: the
        amplitude of Wannier function n, translated to cell R relative
        to the home cell, on orbital o of ``h``) -- along with
        ``wannier_band_indices``, ``wannier_clusters`` (the
        ``auto_split_clusters`` decomposition actually used -- a single
        one-element list when splitting didn't trigger), ``wannier_centres``,
        ``wannier_spreads``, ``wannier_spread_total``,
        ``wannier_setup_result``, ``wannier_run_result`` for diagnostics
        (each a single wannierpy result, or a list with one entry per
        cluster when ``auto_split_clusters`` engaged), plus
        ``wannier_particle_hole_operator`` (the unitary ``C_wan``
        with ``C_wan @ conj(h_R) @ C_wan^-1 == -h_R`` for every real-space
        hopping ``h_R`` of the result) when ``h.has_eh``, and
        ``wannier_symmetries`` (the group-closed list of
        ``pointgroup.CompiledSymmetry`` actually enforced) when
        ``symmetries`` was given.

        Raises ``ValueError`` instead of returning a Hamiltonian with a
        badly wrong spectrum when ``h.has_eh`` and the CG-converged
        Wannier gauge does not admit a well-defined, mesh-wide-constant
        electron-hole operator (see the module docstring's "post-hoc
        approach only works when..." paragraph) -- reliable for the full
        band manifold (the default trial-vector seed for it is chosen
        specifically to make this hold), not generally for a partial
        selection. This is a property of *which smooth gauge the CG
        happens to land on*, not of convergence quality -- more
        ``num_iter``/denser ``nk`` does not fix it once it occurs.
    """
    if h.dimensionality < 1:
        raise NotImplementedError(
            "get_wannier_hamiltonian needs a periodic Hamiltonian (h.dimensionality>=1)")

    num_orbitals = h.intra.shape[0]
    if bands is None:
        raise ValueError(
            "get_wannier_hamiltonian: pass bands=[a,b], the first and last band "
            "(0-indexed, inclusive) to Wannierize")
    a, b = bands
    if a > b:
        raise ValueError(f"get_wannier_hamiltonian: bands=[{a},{b}] needs a<=b")
    band_indices = list(range(a, b + 1))
    num_wann = len(band_indices)
    if num_wann < 1 or num_wann > num_orbitals:
        raise ValueError(
            f"bands=[{a},{b}] selects {num_wann} bands, only {num_orbitals} available")
    if h.has_eh: # Nambu/BdG: band selection must be electron-hole-pair-closed
        _validate_eh_band_indices(band_indices, num_orbitals)

    # resolved/verified/group-closed up front (before the expensive CG
    # run) so a bad symmetries= argument fails fast -- see _resolve_symmetries
    symmetry_group = _resolve_symmetries(h, symmetries)

    particle_hole_operator = None
    seed_particle_hole_perm = None
    if h.has_eh:
        particle_hole_operator = _particle_hole_operator(h, num_orbitals)
        # index-flip guess, good enough to seed the CG (see
        # _default_eh_trial_vectors) but NOT what actually gets used to
        # enforce symmetry below -- see _wannier_particle_hole_operator_from_gauge
        seed_particle_hole_perm = _wannier_particle_hole_permutation(band_indices, num_orbitals)

    trial_vectors_given = trial_vectors is not None
    if trial_vectors is None:
        if h.has_eh and num_wann == num_orbitals:
            # Full manifold: identity trial vectors seed the CG at the
            # (manifestly electron-hole-covariant, since W(k)=I trivially
            # gives a k-independent induced operator equal to
            # particle_hole_operator itself) original orbital basis, and
            # the CG's own spread minimization has no reason to move far
            # from it -- a full manifold's true optimum is a zero-spread,
            # near-atomic-orbital gauge (see test_ladder_full_manifold_
            # reproduces_spectrum_exactly), so this reliably converges to
            # an (almost) exactly covariant gauge. Confirmed empirically
            # to matter: _default_eh_trial_vectors' electron-hole-paired
            # orbital seed, despite being designed with this symmetry in
            # mind, converges to a gauge whose induced operator varies by
            # O(1) across the mesh on some models (Rashba+s-wave chains,
            # Rashba+Zeeman+s-wave honeycomb) even fully converged --
            # silently corrupting _enforce_particle_hole_symmetry's single
            # fixed-operator averaging. This does not extend to partial
            # (non-full) band selections: identity columns there converge
            # to comparably bad, non-covariant gauges (checked
            # numerically), so those still use the paired heuristic below
            # and rely on _wannier_particle_hole_operator_from_gauge's
            # checks to catch the cases it still fails on.
            trial_vectors = np.eye(num_orbitals, dtype=complex)
        elif h.has_eh: # seed with electron-hole-paired trial orbitals, see docstring
            trial_vectors = _default_eh_trial_vectors(
                num_orbitals, seed_particle_hole_perm, particle_hole_operator)
        else:
            trial_vectors = np.random.default_rng().standard_normal((num_orbitals, num_wann))
    trial_vectors = np.asarray(trial_vectors, dtype=complex)
    if trial_vectors.shape != (num_orbitals, num_wann):
        raise ValueError(f"trial_vectors must have shape ({num_orbitals},{num_wann}), "
                          f"got {trial_vectors.shape}")

    mp_grid = _mp_grid(h, nk)
    kpt_latt = _monkhorst_pack(mp_grid)
    real_lattice = _real_lattice(h)
    orbital_positions_frac = _orbital_positions_frac(h, num_orbitals) # ensures fractional coords itself
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

    setup_result, run_result, H_k_mesh, W_k_mesh = _wannierize_one_group(
        seedname, mp_grid, kpt_latt, real_lattice, atom_symbols,
        atoms_cart, num_orbitals, dim, hamiltonian_k, orbital_positions_frac,
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
            W_k_mesh_split = np.zeros((num_orbitals, num_wann, num_kpts), dtype=complex)
            centres_parts, spreads_parts = [], []
            setup_results_split, run_results_split = [], []
            offset = 0
            for cluster in clusters:
                nwc = len(cluster)
                tvc = np.eye(num_orbitals, dtype=complex)[:, :nwc]
                kwc = dict(keywords); kwc["num_wann"] = nwc
                sres, rres, Hk_c, Wk_c = _wannierize_one_group(
                    seedname, mp_grid, kpt_latt, real_lattice, atom_symbols,
                    atoms_cart, num_orbitals, dim, hamiltonian_k, orbital_positions_frac,
                    cluster, tvc, kwc,
                )
                H_k_mesh_split[offset:offset + nwc, offset:offset + nwc, :] = Hk_c
                W_k_mesh_split[:, offset:offset + nwc, :] = Wk_c
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
                W_k_mesh = W_k_mesh_split
                setup_results, run_results = setup_results_split, run_results_split

    particle_hole_perm = None
    if h.has_eh:
        # the operator actually induced by the CG-converged Wannier gauge,
        # not the index-flip seed guess -- see
        # _wannier_particle_hole_operator_from_gauge's docstring for why
        # they generally differ
        particle_hole_perm = _wannier_particle_hole_operator_from_gauge(
            W_k_mesh, kpt_latt, particle_hole_operator)
        H_k_mesh = _enforce_particle_hole_symmetry(H_k_mesh, kpt_latt, particle_hole_perm)

    if symmetry_group:
        H_k_mesh = _enforce_point_group_symmetry(H_k_mesh, W_k_mesh, kpt_latt, symmetry_group)

    hopping = _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=cutoff)
    # W_k_mesh is already C_bare(k) @ U_matrix(k) (see _wannierize_one_group),
    # so the same Fourier-transform-and-truncate step _hopping_from_bloch uses
    # turns it into real-space Wannier function coefficients -- except
    # _mesh_to_real_space extracts a Fourier SERIES coefficient (h_R = (1/N)
    # sum_k H(k) exp(-i*2*pi*R.k), correct for a hopping matrix), while
    # W_k_mesh is a Bloch expansion coefficient substituted directly into
    # pyqula's convention |k,o> = (1/sqrt(N)) sum_R exp(+i*2*pi*k.R) |R,o>,
    # which needs the opposite sign: c_R[o,n] = (1/N) sum_k W_k_mesh[o,n,k]
    # exp(+i*2*pi*k.R). Reusing _mesh_to_real_space unmodified therefore
    # returns the dict keyed by -R instead of R; negate the keys to match
    # the documented convention (see the "Returns" docstring above and
    # every examples/wannier/*/main.py plotting script, all of which assume
    # wannier_functions[R][o,n] sits at h.geometry.r[o] + R*a1).
    wannier_functions = _drop_negligible_cells(
        {tuple(-x for x in R): M for R, M in
         _mesh_to_real_space(W_k_mesh, kpt_latt, mp_grid).items()}, cutoff)

    from .. import geometry as geometry_module
    from .wannierhamiltonian import WannierHamiltonian
    from ..multihopping import MultiHopping

    g2 = geometry_module.Geometry()
    g2.dimensionality = h.dimensionality
    g2.a1 = np.array(h.geometry.a1, dtype=float)
    g2.a2 = np.array(h.geometry.a2, dtype=float)
    g2.a3 = np.array(h.geometry.a3, dtype=float)
    g2.r = wann_centres
    g2.r2xyz()
    g2.get_fractional()

    h2 = WannierHamiltonian(g2)
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
    h2.wannier_functions = wannier_functions
    if particle_hole_perm is not None:
        h2.wannier_particle_hole_operator = particle_hole_perm
    if symmetry_group:
        h2.wannier_symmetries = symmetry_group
    return h2
