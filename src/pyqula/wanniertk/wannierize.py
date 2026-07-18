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


def get_wannier_hamiltonian(h, num_bands=None, band_indices=None, nk=12,
        trial_vectors=None, num_iter=200, conv_tol=1e-10, conv_window=3,
        cutoff=1e-6, seedname="pyqula_wannier", win_keywords=None):
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
        ``band_indices`` is given.
    band_indices : sequence of int, optional
        Explicit 0-indexed band selection, overriding ``num_bands``
        (``num_wann = len(band_indices)``).
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

    Returns
    -------
    Hamiltonian
        A new, multicell pyqula Hamiltonian with ``num_wann`` orbitals
        per cell, positioned at the computed Wannier centres. Also
        carries ``wannier_band_indices``, ``wannier_centres``,
        ``wannier_spreads``, ``wannier_spread_total``,
        ``wannier_setup_result``, ``wannier_run_result`` for diagnostics.
    """
    if h.dimensionality < 1:
        raise NotImplementedError(
            "get_wannier_hamiltonian needs a periodic Hamiltonian (h.dimensionality>=1)")
    wannier90 = _import_wannier90()

    num_orbitals = h.intra.shape[0]
    if band_indices is None:
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

    if trial_vectors is None:
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
    orbital_positions_frac = np.array(h.geometry.frac_r, dtype=np.float64)
    atoms_cart = orbital_positions_frac @ real_lattice
    atom_symbols = ["X"] * num_orbitals

    keywords = {"num_wann": num_wann, "num_iter": num_iter,
                "conv_tol": conv_tol, "conv_window": conv_window}
    if win_keywords:
        keywords.update(win_keywords)

    setup_result = wannier90.setup(
        seedname, mp_grid, kpt_latt, real_lattice, num_orbitals,
        atom_symbols, atoms_cart, win_keywords=keywords, backend="python",
    )

    hk_gen = h.get_hk_gen()
    dim = h.dimensionality
    def hamiltonian_k(k_frac):
        k = np.zeros(3)
        k[:dim] = k_frac[:dim]
        return hk_gen(k)

    M_matrix, A_matrix, eigenvalues = _build_overlaps(
        hamiltonian_k, num_orbitals, kpt_latt, setup_result.nnlist,
        orbital_positions_frac, band_indices, trial_vectors,
    )

    run_result = wannier90.run(
        seedname, setup_result, mp_grid, kpt_latt, real_lattice,
        atom_symbols, atoms_cart, M_matrix, A_matrix, eigenvalues, backend="python",
    )

    H_k_mesh = _bloch_hamiltonian_from_gauge(run_result.U_matrix, eigenvalues)
    hopping = _hopping_from_bloch(H_k_mesh, kpt_latt, mp_grid, cutoff=cutoff)

    from .. import geometry as geometry_module
    from ..hamiltonians import Hamiltonian
    from ..multihopping import MultiHopping

    centres_cart = run_result.wann_centres.T  # (num_wann,3)
    g2 = geometry_module.Geometry()
    g2.dimensionality = h.dimensionality
    g2.a1 = np.array(h.geometry.a1, dtype=float)
    g2.a2 = np.array(h.geometry.a2, dtype=float)
    g2.a3 = np.array(h.geometry.a3, dtype=float)
    g2.r = centres_cart
    g2.r2xyz()
    g2.get_fractional()

    h2 = Hamiltonian(g2)
    h2.has_spin = False
    h2.is_sparse = False
    h2.is_multicell = True
    h2.set_multihopping(MultiHopping(hopping))

    h2.wannier_band_indices = band_indices
    h2.wannier_centres = centres_cart
    h2.wannier_spreads = run_result.wann_spreads
    h2.wannier_spread_total = run_result.spread_total
    h2.wannier_setup_result = setup_result
    h2.wannier_run_result = run_result
    return h2
