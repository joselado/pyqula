"""0D example: a 4-site linear "molecule" (open boundary, no periodicity).

Hard-coded (k-independent) tight-binding Hamiltonian for 4 sites in a row,
nearest-neighbour hopping t, on-site energy 0::

    H = [[ 0,  t,  0,  0],
         [ t,  0,  t,  0],
         [ 0,  t,  0,  t],
         [ 0,  0,  t,  0]]

A finite/isolated system is treated the same way real Wannier90 handles
molecules: a single k-point at Gamma (``mp_grid = (1, 1, 1)``) in an
arbitrarily large box -- with only one k-point there's no neighbouring
k-point to overlap with except itself, so ``M(k, b) = C(k)^dagger C(k) =
identity`` for every b, regardless of the box size or how many (if any) of
the 4 orbitals are kept: the physics is completely local, as it should be
for a molecule.

4 orbitals -> 4 Wannier functions. Trial projections are the site basis
itself; the converged answer exactly recovers those sites with zero
spread -- worth being upfront about *why*: the standard finite-difference
Wannier centre/spread formulas are built entirely out of how the overlap
*changes* moving across the Brillouin zone, and a single-k-point mesh has
no such variation to extract information from (the identity-overlap
argument above holds for *any* subspace at a single k-point, not just the
full one, so trying disentanglement here -- keeping fewer than 4 WFs --
doesn't change this either). Real Wannier90 has a dedicated ``gamma_only``
real-arithmetic code path for genuinely extracting positions from a single
k-point (not yet ported to this package's pure-Python backend, see
``CLAUDE.md``); this example still exercises the full setup/run pipeline
correctly and is a useful sanity check before trusting the periodic
examples, just not a demonstration of spatial localization.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _tb_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for wannier90, if not pip-installed
from _tb_utils import build_overlaps, monkhorst_pack, report  # noqa: E402

import wannier90  # noqa: E402

NUM_SITES = 4
HOPPING = -1.0


def hamiltonian_k(k_frac: np.ndarray) -> np.ndarray:
    """k-independent: a molecule has no Bloch phases at all."""
    H = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)
    for i in range(NUM_SITES - 1):
        H[i, i + 1] = HOPPING
        H[i + 1, i] = HOPPING
    return H


def main():
    mp_grid = np.array([1, 1, 1], dtype=np.int32)
    real_lattice = 15.0 * np.eye(3)  # an arbitrarily large box -- see module docstring
    kpt_latt = monkhorst_pack(mp_grid)  # just Gamma = (0, 0, 0)

    # Sites spaced 1.5 Ang apart along x, purely for a physically sensible
    # plot -- doesn't affect the tight-binding physics above at all.
    site_spacing = 1.5
    atoms_cart = np.array([[i * site_spacing, 0.0, 0.0] for i in range(NUM_SITES)])

    setup_result = wannier90.setup(
        "molecule", mp_grid, kpt_latt, real_lattice, NUM_SITES,
        ["X"] * NUM_SITES, atoms_cart,
        win_keywords={"num_wann": NUM_SITES, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )

    M_matrix, A_matrix, eigenvalues = build_overlaps(
        hamiltonian_k, NUM_SITES, kpt_latt, setup_result.nnlist
    )
    print("Molecular-orbital energies (eigenvalues of H):", np.round(eigenvalues[:, 0], 4))

    run_result = wannier90.run(
        "molecule", setup_result, mp_grid, kpt_latt, real_lattice, ["X"] * NUM_SITES, atoms_cart,
        M_matrix, A_matrix, eigenvalues, backend="python",
    )
    report(run_result, "0D molecule (4-site chain)")


if __name__ == "__main__":
    main()
