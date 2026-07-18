"""2D example: a two-orbital checkerboard-like lattice.

Hard-coded Bloch Hamiltonian, 2 orbitals per unit cell (A at the cell
corner, B at the cell centre) with on-site energies eps1/eps2,
same-orbital nearest-neighbour hopping t1/t2 along x and y, and a constant
(k-independent) inter-orbital hybridization t12 (periodic-gauge
convention -- see ``_tb_utils.py``'s module docstring)::

    H(kx, ky) = [[eps1 + 2*t1*(cos(2*pi*kx) + cos(2*pi*ky)),  t12                                    ],
                 [t12,                                        eps2 + 2*t2*(cos(2*pi*kx) + cos(2*pi*ky))]]

With eps1 != eps2 and a non-zero t12, the two bands hybridize but never
become degenerate (a real symmetric 2x2 matrix is degenerate only where
both the diagonal difference *and* the off-diagonal vanish simultaneously,
and t12 is a non-zero constant here) -- a simple, robust two-band
insulator. 2 orbitals -> 2 Wannier functions -- since nothing is truncated
(``num_wann == num_bands``), the maximally localized Wannier functions are
*exactly* the two lattice sites themselves, zero spread (see
``_tb_utils.py``'s module docstring for why this is the correct,
unavoidable answer, not a limitation of this example). Trial projections
are real-space Gaussians of different widths centred on each site --
deliberately not a "perfect" starting guess, so this still shows the CG
minimisation doing genuine work to get there.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _tb_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for wannier90, if not pip-installed
from _tb_utils import build_overlaps, initial_spread, monkhorst_pack, report  # noqa: E402

import wannier90  # noqa: E402

NUM_ORBITALS = 2
EPS1, EPS2 = -1.5, 1.5
T1, T2 = 0.4, -0.4
T12 = 0.3


def hamiltonian_k(k_frac: np.ndarray) -> np.ndarray:
    kx, ky = k_frac[0], k_frac[1]
    cos_sum = np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
    return np.array([
        [EPS1 + 2 * T1 * cos_sum, T12],
        [T12, EPS2 + 2 * T2 * cos_sum],
    ], dtype=complex)


def main():
    mp_grid = np.array([8, 8, 1], dtype=np.int32)
    a = 3.0  # square lattice constant (Ang)
    real_lattice = np.diag([a, a, 15.0])  # non-periodic z, just needs to be "large"
    kpt_latt = monkhorst_pack(mp_grid)

    orbital_positions_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])  # sites A (corner), B (centre)
    atoms_cart = orbital_positions_frac @ real_lattice.T

    setup_result = wannier90.setup(
        "square_lattice", mp_grid, kpt_latt, real_lattice, NUM_ORBITALS,
        ["A", "B"], atoms_cart,
        win_keywords={"num_wann": NUM_ORBITALS, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )

    M_matrix, A_matrix, eigenvalues = build_overlaps(
        hamiltonian_k, NUM_ORBITALS, kpt_latt, setup_result.nnlist,
        orbital_positions_frac=orbital_positions_frac,
        trial_positions_frac=orbital_positions_frac,  # trial orbitals centred on the atoms
        trial_widths=[0.25, 0.9], periodic_dims=[0, 1],  # deliberately different widths -- see _tb_utils
    )
    num_kpts = kpt_latt.shape[1]
    print(f"Band range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}] eV ({num_kpts} k-points)")

    run_result = wannier90.run(
        "square_lattice", setup_result, mp_grid, kpt_latt, real_lattice, ["A", "B"], atoms_cart,
        M_matrix, A_matrix, eigenvalues, backend="python",
    )
    report(run_result, "2D square lattice", initial_omega=initial_spread(A_matrix, M_matrix, setup_result.nnlist))


if __name__ == "__main__":
    main()
