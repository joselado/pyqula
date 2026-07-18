"""1D example: the Su-Schrieffer-Heeger (SSH) dimerized chain.

Hard-coded Bloch Hamiltonian, 2 orbitals (sites A/B) per unit cell,
intra-cell hopping t1 and inter-cell hopping t2 (periodic-gauge
convention -- see ``_tb_utils.py``'s module docstring)::

    H(kx) = [[       0        , t1 + t2*exp(-2*pi*i*kx)],
             [t1 + t2*exp(2*pi*i*kx),         0        ]]

With ``t1 != t2`` this is the textbook dimerized chain (SSH model): one
band below zero, one above, gapped everywhere except the trivial
``t1 == t2`` point. 2 orbitals -> 2 Wannier functions -- since nothing is
truncated (``num_wann == num_bands``), the maximally localized Wannier
functions are *exactly* the two tight-binding sites themselves, zero
spread (see ``_tb_utils.py``'s module docstring for why this is the
correct, unavoidable answer, not a limitation of this example). Trial
projections are real-space Gaussians of different widths centred on each
site -- deliberately not a "perfect" starting guess, so this still shows
the CG minimisation doing genuine work to get there.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _tb_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for wannier90, if not pip-installed
from _tb_utils import build_overlaps, initial_spread, monkhorst_pack, report  # noqa: E402

import wannier90  # noqa: E402

NUM_ORBITALS = 2
T1 = -1.0  # intra-cell hopping (A-B within the same cell)
T2 = -0.6  # inter-cell hopping (B in cell R to A in cell R+1)


def hamiltonian_k(k_frac: np.ndarray) -> np.ndarray:
    kx = k_frac[0]
    off_diag = T1 + T2 * np.exp(-2j * np.pi * kx)
    return np.array([
        [0.0, off_diag],
        [np.conj(off_diag), 0.0],
    ], dtype=complex)


def main():
    mp_grid = np.array([16, 1, 1], dtype=np.int32)
    a = 2.0  # unit cell length (Ang); sites A/B are half a cell apart
    real_lattice = np.diag([a, 15.0, 15.0])  # non-periodic y/z, just needs to be "large"
    kpt_latt = monkhorst_pack(mp_grid)

    orbital_positions_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # sites A, B
    atoms_cart = orbital_positions_frac @ real_lattice.T

    setup_result = wannier90.setup(
        "ssh_chain", mp_grid, kpt_latt, real_lattice, NUM_ORBITALS,
        ["A", "B"], atoms_cart,
        win_keywords={"num_wann": NUM_ORBITALS, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )

    M_matrix, A_matrix, eigenvalues = build_overlaps(
        hamiltonian_k, NUM_ORBITALS, kpt_latt, setup_result.nnlist,
        orbital_positions_frac=orbital_positions_frac,
        trial_positions_frac=orbital_positions_frac,  # trial orbitals centred on the atoms
        trial_widths=[0.25, 0.9], periodic_dims=[0],  # deliberately different widths -- see _tb_utils
    )
    print(f"Band range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}] eV "
          f"({mp_grid[0]} k-points, gap at k=pi/a since |t1| != |t2|)")

    run_result = wannier90.run(
        "ssh_chain", setup_result, mp_grid, kpt_latt, real_lattice, ["A", "B"], atoms_cart,
        M_matrix, A_matrix, eigenvalues, backend="python",
    )
    report(run_result, "1D SSH chain", initial_omega=initial_spread(A_matrix, M_matrix, setup_result.nnlist))


if __name__ == "__main__":
    main()
