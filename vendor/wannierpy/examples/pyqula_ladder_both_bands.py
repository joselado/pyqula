"""Wannierize *both* bands of the same pyqula ladder as
``pyqula_ladder.py`` -- contrast this with that example, which
pre-selects only the lowest band.

Here ``num_wann == num_bands == 2``: the full local Hilbert space (both
rung sites), no disentanglement, no ``exclude_bands``-like pre-selection.
That puts this squarely in the "complete, untruncated manifold" case from
``_tb_utils.py``'s module docstring: the maximally localized Wannier
functions are provably *exactly* the two original rung sites, each with
zero spread -- unlike ``pyqula_ladder.py``'s single-band result (which
also converges to zero spread, but for a genuine physical reason specific
to that band's symmetry-protected flatness), this zero is a general
algebraic fact about keeping a full manifold, true regardless of the
Hamiltonian. As in the 1D/2D/3D hard-coded examples, this is still shown
as a real (if modest) demonstration: seed the calculation with
differently-sized Gaussian trial orbitals on each rung (not the "obvious"
but degenerate exact-eigenbasis trial -- see the module docstring for why
that shows nothing), report the spread *before* any CG minimisation, and
let it converge down to that exact answer.

Same rung-axis caveat as ``pyqula_ladder.py``: the ladder is periodic only
along its length, so there is no b-vector across the rungs for the centre
formula to resolve a y-position from -- both reported centres' y
coordinate is always exactly 0, regardless of which rung(s) each Wannier
function actually sits on.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _tb_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for wannier90, if not pip-installed
from _tb_utils import build_overlaps, initial_spread, monkhorst_pack, report  # noqa: E402

import wannier90  # noqa: E402


def _import_pyqula():
    """``pyqula`` isn't a dependency of this package -- try a plain
    import first (works if it's pip-installed, e.g. ``pip install -e .``
    from a pyqula checkout), else fall back to the ``PYQULA_SRC``
    environment variable pointing at a checkout's ``src/`` directory
    (mirrors ``scripts/build_fortran_extension.py``'s ``WANNIER90_SRC``)."""
    try:
        import pyqula  # noqa: F401
        return
    except ImportError:
        pass
    env_root = os.environ.get("PYQULA_SRC")
    if env_root and (Path(env_root) / "pyqula").exists():
        sys.path.insert(0, env_root)
        return
    raise RuntimeError(
        "Could not import pyqula (https://github.com/joselado/pyqula). Either 'pip install -e .' "
        "it, or set PYQULA_SRC=/path/to/pyqula/src (the directory containing the pyqula/ package)."
    )


def main():
    _import_pyqula()
    from pyqula import geometry

    g = geometry.ladder()
    h = g.get_hamiltonian(has_spin=False)  # spinless, default first-neighbour hopping
    hk_gen = h.get_hk_gen()

    def hamiltonian_k(k_frac: np.ndarray) -> np.ndarray:
        return hk_gen([k_frac[0], 0.0, 0.0])  # pyqula's 1D generator only reads component 0

    num_orbitals = 2  # 2 rung sites, no spin

    mp_grid = np.array([24, 1, 1], dtype=np.int32)
    real_lattice = np.diag([1.0, 15.0, 15.0])  # matches pyqula's a1 = (1,0,0); y/z non-periodic
    kpt_latt = monkhorst_pack(mp_grid)

    orbital_positions_frac = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])  # the two rung sites
    atoms_cart = orbital_positions_frac @ real_lattice.T

    setup_result = wannier90.setup(
        "pyqula_ladder2", mp_grid, kpt_latt, real_lattice, num_orbitals,
        ["X", "X"], atoms_cart,
        win_keywords={"num_wann": num_orbitals, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )

    M_matrix, A_matrix, eigenvalues = build_overlaps(
        hamiltonian_k, num_orbitals, kpt_latt, setup_result.nnlist,
        orbital_positions_frac=orbital_positions_frac,
        trial_positions_frac=orbital_positions_frac,  # trial orbitals centred on the two rungs
        trial_widths=[0.25, 0.9], periodic_dims=[0],  # deliberately different widths -- see _tb_utils
    )
    print(f"Band range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}] ({mp_grid[0]} k-points)")

    run_result = wannier90.run(
        "pyqula_ladder2", setup_result, mp_grid, kpt_latt, real_lattice, ["X", "X"], atoms_cart,
        M_matrix, A_matrix, eigenvalues, backend="python",
    )
    report(run_result, "pyqula ladder (both bands)",
           initial_omega=initial_spread(A_matrix, M_matrix, setup_result.nnlist))


if __name__ == "__main__":
    main()
