"""Wannierize the lowest band of a two-leg ladder built with
`pyqula <https://github.com/joselado/pyqula>`_ (a separate tight-binding
package), instead of a hard-coded Hamiltonian -- this is the interop demo:
``h.get_hk_gen()`` from an arbitrary pyqula Hamiltonian is *already* a
``hamiltonian_k(k_frac) -> Hermitian ndarray`` in the periodic-gauge
convention ``_tb_utils.build_overlaps`` expects (pyqula's own Bloch
generator only ever inserts phases via lattice-vector direction, see
``htk/bloch.py``'s ``exp(1j*2*pi*d.k)``), so it plugs in directly.

``geometry.ladder()`` (``src/pyqula/geometry.py``) is two coupled 1D
chains -- sites at fractional (x, y) = (0, -0.5) and (0, +0.5), hopping 1
along each leg (x) and 1 across each rung (y), periodic only along x. With
uniform hoppings on both legs, ``get_hamiltonian()``'s default first-
neighbour model is exactly the textbook symmetric ladder::

    H(kx) = [[2*cos(2*pi*kx),        1        ],
             [       1        , 2*cos(2*pi*kx)]]

"The lowest band, no disentanglement": rather than Wannierizing all 2
bands (which -- see ``_tb_utils.py``'s module docstring -- would trivially
collapse to zero spread for an unrelated, purely algebraic reason), this
pre-selects just the lower band via ``build_overlaps``'s ``band_indices``
(the manual equivalent of Wannier90's ``exclude_bands``) and Wannierizes
*that* alone: ``num_bands == num_wann == 1``, genuinely no disentanglement
needed.

What comes out, and why it's *still* exactly zero spread -- this time for
a real physical reason, not an algebraic one: the ladder's leg-exchange
symmetry means ``inter`` (leg hopping) is proportional to the identity
in the rung basis, so it only ever shifts the bonding/antibonding rung
combinations rigidly in energy -- it never mixes them. The lowest band's
eigenvector is therefore the *exact same* bonding combination,
``(1, 1)/sqrt(2)`` up to an arbitrary phase, at *every* k-point. A Bloch
state with a perfectly k-independent (unwinding) character Fourier
transforms to a Wannier function confined to a single unit cell -- an
exactly flat, "molecular-orbital-like" band, not something Wannier90 needs
to iterate towards. Checked directly below by trying several different
(deliberately not pre-aligned) trial vectors: all of them already give
zero spread before any CG iteration runs.

One more thing worth being upfront about: the reported Wannier centre's
rung (y) coordinate is always exactly 0 here, but that's *not* actually
resolving the bonding character spatially -- a ladder is periodic only
along x (``mp_grid = (N, 1, 1)``), so there is no b-vector along y for the
finite-difference centre formula to extract *any* y-position information
from at all (the same fundamental limitation as ``0d_molecule.py``'s
single-k-point case, applied to one direction instead of all three): a
maximally *asymmetric* (all rung-site-0) trial would report the same
y = 0 despite being a physically different state. Don't read anything
physical into it.
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

    num_orbitals = 2  # 2 rungs sites, no spin

    mp_grid = np.array([24, 1, 1], dtype=np.int32)
    real_lattice = np.diag([1.0, 15.0, 15.0])  # matches pyqula's a1 = (1,0,0); y/z non-periodic
    kpt_latt = monkhorst_pack(mp_grid)

    orbital_positions_frac = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])  # the two rung sites
    atoms_cart = orbital_positions_frac @ real_lattice.T

    setup_result = wannier90.setup(
        "pyqula_ladder", mp_grid, kpt_latt, real_lattice, num_orbitals,
        ["X", "X"], atoms_cart,
        win_keywords={"num_wann": 1, "num_iter": 200, "conv_tol": 1e-10, "conv_window": 3},
        backend="python",
    )

    # Deliberately several different, not-pre-aligned trial vectors -- see module
    # docstring for why every one of them already lands on zero spread.
    trials = {
        "rung site 0 only": np.array([[1.0], [0.0]], dtype=complex),
        "rung site 1 only": np.array([[0.0], [1.0]], dtype=complex),
        "asymmetric complex": np.array([[1.0], [0.3 + 0.2j]], dtype=complex),
    }
    for label, trial in trials.items():
        M_matrix, A_matrix, eigenvalues = build_overlaps(
            hamiltonian_k, num_orbitals, kpt_latt, setup_result.nnlist,
            orbital_positions_frac=orbital_positions_frac,
            band_indices=[0],  # lowest band only -- no disentanglement, see module docstring
            trial_vectors=trial,
        )
        if label == "rung site 0 only":
            print(f"Lowest band range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}] "
                  f"({mp_grid[0]} k-points)")

        run_result = wannier90.run(
            "pyqula_ladder", setup_result, mp_grid, kpt_latt, real_lattice, ["X", "X"], atoms_cart,
            M_matrix, A_matrix, eigenvalues, backend="python",
        )
        ini = initial_spread(A_matrix, M_matrix, setup_result.nnlist)
        print(f"\ntrial = {label}: Omega before CG ~ {ini:.8f} Ang^2, "
              f"after CG = {run_result.spread_total:.8f} Ang^2")

    report(run_result, "pyqula ladder (lowest band)")


if __name__ == "__main__":
    main()
