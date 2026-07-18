"""``gamma_only=True`` must raise clearly, not silently produce wrong
results.

This is a regression test for a real bug caught during development: the
general (complex-arithmetic) dis_main/wann_main code path, fed the
gamma-halved k-mesh kmesh_get already produces for gamma_only=True, looks
at first like it should reproduce the dedicated real-arithmetic
"_gamma" Fortran routines (the halving/doubling trick is exactly what
those routines rely on) -- but empirically does not: checked against
test-suite/tests/testw90_na_chain_gamma, it gives a wildly different,
non-converged spread (180.6 vs the fortran backend's 37.5). The "_gamma"
routines constrain the unitary rotations to be real (time-reversal
symmetry), a genuinely different optimization problem, not just a faster
arithmetic path through the same one -- see the comment in
``_engine.wannier_run``. Until that's ported, ``wannier_run`` must reject
``gamma_only=True`` outright.

``kmesh_get`` itself (the b-vector halving) is unaffected and still used
implicitly by other tests; only ``wannier_run``'s disentangle/wannierise
path is guarded here.
"""
import numpy as np
import pytest


def test_gamma_only_run_raises_not_implemented():
    import wannier90

    # A trivial single-k-point cubic cell is enough to exercise kmesh_get's
    # gamma_only path and reach wannier_run -- the physical content of
    # M_matrix/A_matrix doesn't matter since the guard fires before they're used.
    real_lattice = 3.0 * np.eye(3)
    kpt_latt = np.zeros((3, 1))
    atoms_cart = np.zeros((3, 1))
    num_wann = 1

    setup_result = wannier90.setup(
        "gamma", [1, 1, 1], kpt_latt, real_lattice, num_wann, ["X"], atoms_cart,
        win_keywords={"num_wann": num_wann}, gamma_only=True, backend="python",
    )

    M_matrix = np.ones((num_wann, num_wann, setup_result.nntot, 1), dtype=complex)
    A_matrix = np.ones((num_wann, num_wann, 1), dtype=complex)
    eigenvalues = np.zeros((num_wann, 1))

    with pytest.raises(NotImplementedError, match="gamma_only"):
        wannier90.run(
            "gamma", setup_result, [1, 1, 1], kpt_latt, real_lattice, ["X"], atoms_cart,
            M_matrix, A_matrix, eigenvalues, gamma_only=True,
        )
