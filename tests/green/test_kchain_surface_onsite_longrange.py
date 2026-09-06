import numpy as np

from pyqula import geometry
from pyqula.greentk.kchain import (green_kchain, green_kchain_NN,
                                   green_kchain_LR)
from pyqula.htk.kchain import detect_longest_hopping, kchain_LR

# `hs`, the surface onsite matrix, used to be honoured only by
# green_kchain_NN. Beyond nearest neighbours green_kchain dispatches to
# green_kchain_NNN/green_kchain_LR, which forwarded it into
# dysonNNN/dysonLR and from there into green_renormalization's **kwargs,
# where it was silently dropped -- so the caller got the unmodified
# surface Green's function with no warning.

ARGS = dict(energy=0.2, delta=0.05, only_bulk=False)
K = [0.234, 0., 0.]


def test_long_range_path_collapses_onto_the_NN_one():
    """On a nearest-neighbour Hamiltonian the long-range solver builds a
    supercell of exactly one cell, so it must reproduce green_kchain_NN
    -- with and without hs."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.3])
    hs = np.diag([0.7, -0.4]).astype(np.complex128)
    for surface in [None, hs]:
        (gbn, gsn) = green_kchain_NN(h, k=K[0], hs=surface, **ARGS)
        (gbl, gsl) = green_kchain_LR(h, k=K, hs=surface, **ARGS)
        assert np.max(np.abs(np.array(gbn) - np.array(gbl))) < 1e-8
        assert np.max(np.abs(np.array(gsn) - np.array(gsl))) < 1e-8


def test_surface_onsite_beyond_nearest_neighbors():
    """Two independent checks on each dispatch target of green_kchain:
    passing the surface cell's *actual* onsite matrix must return the
    Green's function the decimation already gives (it is the very equation
    g_s solves), and passing a different one must move it."""
    for ts in [[1.0, 0.2], [1.0, 0.2, 0.1]]:
        h = geometry.square_lattice().get_hamiltonian(tij=ts)
        hops = kchain_LR(h, k=K)  # onsite and hoppings of the k-chain
        ons = np.array(hops[0])  # the surface cell's own onsite matrix
        (gb0, gs0) = green_kchain(h, k=K, **ARGS)
        (gb1, gs1) = green_kchain(h, k=K, hs=ons, **ARGS)
        scale = np.max(np.abs(np.array(gs0)))
        assert np.max(np.abs(np.array(gs0) - np.array(gs1))) < 1e-8 * scale
        # a different surface onsite must actually be felt
        hs = ons + np.identity(len(ons)) * 0.5
        (gb2, gs2) = green_kchain(h, k=K, hs=hs, **ARGS)
        assert np.max(np.abs(np.array(gs0) - np.array(gs2))) > 1e-2 * scale
        # but the bulk Green's function never sees it
        assert np.max(np.abs(np.array(gb0) - np.array(gb2))) < 1e-10


def test_surface_onsite_accepts_a_function_of_k():
    """green_kchain_NN takes hs either as a matrix or as a function of k;
    the long-range paths must resolve it the same way."""
    h = geometry.square_lattice().get_hamiltonian(tij=[1.0, 0.2, 0.1])
    assert detect_longest_hopping(h) > 1  # really the long-range path
    n = h.intra.shape[0]
    hs = np.identity(n) * 0.4
    (gb1, gs1) = green_kchain(h, k=K, hs=hs, **ARGS)
    (gb2, gs2) = green_kchain(h, k=K, hs=lambda k: hs, **ARGS)
    assert np.max(np.abs(np.array(gs1) - np.array(gs2))) < 1e-12


def _finite_chain_surface_green(hops, hs, energy, delta, ncells):
    """Surface Green's function of a long *finite* chain of cells, by a
    plain matrix inverse -- no decimation, no supercells, no Dyson step.
    An independent reference for what the hs branch must produce."""
    n = hops[0].shape[0]
    N = ncells * n
    ham = np.zeros((N, N), dtype=np.complex128)
    for i in range(ncells):
        for j in range(ncells):
            d = j - i
            if d == 0:
                m = hops[0]
            elif 0 < d < len(hops):
                m = hops[d]
            elif -len(hops) < d < 0:
                m = np.conjugate(np.array(hops[-d]).T)
            else:
                continue
            ham[i * n:(i + 1) * n, j * n:(j + 1) * n] = np.array(m)
    if hs is not None:
        ham[0:n, 0:n] = np.array(hs)  # only the outermost cell is modified
    ez = (energy + 1j * delta) * np.identity(N)
    return np.linalg.inv(ez - ham)[0:n, 0:n]


def test_against_a_brute_force_finite_chain():
    """The strongest check: a chain of 400 cells with a non-trivial onsite
    matrix on its first cell alone, inverted directly. At delta=0.15 that
    is converged to the semi-infinite limit, so it pins the whole path --
    the supercell layout, which block hs replaces, and the Dyson step --
    against something that shares no code with it."""
    energy, delta, ncells = 0.2, 0.15, 400
    for ts in [[1.0, 0.2], [1.0, 0.2, 0.1]]:
        h = geometry.square_lattice().get_hamiltonian(tij=ts)
        hops = kchain_LR(h, k=K)
        n = hops[0].shape[0]
        rng = np.random.RandomState(3)
        a = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
        hs = np.array(hops[0]) + (a + np.conjugate(a).T) * 0.3
        args = dict(energy=energy, delta=delta, only_bulk=False)
        (gb0, gs0) = green_kchain(h, k=K, **args)
        (gb1, gs1) = green_kchain(h, k=K, hs=hs, **args)
        ref0 = _finite_chain_surface_green(hops, None, energy, delta, ncells)
        ref1 = _finite_chain_surface_green(hops, hs, energy, delta, ncells)
        # the hs result must be as accurate as the no-hs one, which is the
        # already-trusted path -- so any excess error is the hs branch's
        assert np.max(np.abs(np.array(gs0) - ref0)) < 1e-12
        assert np.max(np.abs(np.array(gs1) - ref1)) < 1e-12
