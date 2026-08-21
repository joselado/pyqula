"""The matrix-free solver returns the same excitons as the dense one.

bsetk/iterative.py never assembles the BSE matrix -- it applies the exact
low-rank factorization of the kernel and runs a preconditioned block
LOBPCG on it, from a deterministic start. It is therefore exact up to the
eigensolver's tolerance, and the tests below pin
it against the dense solver where the dense solver can still run, against
supercell folding where it cannot, and against mesh refinement, which is
where an inadequately converged eigensolver goes wrong silently.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

NEIG = 4


def spinless_chain(stagger=0.9):
    """The spinless version of testutils.gapped_ionic_chain: two
    non-degenerate bands, so npair = nk and the mesh refinement below is
    cheap enough to run to nk=4096"""
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: stagger * (-1) ** int(round(r[0] - 0.5)))
    return h.get_multicell().get_dense()


@pytest.mark.parametrize("kernel", ["full", "direct", "exchange", "none"])
@pytest.mark.parametrize("Q", [[0., 0., 0.], [0.25, 0., 0.]])
def test_iterative_matches_dense_chain(kernel, Q):
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0, V1=0.5)
    kw = dict(V=W, Q=Q, nk=16, kernel=kernel)
    ref = h.get_bse(tda=True, **kw).get_energies(n=NEIG).real
    got = h.get_bse(tda=True, solver="iterative", neig=NEIG, **kw)
    assert np.max(np.abs(np.sort(got.get_energies().real) - np.sort(ref))) < 1e-8


def test_iterative_matches_dense_honeycomb():
    h = gapped_honeycomb(spinful=False)
    W = density_interaction(h, V1=0.6, V2=0.2)
    kw = dict(V=W, nk=8, kernel="full")
    ref = h.get_bse(tda=True, **kw).get_energies(n=NEIG).real
    got = h.get_bse(tda=True, solver="iterative", neig=NEIG, **kw)
    assert np.max(np.abs(got.get_energies().real - ref)) < 1e-8


@pytest.mark.parametrize("nk", [256, 1024, 4096])
def test_iterative_stays_accurate_as_the_mesh_is_refined(nk):
    """The regime this solver exists for, and the one an inadequate
    eigensolver fails in silently.

    Refining the mesh packs the low end of the spectrum into an ever
    denser cluster, and an UNSHIFTED diagonal preconditioner stalls on a
    plausible-looking upper bound there -- 1.7949 against the true 1.5023
    at nk=4096. iterative.py's module docstring has that measurement and
    the reason (1/(dE - min dE) blows up as the cluster densifies), and
    is also where ARPACK, which is exact and 100x faster here but cannot
    resolve degenerate multiplicities deterministically, is ruled out.
    The converged answer is mesh independent to eight digits from nk=256
    upward, so a solver drifting with nk is a solver that is not
    converging."""
    h = spinless_chain()
    W = density_interaction(h, V1=0.8)
    e = h.get_bse(V=W, nk=nk, tda=True, solver="iterative",
                  neig=1).get_energies()[0].real
    assert abs(e - 1.50233276830) < 1e-8, (nk, e)


def test_iterative_supercell_folding_past_the_dense_wall():
    """The check that survives where the dense solver cannot run.

    A 4-cell supercell at Q=0 must reproduce the base cell's excitons
    collected over the four Q that fold onto zero. Here the base cell runs
    at nk=128, whose dense BSE matrix (npair = 512 per Q) the dense solver
    would still manage, but the supercell's own npair = 2048 at nk=32 is
    what the folding compares against, and the whole comparison is done
    matrix-free."""
    U, V1 = 1.0, 0.4
    nsuper, nk = 4, 128

    def lowest(h, nkk, Q):
        W = density_interaction(h, U=U, V1=V1)
        b = h.get_bse(V=W, Q=Q, nk=nkk, kernel="full", tda=True,
                      solver="iterative", neig=6)
        return b.get_energies().real

    base = gapped_ionic_chain(nsuper=1)
    sup = gapped_ionic_chain(nsuper=nsuper)
    folded = np.sort(np.concatenate(
        [lowest(base, nk, [i / nsuper, 0., 0.]) for i in range(nsuper)]))
    got = np.sort(lowest(sup, nk // nsuper, [0., 0., 0.]))
    # compare the lowest few, which are the ones both calculations
    # actually converged
    assert np.max(np.abs(got[0:4] - folded[0:4])) < 1e-7, (got[0:4],
                                                           folded[0:4])


def test_full_problem_is_refused():
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0)
    with pytest.raises(ValueError, match="Tamm-Dancoff"):
        h.get_bse(V=W, nk=8, solver="iterative", tda=False)


def test_beyond_the_dense_memory_wall():
    """A mesh whose dense BSE matrix check_memory refuses, solved anyway"""
    h = gapped_honeycomb(spinful=False)
    W = density_interaction(h, V1=0.6)
    with pytest.raises(MemoryError):
        h.get_bse(V=W, nk=64, tda=True, max_memory=0.5)
    b = h.get_bse(V=W, nk=64, tda=True, solver="iterative", neig=2)
    es = b.get_energies().real
    assert len(es) == 2 and np.all(np.isfinite(es))
    # and it is below the independent-particle gap, i.e. actually bound
    assert es[0] < np.min(b.pairs.dE) + 1e-9
