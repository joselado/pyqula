"""Supercell folding. The same physical system described with a unit cell
of n base cells and nk/n k-points must give exactly the same excitons as
the base cell does, collected over every center-of-mass momentum that
folds onto Q=0. This is the test of the momentum bookkeeping: the direct
kernel's W(k-k') dependence and the finite-Q machinery are reached through
a completely different set of momentum differences in the two
descriptions, so an error in either shows up as a mismatch.

It is also the test that separates onsite from extended interactions -- an
error in which momentum the antiresonant exchange block is evaluated at is
invisible for a pure Hubbard U (where W(Q) is momentum independent) and
only shows up once V1 is switched on."""
import numpy as np
import pytest

from testutils import gapped_ionic_chain
from pyqula.bsetk.interaction import density_interaction

NK = 8


def _spectrum(nsuper, nk, Q, U, V1, kernel):
    h = gapped_ionic_chain(nsuper=nsuper)
    W = density_interaction(h, U=U, V1=V1)
    b = h.get_bse(V=W, Q=Q, nk=nk, kernel=kernel)
    return np.sort(b.get_energies().real)


def _folded_reference(nsuper, U, V1, kernel):
    """The base-cell excitons at every Q with nsuper*Q = 0 modulo 1"""
    qs = [i / nsuper for i in range(nsuper)]
    return np.sort(np.concatenate(
        [_spectrum(1, NK, [q, 0., 0.], U, V1, kernel) for q in qs]))


@pytest.mark.parametrize("U,V1", [(1.0, 0.0), (0.0, 0.4), (1.0, 0.4)])
@pytest.mark.parametrize("nsuper", [2, 4])
def test_supercell_folding(U, V1, nsuper):
    sup = _spectrum(nsuper, NK // nsuper, [0., 0., 0.], U, V1, "full")
    ref = _folded_reference(nsuper, U, V1, "full")
    assert len(sup) == len(ref)
    assert np.max(np.abs(sup - ref)) < 1e-9


@pytest.mark.parametrize("kernel", ["direct", "exchange"])
def test_supercell_folding_per_kernel_term(kernel):
    """Both kernel terms must fold on their own, so a failure points at
    which one is wrong instead of only at the total."""
    sup = _spectrum(4, 2, [0., 0., 0.], 0.0, 0.4, kernel)
    ref = _folded_reference(4, 0.0, 0.4, kernel)
    assert np.max(np.abs(sup - ref)) < 1e-9
