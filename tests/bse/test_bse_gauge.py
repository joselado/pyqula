"""The Bloch gauge has no physical content, and it decides the rank.

Two independent things are pinned here.

1. Applying a gauge fix changes NO exciton energy. The gauge is a
   block-diagonal unitary on the pair index, so the whole BSE matrix is
   conjugated by a unitary and the spectrum is invariant. If this ever
   fails, the gauge code is wrong -- it is never a modelling choice.

2. Under a gauge fix the quantics tensor-train rank of the kernel's
   factors SATURATES as the mesh is refined, while in the raw eigh gauge
   it grows with the mesh. That is the property the whole quantics solver
   rests on, and it is cheap enough to assert directly.
"""
import numpy as np
import pytest

from testutils import gapped_ionic_chain, gapped_honeycomb
from pyqula.bsetk.interaction import density_interaction, bare_interaction
from pyqula.bsetk.pairbasis import PairBasis
from pyqula.bsetk.factorize import KernelFactorization
from pyqula.bsetk.gauge import fix_gauge, default_trials


def _ttranks(v, nbit, tol=1e-6):
    """Maximum tensor-train rank of a vector of length 2**nbit"""
    a = np.asarray(v, dtype=np.complex128).reshape([2] * nbit)
    nrm = np.linalg.norm(a)
    M = a.reshape(1, -1)
    r, out = 1, []
    for i in range(nbit - 1):
        M = M.reshape(r * 2, -1)
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        r = max(int(np.sum(s > tol * nrm)), 1)
        out.append(r)
        M = np.diag(s[:r]) @ vh[:r]
    return max(out)


def _max_factor_rank(h, W, nk, gauge, nbit):
    pb = PairBasis(h, Q=[0., 0., 0.], nk=nk, gauge=gauge)
    f = KernelFactorization(pb, W, Wx=W, kernel="full", block="A")
    # weight each factor by its coefficient, so a numerically negligible
    # term cannot dominate a relative-tolerance rank
    w = np.sqrt(np.abs(f.coefs))
    return max(_ttranks(w[t] * f.left[t], nbit) for t in range(f.nterm))


@pytest.mark.parametrize("gauge", ["phase", "projection"])
@pytest.mark.parametrize("Q", [[0., 0., 0.], [0.25, 0., 0.]])
def test_gauge_does_not_change_the_spectrum_chain(gauge, Q):
    h = gapped_ionic_chain()
    W = density_interaction(h, U=1.0, V1=0.5)
    kw = dict(V=W, Q=Q, nk=8, kernel="full")
    ref = np.sort(h.get_bse(**kw).get_energies().real)
    got = np.sort(h.get_bse(gauge=gauge, **kw).get_energies().real)
    assert np.max(np.abs(got - ref)) < 1e-11, (gauge, Q)


@pytest.mark.parametrize("gauge", ["phase", "projection"])
def test_gauge_does_not_change_the_spectrum_honeycomb(gauge):
    h = gapped_honeycomb(spinful=False)
    W = density_interaction(h, V1=0.6, V2=0.2)
    kw = dict(V=W, nk=6, kernel="full")
    ref = np.sort(h.get_bse(**kw).get_energies().real)
    got = np.sort(h.get_bse(gauge=gauge, **kw).get_energies().real)
    assert np.max(np.abs(got - ref)) < 1e-11, gauge


def test_gauge_is_a_unitary_on_each_subspace():
    """The rotation must stay inside the band subspace it is given --
    otherwise it would mix valence into conduction and the pair basis
    would no longer describe the same excitations"""
    h = gapped_honeycomb(spinful=True)
    pb = PairBasis(h, nk=4)
    groups = [pb.vbands, pb.cbands]
    for mode in ("phase", "projection"):
        ck = fix_gauge(pb.ck, groups, mode=mode,
                       trials=default_trials(pb.ck, groups))
        for grp in groups:
            for ik in range(len(pb.kpoints)):
                # the subspace projector is invariant
                P0 = pb.ck[ik][grp].conj().T @ pb.ck[ik][grp]
                P1 = ck[ik][grp].conj().T @ ck[ik][grp]
                assert np.max(np.abs(P0 - P1)) < 1e-12, (mode, grp, ik)


def _spinless_ionic_chain(stagger=0.9):
    """The spinless version of testutils.gapped_ionic_chain, whose two
    bands are NON-degenerate -- which is what a phase fix needs"""
    from pyqula import geometry
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: stagger * (-1) ** int(round(r[0] - 0.5)))
    return h.get_multicell().get_dense()


def test_phase_gauge_saturates_the_rank_on_nondegenerate_bands():
    """The raw gauge is incompressible, the phase-fixed one is not.

    Measured, max factor rank at tolerance 1e-6 on this model:
      nk    32  128  512
      raw    4    8   16     (doubling with the mesh)
      phase  4    7    7     (saturated)
    """
    h = _spinless_ionic_chain()
    W = bare_interaction(h, V=density_interaction(h, V1=0.8))
    nks = (128, 512)
    raw = [_max_factor_rank(h, W, nk, None, int(np.log2(nk))) for nk in nks]
    fixed = [_max_factor_rank(h, W, nk, "phase", int(np.log2(nk)))
             for nk in nks]
    assert raw[1] >= 2 * raw[0], raw           # grows with the mesh
    assert fixed[1] <= fixed[0], fixed         # saturates
    assert fixed[1] < raw[1], (fixed, raw)


def test_projection_gauge_is_the_one_that_handles_degenerate_bands():
    """A spinful model with no spin-orbit coupling is two-fold degenerate
    everywhere, and what is arbitrary inside a degenerate subspace is a
    unitary, not a phase -- so fixing phases cannot help there at all,
    while projecting onto trial orbitals does.

    Measured on the spinful ionic chain, max factor rank at 1e-6:
      nk     32  128  512
      raw     8   16   32
      phase   8   16   32    (identical to raw: it fixes nothing here)
      proj    4    7    7    (saturated)

    This is why "projection", not "phase", is the default gauge of the
    quantics solver.
    """
    h = gapped_ionic_chain()
    W = bare_interaction(h, V=density_interaction(h, U=1.0, V1=0.5))
    nks = (128, 512)
    nbit = [int(np.log2(nk)) + 2 for nk in nks]  # npair = 4*nk
    raw = [_max_factor_rank(h, W, nk, None, nb) for nk, nb in zip(nks, nbit)]
    phase = [_max_factor_rank(h, W, nk, "phase", nb)
             for nk, nb in zip(nks, nbit)]
    proj = [_max_factor_rank(h, W, nk, "projection", nb)
            for nk, nb in zip(nks, nbit)]
    assert phase == raw, (phase, raw)          # a phase fix is no help
    assert proj[1] <= proj[0], proj            # projection saturates
    assert proj[1] < raw[1] / 2, (proj, raw)
