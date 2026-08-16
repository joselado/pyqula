"""The central correctness check: the analytic Kubo superfluid weight must
reproduce a brute-force central finite difference of the grand potential
with respect to the twist.

The analytic formula is the *exact* q-derivative of the very sum the finite
difference evaluates, k-point by k-point, so the agreement does not involve
any BZ convergence: it is just as sharp at nk=6 as at nk=100.  That keeps
these tests fast while still being a stringent test of the whole chain
(mask, current.hk_derivative normalisation, divided differences, the
factor 1/2 of the doubled Nambu basis, the reduced->Cartesian conversion).
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import superfluidweight as sw


def _square(mu=-0.7, delta=0.3):
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(mu)
    h.add_swave(delta)
    return h


def _honeycomb_rashba():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.4)
    h.add_rashba(0.3)
    h.add_swave(0.35)
    return h


def _square_pwave():
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_pairing(mode="pwave", delta=0.3, d=[0., 0., 1.])
    h.add_swave(0.0)
    return h


def _chain():
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(-0.5)
    h.add_swave(0.3)
    return h


def _cubic():
    h = geometry.cubic_lattice().get_hamiltonian()
    h.add_onsite(-1.0)
    h.add_swave(0.5)
    return h


def _triangular_zeeman():
    h = geometry.triangular_lattice().get_hamiltonian()
    h.add_onsite(-1.0)
    h.add_zeeman([0., 0., 0.2])
    h.add_swave(0.6)
    return h


@pytest.mark.parametrize("build,T", [
    (_square, 0.0),
    (_square, 0.15),
    (_honeycomb_rashba, 0.05),   # spin-orbit coupled, spin-split bands
    (_square_pwave, 0.0),        # non-local, non-uniform (p-wave) pairing
    (_chain, 0.0),               # dimensionality 1
    (_cubic, 0.0),               # dimensionality 3
    (_triangular_zeeman, 0.05),  # non-orthogonal lattice vectors, broken TRS
    ])
def test_analytic_superfluid_weight_matches_the_twist_finite_difference(
        build, T):
    h = build()
    nk = 6
    da = sw.superfluid_weight(h, nk=nk, T=T)
    # dq small enough that the O(dq^2) truncation of the stencil, not the
    # formula, sets the residual
    df = sw.superfluid_weight_finite_difference(h, nk=nk, T=T, dQ=3e-4)
    scale = max(np.max(np.abs(da)), 1e-10)
    assert np.max(np.abs(da-df))/scale < 1e-3, (da, df)


def test_finite_difference_converges_to_the_analytic_result():
    """Refining the finite-difference step must drive the discrepancy down
    like dq^2, i.e. the analytic formula is the exact derivative and not
    merely close to it."""
    h = _square()
    nk = 6
    da = sw.superfluid_weight(h, nk=nk, T=0.1)[0, 0]
    errs = []
    for dq in [4e-3, 2e-3, 1e-3]:
        df = sw.superfluid_weight_finite_difference(h, nk=nk, T=0.1, dQ=dq)
        errs.append(abs(df[0, 0]-da))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0]/errs[2] > 8.   # ~16 for a clean second-order stencil


def test_multiorbital_and_supercell_models_also_agree():
    """Multi-orbital cells are where the bond vectors of the twist matter,
    so the oracle is run there too -- on a honeycomb lattice and on a
    supercell of it, whose cell holds eight sites."""
    for (h, nk) in [(_honeycomb_rashba(), 6),
                    (_honeycomb_supercell(), 4)]:
        da = sw.superfluid_weight(h, nk=nk, T=0.05)
        df = sw.superfluid_weight_finite_difference(h, nk=nk, T=0.05,
                                                    dQ=3e-4)
        assert np.max(np.abs(da-df))/np.max(np.abs(da)) < 1e-3, (da, df)


def _honeycomb_supercell():
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian()
    h.add_onsite(0.4)
    h.add_swave(0.4)
    return h
