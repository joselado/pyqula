"""The spin-block oracle for topology.operator_berry.

Every other test of this function pins a number the code itself produced.
This one does not: for a Hamiltonian whose sz commutes with H, the Bloch
matrix is block diagonal in spin and the operator-weighted Berry curvature
must decompose exactly,

    operator_berry(h, k, operator=None) = Omega_up(k) + Omega_dn(k)
    operator_berry(h, k, operator=sz)   = Omega_up(k) - Omega_dn(k)
    operator_berry(h, k, operator=P_s)  = Omega_s(k)

where Omega_s is the ordinary Berry curvature of the spin-s block computed
*on its own*, as a separate spinless Hamiltonian, through the unrelated
Wilson-loop code path (topology.berry_curvature). Kane-Mele makes that
decomposition exact: its spin-up block is literally the spinless Haldane
model with +t2 and its spin-down block the one with -t2 (asserted below,
so the oracle is not taken on trust).

Because the reference is an absolute curvature and not a ratio, this also
pins operator_berry's normalization: the b*pi*pi*8 factor topology.py
applies supplies the Kubo formula's 2 and the (2*pi)^2 that
multicell.derivative omits (it differentiates exp(i*k.R) while the Bloch
phase is exp(i*2*pi*k.R) -- see current.hk_derivative), and any drift in
either would show up here as a constant factor.
"""
import numpy as np
import pytest

from pyqula import geometry, operators, topology

T2 = 0.1
MASS = 0.15
KS = [[0.17, 0.41], [0.28, 0.30], [0.05, 0.90], [0.44, 0.11]]


def _kane_mele():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(T2)
    h.add_sublattice_imbalance(MASS)
    return h


def _haldane_block(t2):
    """The spinless Hamiltonian the corresponding Kane-Mele spin block
    is expected to be."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.add_sublattice_imbalance(MASS)
    return h


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def test_kane_mele_spin_blocks_are_the_haldane_models_the_oracle_assumes():
    """Validate the oracle itself before using it."""
    h = _kane_mele()
    sz = _dense(operators.get_sz(h))
    k = np.array([0.17, 0.41, 0.])
    hk = _dense(h.get_hk_gen()(k))
    assert np.max(np.abs(hk@sz - sz@hk)) < 1e-12  # sz is conserved
    up = np.where(np.diagonal(sz).real > 0)[0]
    dn = np.where(np.diagonal(sz).real < 0)[0]
    for (idx, t2) in [(up, T2), (dn, -T2)]:
        block = hk[np.ix_(idx, idx)]
        ref = _dense(_haldane_block(t2).get_hk_gen()(k))
        assert np.max(np.abs(block - ref)) < 1e-12


@pytest.mark.parametrize("k", KS)
def test_operator_berry_decomposes_into_the_two_spin_blocks(k):
    """None -> Omega_up+Omega_dn, sz -> Omega_up-Omega_dn, against the
    Wilson-loop curvature of each block as an independent Hamiltonian."""
    h = _kane_mele()
    sz = _dense(operators.get_sz(h))
    up = topology.berry_curvature(_haldane_block(T2), np.array(k), dk=1e-3)
    dn = topology.berry_curvature(_haldane_block(-T2), np.array(k), dk=1e-3)
    total = topology.operator_berry(h, k=k, operator=None)
    projected = topology.operator_berry(h, k=k, operator=sz)
    scale = np.abs(up) + np.abs(dn) # the two blocks nearly cancel in `total`
    assert np.abs(total - (up + dn)) < 1e-3*scale, (total, up + dn)
    assert np.abs(projected - (up - dn)) < 1e-3*scale, (projected, up - dn)


@pytest.mark.parametrize("k", KS)
def test_operator_berry_with_a_spin_projector_gives_one_block(k):
    """P_up = (1+sz)/2 must isolate the spin-up block exactly."""
    h = _kane_mele()
    sz = _dense(operators.get_sz(h))
    one = np.identity(sz.shape[0])
    for (proj, t2) in [((one + sz)/2., T2), ((one - sz)/2., -T2)]:
        ref = topology.berry_curvature(_haldane_block(t2), np.array(k), dk=1e-3)
        got = topology.operator_berry(h, k=k, operator=proj)
        assert np.abs(got - ref) < 1e-3*(np.abs(ref) + 1.), (got, ref)


def test_operator_berry_accepts_an_operator_object():
    """h.get_operator("sz") is the canonical way to name an operator in
    this library, and is what spin_chern's docstring and the operator_berry
    comments claim to support -- it used to raise, because
    ndarray @ Operator has no __rmatmul__ to fall back on."""
    h = _kane_mele()
    k = [0.17, 0.41]
    raw = topology.operator_berry(h, k=k,
            operator=_dense(operators.get_sz(h)))
    obj = topology.operator_berry(h, k=k, operator=h.get_operator("sz"))
    assert np.isclose(obj, raw, rtol=1e-10, atol=1e-12)


def test_operator_berry_accepts_a_sparse_hamiltonian():
    """np.asarray of a scipy sparse matrix is a 0-d object array, so the
    whole operator-Berry family used to die on any is_sparse Hamiltonian.
    The answer cannot depend on the storage format."""
    h = _kane_mele()
    hs = h.copy()
    hs.turn_sparse()
    assert hs.is_sparse
    sz = _dense(operators.get_sz(h))
    for k in KS:
        dense_b = topology.operator_berry(h, k=k, operator=sz)
        sparse_b = topology.operator_berry(hs, k=k, operator=sz)
        assert np.isclose(sparse_b, dense_b, rtol=1e-8, atol=1e-10)


@pytest.mark.slow
def test_spin_chern_is_invariant_under_supercell_folding():
    """h.get_supercell(...) produces a sparse Hamiltonian, which is exactly
    the case that used to be impossible to compute. Folding the Brillouin
    zone cannot change a topological invariant, so the spin Chern number of
    the 2x1 supercell must be the one of the unit cell."""
    h = _kane_mele()
    c0 = topology.spin_chern(h, nk=16)
    hs = h.get_supercell([2, 1, 1])
    hs.turn_sparse() # exercise the sparse branch whatever get_supercell does
    c1 = topology.spin_chern(hs, nk=16)
    assert np.isclose(c1, c0, atol=5e-2), (c0, c1)
