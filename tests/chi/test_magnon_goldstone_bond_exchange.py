import numpy as np
import pytest

from pyqula import geometry
from pyqula.meanfield import VJinteraction
from pyqula.chitk.spinchi import V2U_matrix, V2K_matrix, replicateU, \
    _full_spin_operators
from pyqula.chitk.rpa import build_ops_projectors, interaction_at_q, \
    _chi_ops_matrix_vectorized


def test_v2k_matrix_reduces_to_v2u_matrix_for_onsite_only_interaction():
    """V2K_matrix must agree exactly with -V2U_matrix whenever V has no
    same-spin (up-up/down-down) component, e.g. a plain onsite Hubbard U --
    the case both extractions already agreed on before this fix."""
    N = 3
    V = np.zeros((2*N, 2*N), dtype=np.complex128)
    diag_vals = [0.5, 1.5, -0.7]
    for i, u in enumerate(diag_vals):
        V[2*i, 2*i+1] = u/2.
        V[2*i+1, 2*i] = u/2.
    assert np.allclose(V2K_matrix(V), -V2U_matrix(V))


def test_v2k_matrix_captures_same_spin_component_v2u_matrix_drops():
    """For a matrix built the way scftk.spinspin._build_v encodes
    a direct Sz_i Sz_j bond term (+1/4 same-spin, -1/4 cross-spin), V2K_matrix
    must recover the full bond coefficient, while V2U_matrix (cross-spin
    only) recovers only half of it -- this asymmetry is the root cause fixed
    here (see V2K_matrix's docstring)."""
    N = 2
    b = 2.0  # raw _build_v-style bond amplitude between orbitals 0 and 1
    V = np.zeros((2*N, 2*N), dtype=np.complex128)
    V[0, 2] = b/4.; V[1, 3] = b/4.    # up-up, down-down (orbital 0 <-> 1)
    V[0, 3] = -b/4.; V[1, 2] = -b/4.  # up-down, down-up
    K = V2K_matrix(V)
    U = V2U_matrix(V)
    assert np.isclose(K[0, 1], b)
    assert np.isclose(U[0, 1], -b/2.)  # only half of K, and opposite sign


def _uniform_goldstone_kernel(h, nk=200):
    """The RPA kernel 1-V(q=0)*chi0(q=0,w=0) projected onto the uniform
    (Sx_0+Sx_1)/sqrt(2) global-spin-rotation generator of a 2-site unit
    cell -- the direction the Goldstone theorem actually applies to (not
    the smallest eigenvalue of the full kernel, and not a staggered
    combination -- see the module docstring below).

    Builds the vertex inline (V2K_matrix/replicateU, the same formula
    _full_spin_U uses) instead of calling _full_spin_U(h) directly, since
    h.V here is deliberately non-onsite (a VJinteraction bichain with J1
    bond exchange) -- chitk.spinchi._require_onsite_only_V now rejects
    that at the Hamiltonian-API level (get_magnon_bands/get_spinchi_full),
    because non-onsite spin-channel RPA isn't properly verified in
    general. This test intentionally bypasses that guard to keep
    regression coverage for the underlying vertex-extraction math
    (V2K_matrix) that the guard's caveat is about -- the FM-saturation
    check below is exactly the evidence that motivated fixing V2K_matrix
    in the first place, even though the public API no longer exposes this
    combination directly."""
    Ss = _full_spin_operators(h)
    U = {d: 2*replicateU(V2K_matrix(m), n=3) for d, m in h.V.items()}
    pAs, pBs = build_ops_projectors(h, Ss)
    q = [0., 0., 0.]
    es, chis = _chi_ops_matrix_vectorized(h, ops=Ss, pAs=pAs, pBs=pBs, q=q,
                                           energies=np.array([0.0]),
                                           delta=1e-4, nk=nk)
    Vq = interaction_at_q(U, h, q)
    iden = np.identity(Vq.shape[0], dtype=np.complex128)
    kernel = iden - Vq@chis[0]
    v = np.zeros(6, dtype=np.complex128)
    v[0] = 1.; v[1] = 1.  # ops ordering [Sx0,Sx1,Sy0,Sy1,Sz0,Sz1]
    v = v/np.sqrt(2)
    return complex(np.asarray(v.conj()@kernel@v).ravel()[0])


@pytest.mark.slow
def test_goldstone_residual_shrinks_with_bond_exchange_in_saturated_limit():
    """Regression test for a real coefficient bug in _full_spin_U's
    handling of a direct (non-onsite) Sz_i Sz_j bond interaction: the RPA
    vertex it built from H.V used to be off by a factor of 2 for any
    interaction carrying a genuine same-spin (up-up/down-down) component,
    which V2U_matrix (built for onsite Hubbard, which has none) silently
    dropped -- see V2K_matrix's docstring for the derivation.

    Physical check: for a genuinely saturated ferromagnet, Hartree-Fock
    becomes asymptotically EXACT as the coupling grows (same Nagaoka-type
    argument that makes the onsite-U Goldstone residual shrink to 0 as
    U->infinity, see tests/chi/test_spinchi_rotation.py), so the uniform-
    rotation-direction kernel eigenvalue at q=0 must shrink towards 0 as
    |J1| grows deep into saturation. Before the fix this residual sat flat
    at ~0.5, completely independent of J1 (a specific coefficient bug, not
    just approximation-quality) -- this test would have failed against
    that old behavior."""
    g = geometry.bichain()

    def residual(J1):
        h = g.get_hamiltonian()
        v = np.array([0., 0., 1.])
        h.add_exchange(0.5*v)
        mf = h.copy()
        mf.add_exchange(0.5*v)
        scf = VJinteraction(h, J1=J1, mf=mf, nk=200, mix=0.2,
                             maxerror=1e-8, maxite=1000, filling=0.5)
        assert scf.converged
        return abs(_uniform_goldstone_kernel(scf.hamiltonian, nk=200))

    r_weak = residual(-6.)
    r_strong = residual(-20.)
    assert r_strong < 0.7*r_weak, \
        f"expected the Goldstone residual to shrink deep into saturation: {r_weak} -> {r_strong}"
    assert r_strong < 0.1, f"residual still far from vanishing: {r_strong}"
