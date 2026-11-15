"""constrains=["no_net_moment"] removes the site average of the spin part
of the onsite mean field, the uniform exchange field, and keeps the charge
and every site-resolved part. It is the zero net moment of the ultra
long-range method, in the form decided in
future_development/ultra_long_range.md (decision 7). It works on dense and
on csr mean fields, so on the sparse engine too; the agreement of the two
is in test_mfconstrains_sparse.py, which runs every registered name."""
import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from pyqula import geometry
from pyqula.scftk import mfconstrains
from pyqula.scftk.spinspin import VJinteraction

SX = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
SY = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
SZ = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)


def _block(q, B):
    """The onsite block of an orbital with charge q and exchange field B"""
    return q*np.eye(2) + B[0]*SX + B[1]*SY + B[2]*SZ


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def _fields(m):
    """The exchange field of every orbital of an onsite matrix"""
    m = _dense(m)
    return np.array([[np.real(np.trace(m[2*i:2*i+2, 2*i:2*i+2]@s))/2.
        for s in (SX, SY, SZ)] for i in range(m.shape[0]//2)])


def _uniform_and_staggered():
    """A mean field of the honeycomb lattice with a charge on each site, a
    uniform and a staggered exchange field in generic directions, and a
    bond term, together with the onsite block that should remain"""
    q = [0.3, -0.1]
    u, s = np.array([0.1, 0.2, 0.3]), np.array([0.2, -0.1, 0.4])
    onsite = block_diag(_block(q[0], u + s), _block(q[1], u - s))
    expected = block_diag(_block(q[0], s), _block(q[1], -s))
    np.random.seed(2)
    t = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    return {(0, 0, 0): onsite, (1, 0, 0): t, (-1, 0, 0): t.conj().T}, expected


@pytest.mark.parametrize("sparse", [False, True])
def test_only_the_uniform_exchange_field_is_removed(sparse):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    mf, expected = _uniform_and_staggered()
    if sparse: mf = {d: csr_matrix(m) for (d, m) in mf.items()}
    out = mfconstrains.enforce_constrains(mf, h, ["no_net_moment"])
    assert np.max(np.abs(_dense(out[(0, 0, 0)]) - expected)) < 1e-14
    for d in [(1, 0, 0), (-1, 0, 0)]: # a bond carries no moment of a site
        assert np.max(np.abs(_dense(out[d]) - _dense(mf[d]))) < 1e-14


def test_the_nambu_mean_field_keeps_its_pairing():
    """With Nambu the electron block is constrained, the pairing is kept and
    the hole block is rebuilt from the constrained electron block"""
    from pyqula.superconductivity import build_nambu_matrix, get_eh_sector
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.setup_nambu_spinor()
    mf, expected = _uniform_and_staggered()
    pairing = 0.2*np.kron(np.eye(2), 1j*SY)
    def nambu(m):
        return build_nambu_matrix(m, c12=pairing, c21=pairing.conj().T)
    out = mfconstrains.enforce_constrains({(0, 0, 0): nambu(mf[(0, 0, 0)])},
            h, ["no_net_moment"])[(0, 0, 0)]
    assert np.max(np.abs(_dense(get_eh_sector(out, i=0, j=0))
        - expected)) < 1e-14
    assert np.max(np.abs(_dense(get_eh_sector(out, i=0, j=1))
        - pairing)) < 1e-14
    assert np.max(np.abs(_dense(out) - _dense(nambu(expected)))) < 1e-14


def test_a_spinless_hamiltonian_is_refused():
    h = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError, match="needs a spinful Hamiltonian"):
        mfconstrains.enforce_constrains({(0, 0, 0): _dense(h.intra)}, h,
                ["no_net_moment"])


def test_the_name_comes_from_the_registry():
    assert "no_net_moment" in mfconstrains.get_constrain_names()
    with pytest.raises(ValueError, match="no_net_moment"): # the known names
        mfconstrains.enforce_constrains({}, None, ["no_such_constrain"])


def test_the_hubbard_ferromagnet_loses_its_uniform_exchange_field():
    """The Hubbard chain at U=10 and filling 0.2 converges to a saturated
    ferromagnet from a ferromagnetic guess (test_rpa_ferro_chain_bands.py).
    With the constraint the mean field holds no uniform exchange field, and
    with one site per cell no exchange field at all, so the loop ends in
    the paramagnet, whose two spins are degenerate"""
    h = geometry.chain().get_hamiltonian() # spinful
    scf = VJinteraction(h, U=10.0, filling=0.2, nk=20, mf="ferro",
            constrains=["no_net_moment"])
    assert scf.converged
    assert np.max(np.abs(_fields(scf.mf[(0, 0, 0)]))) < 1e-12
    assert np.max(np.abs(_fields(scf.hamiltonian.intra))) < 1e-12


def test_an_antiferromagnet_is_unchanged():
    """The Neel state of the honeycomb Hubbard model at U=3 and half
    filling has zero net moment, so the constraint changes neither its mean
    field nor the loop that converges to it"""
    h = geometry.honeycomb_lattice().get_hamiltonian() # spinful
    kw = dict(U=3.0, filling=0.5, mf="antiferro", nk=6, mix=0.5,
            maxerror=1e-8)
    free = VJinteraction(h, **kw)
    assert free.converged
    B = _fields(free.mf[(0, 0, 0)])
    assert np.min(np.abs(B[:, 2])) > 0.1 # a Neel state
    assert np.max(np.abs(B[0] + B[1])) < 1e-10 # with zero net moment
    out = mfconstrains.enforce_constrains(free.mf, h, ["no_net_moment"])
    for d in free.mf:
        assert np.max(np.abs(_dense(out[d]) - _dense(free.mf[d]))) < 1e-10
    held = VJinteraction(h, constrains=["no_net_moment"], **kw)
    assert held.converged
    for d in free.mf:
        assert np.max(np.abs(_dense(held.mf[d]) - _dense(free.mf[d]))) < 1e-6
