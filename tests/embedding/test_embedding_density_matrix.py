import numpy as np
import pytest

from pyqula import geometry
from pyqula import embedding


def _dense(m):
    return np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def _rashba_chain_with_defect():
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(0.3)
    hd = h.copy()
    hd.add_onsite(1.0)
    hd.add_exchange([0.4, -0.6, 0.3])
    return h, hd


def _finite_chain_defect_block(h, hd, n):
    """Occupied projector on the defect cell of an n-cell open chain whose
    middle cell carries the defect onsite matrix, from exact diagonalization.
    Reversing the chain maps the middle cell onto itself, so the direction
    in which the hopping h.inter is laid down does not change this block."""
    intra, inter, m = _dense(h.intra), _dense(h.inter), _dense(hd.intra)
    d = intra.shape[0]
    H = np.zeros((n*d, n*d), dtype=np.complex128)
    for i in range(n):
        H[i*d:(i+1)*d, i*d:(i+1)*d] = intra
        if i < n-1:
            H[i*d:(i+1)*d, (i+1)*d:(i+2)*d] = inter
            H[(i+1)*d:(i+2)*d, i*d:(i+1)*d] = inter.conj().T
    c = n//2
    H[c*d:(c+1)*d, c*d:(c+1)*d] = m
    e, v = np.linalg.eigh(H)
    vo = v[c*d:(c+1)*d, e < 0.]
    return vo @ vo.conj().T


def test_defect_density_matrix_matches_finite_chain():
    """The contour integral of Embedding.get_density_matrix used to run at a
    fixed absolute tolerance of 1e-2 that no keyword reached, leaving an
    error of 3e-3 on the defect block that neither delta nor nk removed.
    With the default tolerance the diagonal of the defect block must agree
    with a long finite chain carrying the same defect; the 601-cell chain
    is within 4e-4 of a 2401-cell one. The diagonal does not depend on the
    index convention of the returned matrix."""
    h, hd = _rashba_chain_with_defect()
    eb = embedding.Embedding(h, m=hd.intra)
    dm = np.array(eb.get_density_matrix(delta=1e-3))
    rho = _finite_chain_defect_block(h, hd, 601)
    assert np.allclose(np.diag(dm).real, np.diag(rho).real, atol=1e-3, rtol=0.)


def test_embedded_hamiltonian_takes_the_callers_delta():
    """Embedded_Hamiltonian.get_density_matrix passed its own delta together
    with the caller's keywords, so get_density_matrix(delta=...) raised a
    TypeError. The caller's delta must win over the one stored at
    construction, the stored one must be the fallback, and a small delta
    must reproduce the occupied projector of the island."""
    from pyqula.embeddingtk.embedded import Embedded_Hamiltonian
    g = geometry.chain().get_supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.3)
    h.add_exchange([0.3, 0.2, 0.1])
    h.add_onsite(lambda r: 0.4*np.cos(r[0])) # breaks particle-hole symmetry
    wide = Embedded_Hamiltonian(h, delta=1e-1)
    narrow = Embedded_Hamiltonian(h, delta=1e-3)
    dm = np.array(wide.get_density_matrix(delta=1e-3))
    assert np.allclose(dm, np.array(narrow.get_density_matrix()), atol=1e-8)
    assert np.allclose(np.array(wide.get_density_matrix()),
                       np.array(narrow.get_density_matrix(delta=1e-1)), atol=1e-8)
    e, v = np.linalg.eigh(_dense(h.intra))
    vo = v[:, e < 0.]
    rho = vo @ vo.conj().T
    assert np.allclose(np.diag(dm).real, np.diag(rho).real, atol=1e-3, rtol=0.)


def test_embedded_hamiltonian_shares_the_full_dm_convention():
    """Embedded_Hamiltonian.get_density_matrix used to return the usual
    rho, the transpose of what Hamiltonian.get_density_matrix returns, so
    the contraction sum(dm*A) that gives <A> on the latter gave <A*> on the
    former, with the opposite sign for sy. On an island with complex
    amplitudes both must agree to the broadening, and so must <sy>."""
    from pyqula.embeddingtk.embedded import Embedded_Hamiltonian
    g = geometry.chain().get_supercell(6)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.5)
    h.add_exchange([0.3, 0.5, 0.2])
    h.add_onsite(lambda r: 0.2*np.cos(1.3*r[0]) + 0.1)
    dm0 = _dense(h.get_density_matrix())
    dm = np.array(Embedded_Hamiltonian(h, delta=1e-4).get_density_matrix())
    assert np.max(np.abs(dm0 - dm0.T)) > 0.1 # the two conventions differ
    assert np.allclose(dm, dm0, atol=1e-3, rtol=0.)
    sy = _dense(h.get_operator("sy").get_matrix())
    vev0 = np.sum(dm0*sy).real
    assert abs(vev0) > 0.1
    assert abs(np.sum(dm*sy).real - vev0) < 1e-3


def test_defect_density_matrix_is_in_the_full_dm_convention():
    """Embedding.get_density_matrix used to return the usual rho of the
    defect cell. The whole block, off-diagonal included, must agree with
    the full_dm-convention block of a long finite chain carrying the same
    defect, which is the transpose of the occupied projector; the
    imaginary part of the off-diagonal is what tells the two apart."""
    h, hd = _rashba_chain_with_defect()
    eb = embedding.Embedding(h, m=hd.intra)
    dm = np.array(eb.get_density_matrix(delta=1e-3))
    rho = _finite_chain_defect_block(h, hd, 601)
    assert np.max(np.abs(rho - rho.T)) > 0.05 # the two conventions differ
    assert np.allclose(dm, rho.T, atol=1e-3, rtol=0.)


def test_the_density_matrix_of_a_3d_embedding_with_the_tetrahedra():
    """In 3d the tetrahedra (gf_mode="triangle") are the only bulk Green's
    function of an embedding, and they refused the complex energies of the
    density-matrix contour, so a 3d embedding had no density matrix. A
    complex energy is now the real energy with its imaginary part as the
    broadening. For a pristine insulator every band of the interpolant is
    full or empty in every tetrahedron, so the contour gives the mean of
    the occupied projectors over the same mesh, up to the tails of the
    broadening delta, in the full_dm convention (the transpose)"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.2)
    # a staggered field with a y component, so that the density matrix is
    # complex and its transpose is another matrix
    h.add_antiferromagnetism([[0.3, 0.8, 0.2], [0.3, 0.8, 0.2]])
    nk = 6
    eb = embedding.Embedding(h)
    eb.gf_mode = "triangle"
    dm = np.array(eb.get_density_matrix(delta=1e-3, nk=nk))
    hk = h.get_hk_gen()
    ks = np.linspace(0., 1., nk, endpoint=False)
    rho = 0.
    for a in ks:
        for b in ks:
            for c in ks:
                e, v = np.linalg.eigh(_dense(hk([a, b, c])))
                assert np.min(np.abs(e)) > 0.5 # a gap around the Fermi level
                vo = v[:, e < 0.]
                rho = rho + vo @ vo.conj().T
    rho = rho/nk**3
    assert np.max(np.abs(rho - rho.T)) > 0.05 # the two conventions differ
    assert np.allclose(dm, rho.T, atol=1e-3, rtol=0.)


def test_a_3d_embedding_takes_the_tetrahedra_by_default():
    """The default gf_mode of an Embedding was "renormalization" in every
    dimension, which stops in 3d, so a 3d Embedding raised unless the mode
    was set by hand. A 3d crystal takes the tetrahedra by default: the
    density matrix of a pristine insulator, one band of two occupied, is
    Hermitian with trace one up to the tails of the broadening, and a 2d
    crystal keeps the renormalization"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(1.0)
    eb = embedding.Embedding(h)
    assert eb.gf_mode == "triangle"
    dm = np.array(eb.get_density_matrix(delta=1e-3, nk=4))
    assert np.max(np.abs(dm - dm.conj().T)) < 1e-8
    assert abs(np.trace(dm).real - 1.) < 1e-2
    h2 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    assert embedding.Embedding(h2).gf_mode == "renormalization"
