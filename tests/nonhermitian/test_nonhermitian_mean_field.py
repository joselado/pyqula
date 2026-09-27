"""Mean field of a non-Hermitian Hamiltonian, from its biorthogonal
density matrix rho = sum_occ |R_m><L_m| with <L_m|R_m> = 1 (D. C. Brody,
Biorthogonal quantum mechanics, arXiv:1308.2609, Eqs. (10) and (27)).

The self-consistent loop used to build the density matrix with a
Hermitian eigensolver, which reads only the lower triangle and the real
part of the diagonal of H(k): an onsite gain and loss i*gamma was dropped,
and a non-reciprocal hopping was replaced by the mirror image of one of
its two directions. Every test here fails on that code."""

import numpy as np
import scipy.linalg as lg

from pyqula import geometry
from pyqula import densitymatrix


def _open_geometry(n):
    """Open chain of n sites. Its two sublattices, which fix the sign of
    the antiferromagnetic guess, are labeled from a random site, so every
    chain compared within a test is built on the same geometry"""
    g = geometry.chain(n)
    g.dimensionality = 0 # open boundary conditions
    g.get_sublattice() # for the antiferromagnetic guess
    return g


def _open_chain(g, t_right, t_left, non_hermitian=True, onsite=None,
        has_spin=True):
    """Open chain on the geometry g, with t_right the coefficient of
    c^dag_i c_{i+1} and t_left that of c^dag_{i+1} c_i (Hatano-Nelson),
    and an optional onsite(i) term"""
    def tij(r1, r2):
        dx = r2[0] - r1[0]
        if abs(dx - 1.) < 1e-6: return t_right
        if abs(dx + 1.) < 1e-6: return t_left
        return 0.
    h = g.get_hamiltonian(tij=tij, has_spin=has_spin,
            non_hermitian=non_hermitian)
    if onsite is not None:
        x = np.array(g.r)[:, 0]
        h.add_onsite(lambda r: onsite(int(round(r[0] - x.min()))))
    return h


def _scf(h, U, **kwargs):
    return h.get_mean_field_hamiltonian(U=U, filling=0.5, mf="antiferro",
            maxerror=1e-10, mix=0.5, maxite=5000, verbose=0, **kwargs)


def test_hatano_nelson_chain_is_the_gauge_transformed_hermitian_chain():
    """The open Hatano-Nelson chain is S H0 S^-1 with S=diag(s^i),
    s=sqrt(t_left/t_right), and H0 the Hermitian chain of hopping
    t=sqrt(t_left t_right). The transformation leaves every biorthogonal
    density n_i = R_i conj(L_i) unchanged, so the onsite Hubbard mean field
    is the same and the self-consistent spectrum is that of the Hermitian
    chain (the imaginary gauge transformation of Hatano and Nelson,
    arXiv:cond-mat/9603165, which applies to interacting systems too).
    The old code solved instead a Hermitian chain of hopping t_left."""
    n, U, t_right, t_left = 12, 2.5, 0.6, 1.5
    g = _open_geometry(n)
    hnh = _scf(_open_chain(g, t_right, t_left), U)
    hh = _scf(_open_chain(g, np.sqrt(t_right*t_left), np.sqrt(t_right*t_left),
            non_hermitian=False), U)
    enh = np.sort_complex(lg.eigvals(hnh.intra))
    eh = np.sort(np.linalg.eigvalsh(hh.intra))
    assert np.max(np.abs(enh.imag)) < 1e-8 # a real spectrum
    assert np.max(np.abs(np.sort(enh.real) - eh)) < 1e-7, (enh, eh)
    # the onsite mean fields, the Hubbard fields of the densities, agree
    dnh, dh = np.diag(hnh.intra), np.diag(hh.intra)
    assert np.max(np.abs(dnh - dh)) < 1e-7
    mnh, mh = hnh.get_magnetization(), hh.get_magnetization()
    assert np.max(np.abs(mnh - mh)) < 1e-7
    assert np.max(np.abs(mh[:, 2])) > 0.2 # an antiferromagnet


def test_hatano_nelson_chain_with_fock_terms(tmp_path, monkeypatch):
    """A first-neighbor V1 on the spinless Hatano-Nelson chain has Fock
    terms on the bonds, <c^dag_i c_{i+1}>, which transform under the
    imaginary gauge transformation exactly as the hopping does, so the
    self-consistent spectrum is again that of the Hermitian chain with
    t=sqrt(t_left t_right). This goes through the density-density engine
    (Vinteraction) and the Fock term of the reversed bond, which the
    Hermitian decoupling writes as a complex conjugate"""
    monkeypatch.chdir(tmp_path) # the density-density engine writes MF.pkl
    n, V1, t_right, t_left = 12, 2.5, 0.7, 1.3
    g = _open_geometry(n)
    kw = dict(V1=V1, filling=0.5, mf="CDW", maxerror=1e-10, mix=0.5,
              maxite=5000, verbose=0)
    hnh = _open_chain(g, t_right, t_left, has_spin=False)
    hnh = hnh.get_mean_field_hamiltonian(**kw)
    t = np.sqrt(t_right*t_left)
    hh = _open_chain(g, t, t, non_hermitian=False, has_spin=False)
    hh = hh.get_mean_field_hamiltonian(**kw)
    enh = np.sort_complex(lg.eigvals(hnh.intra))
    eh = np.sort(np.linalg.eigvalsh(hh.intra))
    assert np.max(np.abs(enh.imag)) < 1e-8 # a real spectrum
    assert np.max(np.abs(np.sort(enh.real) - eh)) < 1e-7, (enh, eh)
    # the bond mean fields transform as the hopping: s^(i-j) times the
    # Hermitian ones, s = sqrt(t_left/t_right)
    x = np.array(g.r)[:, 0] # position along the chain
    s = np.sqrt(t_left/t_right)
    gauge = s**(x[:, None] - x[None, :])
    assert np.max(np.abs(np.asarray(hnh.intra)
            - gauge*np.asarray(hh.intra))) < 1e-7


def _reference_biorthogonal_hubbard(h0, U, n_up, n_dn, ne, ite=4000):
    """Independent collinear Hartree loop for a spinful 0d Hamiltonian
    whose spin blocks do not mix: H_sigma = H0_sigma + U diag(n_{-sigma}),
    with the ne lowest states in real part occupied, and biorthogonal
    densities n = diag(R f R^-1)"""
    m = np.array(h0.intra.todense()) if hasattr(h0.intra, "todense") \
        else np.array(h0.intra)
    hup, hdn = m[0::2, 0::2], m[1::2, 1::2]
    for i in range(ite):
        es, rs = [], []
        for (hs, nop) in [(hup, n_dn), (hdn, n_up)]:
            e, r = lg.eig(hs + U*np.diag(nop))
            es.append(e) ; rs.append(r)
        allre = np.sort(np.concatenate(es).real)
        ef = (allre[ne-1] + allre[ne])/2. # Fermi energy on Re E
        new = []
        for (e, r) in zip(es, rs):
            f = (e.real < ef).astype(float)
            new.append(np.diag(r@np.diag(f)@lg.inv(r)))
        err = np.max(np.abs(new[0] - n_up)) + np.max(np.abs(new[1] - n_dn))
        n_up = 0.5*n_up + 0.5*new[0] ; n_dn = 0.5*n_dn + 0.5*new[1]
        if err < 1e-11: break
    return n_up, n_dn


def test_gain_and_loss_enter_the_self_consistent_fields():
    """A chain with gain and loss +-i*gamma on alternate pairs of sites
    and a Hubbard U: the biorthogonal densities are complex, and the
    converged onsite fields must be those of an independent biorthogonal
    Hartree loop, and differ from those of the same chain without the gain
    and loss, which is what the old code converged to"""
    n, U, gamma = 8, 3.0, 0.4
    pattern = lambda i: 1j*gamma*(1 if (i//2) % 2 == 0 else -1)
    g = _open_geometry(n)
    h0 = _open_chain(g, 1., 1., onsite=pattern)
    hscf = _scf(h0, U)
    dm = densitymatrix.full_dm(hscf) # biorthogonal, at the Fermi energy 0
    occ = np.diag(dm)
    # the Neel pattern of mf="antiferro", which picks one of the two
    # degenerate solutions (the other is its PT image)
    n_up0 = 0.5 - 0.3*np.array(g.sublattice) ; n_dn0 = 1. - n_up0
    ref_up, ref_dn = _reference_biorthogonal_hubbard(h0, U, n_up0.astype(complex),
            n_dn0.astype(complex), ne=n)
    assert np.max(np.abs(occ[0::2] - ref_up)) < 1e-6, (occ[0::2], ref_up)
    assert np.max(np.abs(occ[1::2] - ref_dn)) < 1e-6, (occ[1::2], ref_dn)
    assert np.max(np.abs(ref_up.imag)) > 1e-3 # complex densities
    hh = _scf(_open_chain(g, 1., 1.), U) # the same chain, Hermitian
    field = lambda h: np.diag(h.intra).real - np.mean(np.diag(h.intra).real)
    assert np.max(np.abs(field(hscf) - field(hh))) > 1e-2


def test_hermitian_matrix_flagged_non_hermitian_gives_the_hermitian_result():
    """For a Hermitian matrix the biorthogonal density matrix is the
    orthonormal one, so the non-Hermitian code path (left eigenvectors from
    R^-1, the Fock term of the opposite bond without conjugation, the
    biorthogonal double counting) must reproduce the Hermitian one,
    including the Fock terms of a first-neighbor V1 and the total energy"""
    g = geometry.honeycomb_lattice()
    out = []
    for nh in [False, True]:
        h = g.get_hamiltonian(non_hermitian=nh)
        (hs, etot) = h.get_mean_field_hamiltonian(U=3., V1=0.5, filling=0.5,
                mf="antiferro", nk=6, maxerror=1e-11, mix=0.8, maxite=3000,
                verbose=0, return_total_energy=True)
        out.append((hs, etot))
    (h1, e1), (h2, e2) = out
    assert abs(e1 - e2) < 1e-8, (e1, e2)
    d1, d2 = h1.get_dict(), h2.get_dict()
    for key in d1:
        assert np.max(np.abs(np.array(d1[key]) - np.array(d2[key]))) < 1e-7
