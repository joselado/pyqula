"""Reference check of the whole kernel against an independent
implementation. On a 0D cluster there is a single k-point and no momentum
bookkeeping at all, so the BSE matrix can be rebuilt directly in the
molecular-orbital basis from explicit four-index Coulomb integrals

    <pq|V|rs> = sum_ab conj(C_ap) C_ar W_ab conj(C_bq) C_bs

and the Casida blocks A_ia,jb = d(e_a-e_i) + <aj|V|ib> - <aj|V|bi>,
B_ia,jb = <ab|V|ij> - <ab|V|ji>. That reference shares no code with
bsetk/kernel.py, so agreement pins down the direct and exchange
contractions, their relative sign, and the A/B block layout at once.

A generic random interaction is used rather than a Hubbard U, so that no
term of the kernel can accidentally vanish."""
import numpy as np

from pyqula import algebra, geometry


def _cluster():
    g = geometry.chain().supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., 0.35])  # gap it, so valence/conduction is defined
    return h.get_multicell().get_dense()


def _random_interaction(n, seed=3):
    np.random.seed(seed)
    W = np.random.random((n, n)) * 0.4
    W = W + W.T                # density-density interactions are symmetric
    np.fill_diagonal(W, 0.)    # no self-interaction
    return W.astype(np.complex128)


def _molecular_orbital_tdhf(h, W):
    """Independent reference: build the TDHF/BSE matrix in the
    molecular-orbital basis and diagonalize it"""
    ev, C = algebra.eigh(h.intra)
    C = np.array(C)  # columns are eigenvectors
    n = len(ev)
    occ = [i for i in range(n) if ev[i] < 0.]
    emp = [i for i in range(n) if ev[i] >= 0.]

    def V4(p, q, r, s):
        return (np.conj(C[:, p]) * C[:, r]) @ W @ (np.conj(C[:, q]) * C[:, s])

    pairs = [(i, a) for i in occ for a in emp]
    npair = len(pairs)
    A = np.zeros((npair, npair), dtype=np.complex128)
    B = np.zeros((npair, npair), dtype=np.complex128)
    for m, (i, a) in enumerate(pairs):
        for mp, (j, b) in enumerate(pairs):
            A[m, mp] = V4(a, j, i, b) - V4(a, j, b, i)
            B[m, mp] = V4(a, b, i, j) - V4(a, b, j, i)
            if m == mp:
                A[m, mp] += ev[a] - ev[i]
    H = np.block([[A, B], [-B.conj().T, -A.conj()]])
    es = np.linalg.eigvals(H)
    return np.sort(es[es.real > 0].real)


def test_cluster_bse_matches_molecular_orbital_tdhf():
    h = _cluster()
    W = _random_interaction(h.intra.shape[0])
    es = np.sort(h.get_exciton_energies(V={(0, 0, 0): W}, nk=1).real)
    ref = _molecular_orbital_tdhf(h, W)
    assert len(es) == len(ref)
    assert np.max(np.abs(es - ref)) < 1e-10


def test_cluster_tda_matches_the_resonant_block_alone():
    """The Tamm-Dancoff spectrum must be the eigenvalues of A on its own"""
    h = _cluster()
    W = _random_interaction(h.intra.shape[0])
    b = h.get_bse(V={(0, 0, 0): W}, nk=1, tda=True)
    assert np.max(np.abs(np.sort(b.get_energies())
                         - np.sort(np.linalg.eigvalsh(b.A)))) < 1e-10
