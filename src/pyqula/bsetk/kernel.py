import numpy as np
from numba import jit,prange
from .. import parallel
from .interaction import interaction_at_q


def qdifference_map(kpoints):
    """Return (qs,iq) where qs is the list of distinct k-k' differences of
    the mesh (modulo a reciprocal lattice vector) and iq[i,j] indexes the
    difference kpoints[i]-kpoints[j] into it.

    The interaction only ever enters the direct kernel through W(k-k'), and
    W(q) is periodic in q (the Bloch phase exp(2 pi i q.d) has integer d),
    so on the regular Gamma-centered mesh geometry.get_kmesh returns there
    are only nk distinct differences rather than nk^2. Fourier transforming
    the interaction once per distinct difference instead of once per pair
    of k-points is what keeps the kernel build from dominating the cost."""
    nk = len(kpoints)
    keys,qs = dict(),[]
    iq = np.zeros((nk,nk),dtype=np.int64)
    for i in range(nk):
        for j in range(nk):
            q = np.mod(kpoints[i]-kpoints[j],1.0) # fold into [0,1)
            key = tuple(np.round(q,7)%1.0) # %1 again: 0.9999999 -> 1.0 -> 0.0
            if key not in keys:
                keys[key] = len(qs)
                qs.append(np.array(key))
            iq[i,j] = keys[key]
    return np.array(qs),iq


def interaction_tensor(W,g,qs):
    """Return the (nq,norb,norb) array of W(q) for every q in qs"""
    out = [interaction_at_q(W,g,q) for q in qs]
    return np.array(out,dtype=np.complex128)


def nonzero_pattern(Wqs,tol=1e-10):
    """Return the (row,col) indices where the interaction is non-zero.

    W(q) = sum_d W(d) exp(2 pi i q.d) has the same sparsity pattern in the
    orbital indices as the real-space interaction, whatever q is, and that
    pattern is typically very sparse -- a Hubbard U only couples the two
    spin components of each site, so 2*nsites entries out of (2*nsites)^2.
    Restricting the kernel's inner contraction to those entries turns its
    cost from O(norb^2) per matrix element into O(nnz)."""
    mask = np.max(np.abs(Wqs),axis=0)>tol
    rows,cols = np.nonzero(mask)
    return np.array(rows,dtype=np.int64),np.array(cols,dtype=np.int64)


@jit(nopython=True,parallel=True,cache=True)
def direct_block_jit(u1,u2,w1,w2,Wqs,iq,rows,cols,norm):
    """Generic direct (screened-interaction) block,

      out[m,n] = norm * sum_{ab} conj(u1[m,a]) u2[n,a] W_q(m,n)[a,b]
                                 conj(w1[n,b]) w2[m,b]

    All three direct blocks of the BSE (resonant, antiresonant and the
    resonant-antiresonant coupling) have this shape and differ only in
    which electron/hole coefficient array is passed as u1,u2,w1,w2 -- see
    build_blocks."""
    n1 = u1.shape[0]
    n2 = u2.shape[0]
    nnz = rows.shape[0]
    out = np.zeros((n1,n2),dtype=np.complex128)
    for m in prange(n1): # loop over rows, in parallel
        for n in range(n2): # loop over columns
            iqmn = iq[m,n] # which W(q) this matrix element needs
            acc = 0.0+0.0j
            for t in range(nnz): # loop over the non-zero entries of W
                a = rows[t]
                b = cols[t]
                acc = acc + (np.conj(u1[m,a])*u2[n,a]*Wqs[iqmn,a,b]
                             *np.conj(w1[n,b])*w2[m,b])
            out[m,n] = acc*norm
    return out


def direct_block(u1,u2,w1,w2,Wqs,iq,rows,cols,norm):
    """Wrapper of direct_block_jit that sets the thread count first"""
    parallel.set_num_threads() # honor parallel.py's thread configuration
    return direct_block_jit(u1,u2,w1,w2,Wqs,iq,rows,cols,norm)


def exchange_block(F1,F2,WQ,norm,conjugate=True):
    """Exchange (bare-interaction) block. Unlike the direct term this one
    factorizes: every exchange matrix element involves the interaction at
    the single center-of-mass momentum Q, never at k-k', so the block is
    just a product of the per-pair density form factors F[m,a] =
    conj(electron[m,a])*hole[m,a] with W(Q) between them. No loop over
    pairs is needed, and no numba kernel either.

    conjugate=True gives F1 W(Q) F2^dagger (the resonant and antiresonant
    diagonal blocks), conjugate=False gives F1 W(Q) F2^T (the coupling
    block, whose second index is an antiresonant pair and therefore enters
    unconjugated)."""
    G = F2.conj().T if conjugate else F2.T
    return norm*(F1@WQ@G)


def build_blocks(pb,W,kernel="full"):
    """Return (A,Abar,B), the three blocks of the BSE matrix at pb.Q.

      A     resonant block,        A[m,m']    = dE_m delta + X - D
      Abar  antiresonant block,    Abar[n,n'] = dEA_n delta + X - D
      B     coupling block,        B[m,n]     = X - D

    with the direct term D built from W(k-k') and the exchange term X from
    W(Q), following the standard localized-orbital BSE of the Xatu code
    (arXiv:2307.01572) generalized to the full (non-Tamm-Dancoff) problem.
    Signs follow the Casida convention A = dE + X - D, so the direct term
    binds and the exchange term (which is the only one surviving if the
    direct term is switched off) reproduces the RPA.

    kernel selects which terms are included:
      "full"     both, i.e. time-dependent Hartree-Fock on top of the mean
                 field -- the physical choice
      "direct"   ladder only (no exchange): no singlet/triplet splitting
      "exchange" Hartree only: this is exactly the RPA, and is what
                 tests/bse cross-checks against chitk.rpa's independent
                 frequency-scan implementation
      "none"     no interaction: eigenvalues collapse onto the
                 independent-particle transition energies
    """
    g = pb.geometry
    norm = 1.0/len(pb.kpoints) # 1/N, N the number of unit cells
    A = np.diag(pb.dE).astype(np.complex128)
    Abar = np.diag(pb.dEA).astype(np.complex128)
    B = np.zeros((pb.npair,pb.npair),dtype=np.complex128)
    if kernel=="none": return A,Abar,B
    if kernel not in ("full","direct","exchange"):
        raise ValueError("kernel must be one of 'full', 'direct', "
                "'exchange', 'none', got %r"%(kernel,))
    if kernel in ("full","exchange"): # exchange (Hartree) term
        WQ = interaction_at_q(W,g,pb.Q) # W(+Q)
        WmQ = np.conj(WQ) # W(-Q); W(d) is real, so this is the same as
        # interaction_at_q(W,g,-pb.Q), just without a second Fourier sum
        Fr = np.conj(pb.el)*pb.ho # resonant density form factors
        Fa = np.conj(pb.elA)*pb.hoA # antiresonant density form factors
        # NOTE the antiresonant block takes W(-Q), not W(+Q). Its pair
        # densities run the other way round -- the resonant pair puts the
        # electron at k+Q and the hole at k, the antiresonant one the hole
        # at k+Q and the electron at k -- so its lattice sum picks up
        # exp(-i Q.d) instead of exp(+i Q.d). W(Q) is Hermitian but not
        # real once the interaction reaches beyond the unit cell, so
        # W(-Q) = conj(W(Q)) genuinely differs from W(Q) there, and using
        # W(Q) for both silently gives wrong energies at finite Q for any
        # extended (V1, V2, ... ) interaction while staying exactly right
        # for a purely onsite Hubbard U. tests/bse/test_bse_supercell.py
        # is what separates the two cases.
        A = A + exchange_block(Fr,Fr,WQ,norm)
        Abar = Abar + exchange_block(Fa,Fa,WmQ,norm)
        B = B + exchange_block(Fr,Fa,WQ,norm,conjugate=False)
    if kernel in ("full","direct"): # direct (screened) term, W(k-k')
        qs,iqk = qdifference_map(pb.kpoints)
        Wqs = interaction_tensor(W,g,qs)
        rows,cols = nonzero_pattern(Wqs)
        iq = iqk[np.ix_(pb.kindex,pb.kindex)] # per-pair difference index
        A = A - direct_block(pb.el,pb.el,pb.ho,pb.ho,Wqs,iq,rows,cols,norm)
        Abar = Abar - direct_block(pb.elA,pb.elA,pb.hoA,pb.hoA,
                                   Wqs,iq,rows,cols,norm)
        B = B - direct_block(pb.el,pb.hoA,pb.elA,pb.ho,
                             Wqs,iq,rows,cols,norm)
    return A,Abar,B
