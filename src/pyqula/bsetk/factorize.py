"""Exact low-rank factorization of the BSE kernel.

The dense BSE matrix of kernel.build_blocks is never needed: every block of
it is a diagonal plus a sum of R rank-one terms, with R fixed by the
interaction and INDEPENDENT of the k-mesh.

Why. The direct (ladder) term reads

    D[m,n] = (1/N) sum_ab conj(u1[m,a]) u2[n,a] W_ab(k_m-k_n)
                          conj(w1[n,b]) w2[m,b]

and a real-space interaction dictionary Fourier transforms as
W_ab(q) = sum_d W_ab(d) exp(2 pi i q.d), so the only k-dependence of the
interaction factorizes,

    W_ab(k_m-k_n) = sum_d W_ab(d) phi(d,k_m) conj(phi(d,k_n))

with phi = geometry.bloch_phase. Substituting and collecting the m- and
n-dependent pieces gives

    D[m,n] = (1/N) sum_{a,b,d} W_ab(d) L_{abd}[m] conj(R_{abd}[n])
    L_{abd}[m] = conj(u1[m,a]) w2[m,b] phi(d,k_m)
    R_{abd}[n] = conj(u2[n,a]) w1[n,b] phi(d,k_n)

i.e. one rank-one term per non-zero entry (a,b,d) of the real-space
interaction. For the resonant block u1=u2 and w1=w2, so L=R and the block
is a sum of |L><L| projectors; the coupling block B mixes resonant and
antiresonant coefficients and needs both.

The exchange (Hartree) term is already factorized in kernel.exchange_block
as F1 W(Q) F2^dag, which is a sum of at most norb^2 rank-one terms with no
lattice sum at all.

Consequences:

  - the matrix-vector product costs O(R*npair) instead of O(npair^2), and
    needs O(R*npair) memory instead of O(npair^2). That is what
    bsetk/iterative.py uses to get past solve.check_memory's wall.
  - each factor is a function of the pair index alone, so it can be
    tensor-train compressed one at a time. That is what bsetk/qtt.py uses.

R is the number of non-zeros of the real-space interaction: 2*nsites for a
Hubbard U, and norb^2 times the number of neighbor shells for an extended
one (36 for a 4-orbital cell with U+V1+V2). A tabulated, RPA-screened
interaction is the exception -- inverse Fourier transforming it over the
mesh gives nk real-space vectors, so R = nk*norb^2 and the rank
independence is lost. Truncate it in real space first; see
screening.ScreenedInteraction.get_dict and the note in solve.py.

The factorization is exact, not an approximation: tests/bse/
test_bse_factorize.py reconstructs kernel.build_blocks' own dense blocks
from it and requires agreement to machine precision.

Reference for this construction (low-rank factorization of the BSE kernel
feeding an iterative/tensor-train eigensolver): Benner, Dolgov,
Khoromskaia and Khoromskij, arXiv:1602.02646.
"""
import numpy as np

from .interaction import interaction_at_q


class KernelFactorization():
    """One block of the BSE matrix, held as diag(dE) + sum_t c_t |L_t><R_t|.

    block selects which one:
      "A"     resonant,      diagonal dE,  L=R built from (el,ho)
      "Abar"  antiresonant,  diagonal dEA, L=R built from (elA,hoA)
      "B"     coupling,      no diagonal,  L from (el,ho), R from (hoA,elA)

    Attributes:
      diag    (npair,) real diagonal, zero for the coupling block
      coefs   (nterm,) complex coefficients c_t
      left    (nterm,npair) complex, the L_t
      right   (nterm,npair) complex, the R_t (the same array object as
              left for the Hermitian blocks, so no extra memory)
    """
    def __init__(self,pb,W,Wx=None,kernel="full",block="A"):
        if Wx is None: Wx = W # exchange defaults to the direct interaction
        if kernel not in ("full","direct","exchange","none"):
            raise ValueError("kernel must be one of 'full', 'direct', "
                    "'exchange', 'none', got %r"%(kernel,))
        if block not in ("A","Abar","B"):
            raise ValueError("block must be 'A', 'Abar' or 'B', got %r"
                    %(block,))
        self.pairs = pb
        self.block = block
        self.kernel = kernel
        self.npair = pb.npair
        self.diag = _block_diagonal(pb,block)
        coefs,left,right = [],[],[]
        if kernel in ("full","exchange"):
            c,l,r = _exchange_factors(pb,Wx,block)
            coefs.append(c) ; left.append(l) ; right.append(r)
        if kernel in ("full","direct"):
            c,l,r = _direct_factors(pb,W,block)
            # the direct term enters the Casida form with a minus sign,
            # A = dE + X - D, exactly as kernel.build_blocks writes it
            coefs.append(-c) ; left.append(l) ; right.append(r)
        if len(coefs)==0: # kernel="none", the diagonal alone
            z = np.zeros((0,pb.npair),dtype=np.complex128)
            self.coefs = np.zeros(0,dtype=np.complex128)
            self.left,self.right = z,z
        else:
            self.coefs = np.concatenate(coefs)
            self.left = np.concatenate(left,axis=0)
            self.right = np.concatenate(right,axis=0)
        self.nterm = len(self.coefs)
    def matvec(self,x):
        """Apply the block to a vector (or to a set of column vectors),
        without ever forming the matrix. Cost O(nterm*npair) per column."""
        x = np.asarray(x)
        vec = x.ndim==1
        X = x.reshape(self.npair,-1)
        out = self.diag[:,None]*X
        if self.nterm>0:
            # sum_t c_t L_t (R_t^dag X); the inner product first, so the
            # npair x npair matrix is never formed
            proj = self.right.conj()@X # (nterm,ncol)
            out = out + self.left.T@(self.coefs[:,None]*proj)
        return out[:,0] if vec else out
    def to_dense(self):
        """Reconstruct the dense block. For tests and small meshes only --
        this is exactly the allocation the factorization exists to avoid."""
        out = np.diag(self.diag).astype(np.complex128)
        for t in range(self.nterm):
            out = out + self.coefs[t]*np.outer(self.left[t],
                                               self.right[t].conj())
        return out
    def diagonal(self):
        """The diagonal of the block, again without forming it. Useful as
        a preconditioner for the iterative solver."""
        out = np.array(self.diag,dtype=np.complex128)
        if self.nterm>0:
            out = out + np.einsum("t,tm,tm->m",self.coefs,self.left,
                    self.right.conj())
        return out


def _block_diagonal(pb,block):
    """The independent-particle transition energies of a block"""
    if block=="A": return np.array(pb.dE,dtype=np.float64)
    if block=="Abar": return np.array(pb.dEA,dtype=np.float64)
    return np.zeros(pb.npair,dtype=np.float64) # the coupling block has none


def _block_coefficients(pb,block):
    """Return (u1,u2,w1,w2), the four coefficient arrays this block's
    direct term contracts, in the same order kernel.direct_block takes
    them. Reading these off build_blocks rather than re-deriving them is
    deliberate: the finite-Q bookkeeping is the part that is easy to get
    wrong and it already lives there."""
    if block=="A": return pb.el,pb.el,pb.ho,pb.ho
    if block=="Abar": return pb.elA,pb.elA,pb.hoA,pb.hoA
    return pb.el,pb.hoA,pb.elA,pb.ho # coupling block


def _direct_factors(pb,W,block):
    """Rank-one factors of the direct term of a block.

    Returns (coefs,left,right) with the block equal to
    (1/N) sum_t coefs[t] |left[t]><right[t]|, which reproduces

      D[m,n] = (1/N) sum_ab conj(u1[m,a]) u2[n,a] W_ab(k_m-k_n)
                            conj(w1[n,b]) w2[m,b]

    See the module docstring for the derivation. The 1/N is folded into
    the coefficients."""
    if hasattr(W,"at"):
        raise ValueError("the low-rank factorization needs a real-space "
            "interaction dictionary, but got a tabulated interaction "
            "(a screening.ScreenedInteraction). Its Fourier transform "
            "does not factorize into a fixed number of lattice terms -- "
            "inverse transforming it over the mesh gives nk real-space "
            "vectors, so the factorization rank would grow with the mesh "
            "and the whole point would be lost. Call .get_dict(cutoff=...) "
            "on it first to truncate it in real space, and read that "
            "method's note on the truncation error")
    g = pb.geometry
    u1,u2,w1,w2 = _block_coefficients(pb,block)
    norm = 1.0/len(pb.kpoints) # 1/N, N the number of unit cells
    # the Bloch phase of every mesh point, per lattice vector of W; the
    # pair index inherits it through the k-point of its pair
    coefs,left,right = [],[],[]
    for d,m in W.items():
        m = np.array(m,dtype=np.complex128)
        phase = np.array([g.bloch_phase(d,k) for k in pb.kpoints])
        ph = phase[pb.kindex] # (npair,)
        rows,cols = np.nonzero(np.abs(m)>1e-10)
        for a,b in zip(rows,cols):
            coefs.append(norm*m[a,b])
            left.append(np.conj(u1[:,a])*w2[:,b]*ph)
            right.append(np.conj(u2[:,a])*w1[:,b]*ph)
    if len(coefs)==0: # an interaction that is identically zero
        z = np.zeros((0,pb.npair),dtype=np.complex128)
        return np.zeros(0,dtype=np.complex128),z,z
    return (np.array(coefs,dtype=np.complex128),
            np.array(left,dtype=np.complex128),
            np.array(right,dtype=np.complex128))


def _exchange_factors(pb,Wx,block):
    """Rank-one factors of the exchange term of a block.

    kernel.exchange_block already writes this term as F1 W(Q) F2^dag (or
    F1 W(Q) F2^T for the coupling block), so the factorization is just the
    orbital sum written out: sum_ab W_ab(Q) |F1_a><F2_b|.

    NOTE the antiresonant block takes W(-Q) = conj(W(Q)), and the coupling
    block's second index is an antiresonant pair entering unconjugated --
    both exactly as build_blocks does them, and both invisible at Q=0. See
    kernel.build_blocks' comment for why."""
    g = pb.geometry
    norm = 1.0/len(pb.kpoints)
    WQ = interaction_at_q(Wx,g,pb.Q) # W(+Q), the bare interaction
    Fr = np.conj(pb.el)*pb.ho # resonant density form factors
    Fa = np.conj(pb.elA)*pb.hoA # antiresonant density form factors
    # matvec contracts `right` conjugated, so store whichever of F2 /
    # conj(F2) makes conj(right) reproduce exchange_block's own second
    # factor: F2^dag for the diagonal blocks, F2^T for the coupling one
    if block=="A": F1,F2,M = Fr,Fr,WQ
    elif block=="Abar": F1,F2,M = Fa,Fa,np.conj(WQ) # W(-Q)
    else: F1,F2,M = Fr,np.conj(Fa),WQ # coupling: F2 enters unconjugated
    coefs,left,right = [],[],[]
    rows,cols = np.nonzero(np.abs(M)>1e-10)
    for a,b in zip(rows,cols):
        coefs.append(norm*M[a,b])
        left.append(F1[:,a])
        right.append(F2[:,b])
    if len(coefs)==0:
        z = np.zeros((0,pb.npair),dtype=np.complex128)
        return np.zeros(0,dtype=np.complex128),z,z
    return (np.array(coefs,dtype=np.complex128),
            np.array(left,dtype=np.complex128),
            np.array(right,dtype=np.complex128))
