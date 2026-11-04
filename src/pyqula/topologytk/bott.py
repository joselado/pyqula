"""Bott index and spin Bott index of a two-dimensional insulator

T. A. Loring and M. B. Hastings, Disordered topological insulators via
C*-algebras, EPL 92, 67004 (2010), arXiv:1005.4883
H. Huang and F. Liu, Theory of spin Bott index for quantum spin Hall states
in non-periodic systems, Phys. Rev. B 98, 125130 (2018), arXiv:1810.00081
D. Toniolo, On the Bott index of unitary matrices on a finite torus,
arXiv:1708.05912 (the equality with the Chern number up to O(1/L))

The Hamiltonian is a periodic supercell taken at k=0, so the sample is a
torus. With P the projector on the states below zero energy and X, Y the
fractional coordinates of the orbitals along a1 and a2, in [0,1),

    U = P exp(2 pi i X) P,  V = P exp(2 pi i Y) P,
    B = Im Tr log(V U V^dag U^dag) / 2 pi.

U and V are written in the occupied states, a matrix of size N_occ, which
gives the same B as the full matrices with the complement 1-P added
(Huang and Liu), since that block only adds eigenvalues equal to one. Both
are replaced by their unitary part, Z W^dag of the singular value
decomposition Z S W^dag, which leaves B an integer whenever U and V are
far from singular.

The sign follows the order of a1 and a2, as h.get_chern() does, so a lattice
with the two vectors swapped gives -B, and B equals the Chern number of the
same Hamiltonian in the k-space routines. PyBott (GPL-3) takes the product
in the opposite order, U V U^dag V^dag, and so gives -B.
"""
import numpy as np
from .. import algebra


def _gamma_matrix(h):
    """The Hamiltonian of a two-dimensional supercell at k=0, as a dense
    matrix: the Hamiltonian of the sample closed into a torus"""
    if h.dimensionality!=2:
        raise ValueError("the Bott index needs a two-dimensional periodic "
          +"Hamiltonian (a supercell, closed into a torus at k=0), and this "
          +"one has dimensionality "+str(h.dimensionality))
    return np.array(algebra.todense(h.get_hk_gen()([0.,0.,0.])),dtype=complex)


def _occupied(m,tol):
    """Eigenvectors (columns) of m below zero energy, raising if a level is
    closer to zero than tol, where the projector is not defined. A level
    close to zero but not at it is allowed: in a disordered sample it is a
    localized state, and the Bott index stays defined in the mobility gap"""
    (es,vs) = algebra.eigh(m)
    emin = np.min(np.abs(es))
    if emin<tol:
        raise ValueError("the Hamiltonian has a level at E="+str(emin)
          +" on the torus, closer to the Fermi energy at zero than tol="
          +str(tol)+", so the projector on the occupied states is not "
          +"defined; shift the Fermi energy into the gap with h.set_filling() "
          +"or h.add_onsite()")
    return np.array(vs)[:,es<0.]


def _phases(h,n):
    """exp(2 pi i X) and exp(2 pi i Y), the diagonal of the two unitary
    position operators of the torus, one entry per orbital"""
    from .qgt import _orbital_fractions
    f = _orbital_fractions(h,n) # (n,2), fractions along a1 and a2
    return np.exp(2j*np.pi*f[:,0]),np.exp(2j*np.pi*f[:,1])


def _unitary_part(m):
    """The unitary factor Z W^dag of the singular value decomposition"""
    (z,s,wh) = np.linalg.svd(m)
    return z@wh


def bott_of_states(w,ex,ey):
    """Bott index of the projector on the columns of w, with ex and ey the
    diagonals of exp(2 pi i X) and exp(2 pi i Y)"""
    if w.shape[1]==0: return 0.
    u = _unitary_part(np.conjugate(w.T)@(ex[:,None]*w))
    v = _unitary_part(np.conjugate(w.T)@(ey[:,None]*w))
    ev = np.linalg.eigvals(v@u@np.conjugate(v.T)@np.conjugate(u.T))
    return float(np.sum(np.angle(ev))/(2.*np.pi))


def bott_index(h,tol=1e-8):
    """Bott index of the states below zero energy of a periodic supercell,
    taken at k=0, a real number that is an integer, equal to the Chern
    number, in a large enough insulating sample, with or without disorder.

    tol: the smallest distance of a level to zero energy that is accepted

    Close to a transition, or in a sample smaller than the localization
    length, U and V are close to singular and B is not an integer; an
    integer B is the sign that the sample is large enough"""
    m = _gamma_matrix(h)
    w = _occupied(m,tol)
    (ex,ey) = _phases(h,m.shape[0])
    return bott_of_states(w,ex,ey)


def split_bott(h,operator=None,tol=1e-8):
    """(B_+ - B_-)/2, with the occupied states split by the sign of their
    projected operator P O P (the construction of Huang and Liu for s_z),
    raising if P O P has an eigenvalue closer to zero than tol, where the
    two sectors are not separated"""
    from .topologicalsector import _hermitian_operator
    m = _gamma_matrix(h)
    w = _occupied(m,tol)
    (ex,ey) = _phases(h,m.shape[0])
    n = m.shape[0]
    op = _hermitian_operator(h,operator)
    o = np.array(algebra.todense(op(np.identity(n,dtype=complex),
                                    k=[0.,0.,0.])))
    po = np.conjugate(w.T)@o@w # P O P in the occupied states
    (s,c) = np.linalg.eigh((po+np.conjugate(po.T))/2.)
    if len(s)>0 and np.min(np.abs(s))<tol:
        raise ValueError("the occupied states cannot be split by the sign "
          +"of the operator, since its projection on them has an eigenvalue "
          +"of "+str(np.min(np.abs(s)))+"; the split Bott index is not "
          +"defined when that gap closes")
    bp = bott_of_states(w@c[:,s>0.],ex,ey)
    bm = bott_of_states(w@c[:,s<0.],ex,ey)
    return (bp-bm)/2.


def spin_bott_index(h,operator="sz",tol=1e-8):
    """Spin Bott index (B_+ - B_-)/2 of a periodic supercell at k=0, with the
    occupied states split by the sign of P s_z P (Huang and Liu,
    arXiv:1810.00081). It is the spin Chern number of a large enough
    sample, and with time reversal its parity is the Z2 invariant; it stays
    defined without spin conservation as long as P s_z P keeps a gap around
    zero. Only an eigenvalue of P s_z P within tol of zero raises, so a
    non-integer result is the sign that the two sectors are not separated.

    operator: the spin component, a name, a matrix or an Operator
    tol: the smallest accepted distance to zero of an energy level and of
        an eigenvalue of P s_z P"""
    from ..check import require_spin
    require_spin(h,"the spin Bott index")
    return split_bott(h,operator=operator,tol=tol)
