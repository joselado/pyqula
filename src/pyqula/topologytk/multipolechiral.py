"""Multipole chiral number of a chiral-symmetric two-dimensional insulator

W. A. Benalcazar and A. Cerjan, Chiral-symmetric higher-order topological
phases of matter, Phys. Rev. Lett. 128, 127601 (2022), arXiv:2109.06892,
Eqs. (3) to (5) of the main text and section II of the Supplemental Material

The chiral operator S, a unitary with S^2=1 acting inside each unit cell,
splits the states of every cell into the sublattices A (S=+1) and B (S=-1).
In that basis the Hamiltonian is off-diagonal,

    H = ( 0     h )
        ( h^dag 0 ),

and the singular value decomposition h = U_A Sigma U_B^dag pairs every state
of A with one of B, the two halves of the levels +-Sigma. The sample is a
supercell of L_x by L_y unit cells, and x, y are the integer indices of the
unit cell of every orbital, counted from 0 to L-1 from the corner of the
supercell. The sublattice quadrupole operators are the diagonal matrices

    Q_s = sum_{R, alpha in s} |R,alpha> exp(-2 pi i x y / L_x L_y) <R,alpha|,

for s = A, B, and the number is

    N_xy = Tr Log( Qbar_A Qbar_B^dag ) / 2 pi i,   Qbar_s = U_s^dag Q_s U_s.

U_A, U_B and Q_s are unitary, so the product is unitary with determinant
det(Q_A) det(Q_B)^* = 1 when every cell holds as many A as B states, and
the sum of the principal phases of its eigenvalues is a multiple of 2 pi:
N_xy comes out an integer to machine precision, and it is the right
integer only in a sample larger than the localization length. The product
equals, up to the similarity U_A, Q_A q Q_B^dag q^dag with
q = U_A U_B^dag = h (h^dag h)^(-1/2) the flattened h, so N_xy does not
depend on the freedom of the singular vectors as long as no singular value
is zero.

Boundary conditions. The main text defines N_xy in the bulk, the sample
closed into a torus (the supercell at k=0, as for the Bott index), where a
gapped sample has no zero modes and q is unique. The phase
exp(-2 pi i x y/L_x L_y) jumps across the seam of the torus, like the
quadrupole operators it is built after, and the number is quantized
anyway. The Supplemental Material computes it in the open flake, where the
corner modes give singular values close to zero, whose pairing is
arbitrary. Q_s is nearly constant across each corner mode, varying by
about 2 pi xi/L over its localization length xi, and at the two corners
with x=0 or y=0 it is close to the identity, so it nearly commutes with
that freedom and the modes barely enter. Both are built here:
boundary="periodic" (the default) and boundary="open".

Size. Even with the corner modes on a single site (the fully dimerized BBH
model) the corner origin gives N_xy = round((L_x-1)(L_y-1)/(L_x L_y)),
which is 0 for L=2 or 3 and exactly 1/2 at (3,4), so the sample needs at
least 4 unit cells along each direction.

Sign. N_xy changes sign when A and B are exchanged (S -> -S), and it
depends on the corner chosen as the origin of x and y (a centered origin
gives 0 in the topological phase of the BBH model). In the open flake
|N_xy| zero modes sit at each corner, all on one sublattice: on A at the
corners (0,0) and (L_x,L_y) when N_xy>0, and on B at the other two, so the
flake has as many zero modes on A as on B. The sublattice labels of
g.get_sublattice() start from a random site and have a random global sign,
so with chiral=None (or "sublattice") the sign is pinned with the first
orbital on A.
"""
import numpy as np
from .. import algebra

_BOUNDARIES = ("periodic","open") # accepted values of boundary=


def get_boundary_names():
    """Accepted values of boundary= in multipole_chiral_number"""
    return list(_BOUNDARIES)


def check_boundary(boundary):
    """Return boundary if it is one of the accepted names, raise if not"""
    if boundary not in _BOUNDARIES:
        raise ValueError("boundary="+repr(boundary)+" is not one of the "
          +"accepted values "+str(get_boundary_names()))
    return boundary


def cell_indices(h,cells):
    """Integer indices (x,y) of the unit cell of every orbital of a sample
    of cells=(Lx,Ly) unit cells, an (n,2) array from 0 to L-1. The
    fractional coordinate of an orbital along a1 of the sample, times Lx,
    is its cell index plus its fractional position inside the cell, so the
    cell index is its integer part measured from the smallest one"""
    from .qgt import _orbital_fractions
    n = h.intra.shape[0]
    f = _orbital_fractions(h,n)*np.array(cells)[None,:] # (n,2)
    idx = np.floor(f-np.min(f,axis=0)[None,:]+1e-6).astype(int)
    nc = cells[0]*cells[1]
    count = np.bincount(idx[:,0]*cells[1]+idx[:,1],minlength=nc)
    if np.max(idx[:,0])>=cells[0] or np.max(idx[:,1])>=cells[1] or \
            np.min(count)!=np.max(count):
        raise ValueError("the "+str(n)+" orbitals of this sample do not "
          +"split into "+str(cells[0])+" by "+str(cells[1])+" unit cells "
          +"with the same number of orbitals each; build the sample with "
          +"h.get_supercell(cells) and pass the same cells")
    return idx


def _chiral_matrix(h,chiral):
    """Dense chiral operator of the sample, checked to be Hermitian, to
    square to the identity and to anticommute with the Hamiltonian at
    k=0, the sample closed into a torus, and inside the sample"""
    from ..topology import get_operator
    from .. import check
    n = h.intra.shape[0]
    if callable(chiral) and not hasattr(chiral,"get_matrix"):
        raise TypeError("the chiral operator has to be a name, a matrix or "
          +"an Operator with a matrix, got the function "+repr(chiral))
    default = chiral is None or (isinstance(chiral,str)
            and chiral=="sublattice")
    if default:
        check.require_sublattice(h,"the multipole chiral number with the "
          +"sublattice as its chiral operator")
        S = h.get_operator("sublattice").get_matrix()
    else: S = get_operator(h,chiral).get_matrix()
    S = np.array(algebra.todense(S),dtype=complex)
    # g.get_sublattice() two-colors the lattice from a random site, so its
    # global sign is random and with it the sign of N_xy; pin the first
    # orbital on A
    if default and S.shape==(n,n) and np.real(S[0,0])<0.: S = -S
    if S.shape!=(n,n):
        raise ValueError("the chiral operator has shape "+str(S.shape)
          +" and the sample has "+str(n)+" orbitals")
    if np.max(np.abs(S@S-np.identity(n)))>1e-6 or \
            np.max(np.abs(S-np.conjugate(S.T)))>1e-6:
        raise ValueError("the chiral operator has to be Hermitian and "
          +"square to the identity")
    for m in [h.get_hk_gen()([0.,0.,0.]),h.intra]:
        m = np.array(algebra.todense(m),dtype=complex)
        r = np.max(np.abs(S@m+m@S))
        if r>1e-6*max(1.,np.max(np.abs(m))):
            raise ValueError("the chiral operator does not anticommute "
              +"with the Hamiltonian, the residual of S H + H S is "+str(r)
              +", so this Hamiltonian has no such chiral symmetry and no "
              +"multipole chiral number")
    return S


def _sublattice_states(S,idx,cells):
    """States of the sublattices A (S=+1) and B (S=-1), as two (n,n/2)
    matrices of orthonormal columns, and the cell of every column. S has to
    act inside each unit cell, so that every column sits in one cell and
    has a quadrupole phase"""
    n = S.shape[0]
    cell = idx[:,0]*cells[1]+idx[:,1] # one label per orbital
    if np.max(np.abs(S[cell[:,None]!=cell[None,:]]))>1e-6:
        raise ValueError("the chiral operator mixes different unit cells, "
          +"and the multipole chiral number needs it to act inside each "
          +"unit cell")
    va,vb,ca,cb = [],[],[],[]
    for c in range(cells[0]*cells[1]):
        io = np.where(cell==c)[0]
        (es,vs) = np.linalg.eigh(S[np.ix_(io,io)])
        na,nb = np.sum(es>0.),np.sum(es<0.)
        if na!=nb:
            raise ValueError("the chiral operator has "+str(na)+" states "
              +"with S=+1 and "+str(nb)+" with S=-1 in the unit cell "
              +str(divmod(c,cells[1]))+", and the multipole chiral number "
              +"needs as many on each sublattice in every unit cell")
        for (vv,cc,sel) in [(va,ca,es>0.),(vb,cb,es<0.)]:
            w = np.zeros((n,np.sum(sel)),dtype=complex)
            w[io,:] = vs[:,sel]
            vv.append(w)
            cc += [c]*int(np.sum(sel))
    return np.concatenate(va,axis=1),np.concatenate(vb,axis=1),\
            np.array(ca),np.array(cb)


def multipole_chiral_number_matrix(m,S,idx,cells,tol=None):
    """N_xy of the dense matrix m of a sample of cells=(Lx,Ly) unit cells,
    with S the chiral operator and idx the (n,2) cell indices of the
    orbitals (see cell_indices). With tol a number, raise if a level is
    closer to zero than tol, where the pairing of A and B is not defined"""
    (va,vb,ca,cb) = _sublattice_states(S,idx,cells)
    x,y = np.divmod(np.arange(cells[0]*cells[1]),cells[1])
    phase = np.exp(-2j*np.pi*x*y/(cells[0]*cells[1])) # one per cell
    hab = np.conjugate(va.T)@m@vb # the block h between A and B
    (ua,sig,ubh) = np.linalg.svd(hab) # h = U_A Sigma U_B^dag
    if tol is not None and np.min(sig)<tol:
        raise ValueError("the Hamiltonian has a level at E="
          +str(np.min(sig))+" on the torus, closer to zero than tol="
          +str(tol)+", so the gap is closed and the number is not defined; "
          +"move away from the transition or change the size of the sample")
    ub = np.conjugate(ubh.T)
    qa = np.conjugate(ua.T)@(phase[ca][:,None]*ua) # projected Q_A
    qb = np.conjugate(ub.T)@(phase[cb][:,None]*ub) # projected Q_B
    ev = np.linalg.eigvals(qa@np.conjugate(qb.T))
    return float(np.sum(np.angle(ev))/(2.*np.pi))


def multipole_chiral_number(h,cells=None,chiral=None,boundary="periodic",
        tol=1e-8):
    """Multipole chiral number N_xy of a sample of a two-dimensional
    chiral-symmetric Hamiltonian, a supercell of cells=(Lx,Ly) unit cells.
    In a large enough sample it is the number of zero modes at each corner
    of the open flake, with the sign giving their sublattice

    - cells: (Lx,Ly), the number of unit cells of the sample along a1 and
      a2, as in h0.get_supercell(cells); it fixes the unit cell of every
      orbital and the L_x L_y of the quadrupole phase
    - chiral: the chiral operator of the sample, a name, a matrix or an
      Operator, acting inside each unit cell; None takes the sublattice
      operator of the geometry, with the first orbital on A
    - boundary: "periodic" closes the sample into a torus, the definition of
      the paper; "open" takes the open flake
    - tol: with boundary="periodic", raise if a level of the torus is closer
      to zero than tol, where the gap is closed; a closing at a momentum
      that the torus of Lx x Ly cells does not sample is not detected

    Returns a float, not rounded, as h.get_bott_index() does. It is an
    integer to machine precision by construction, so a sample smaller than
    the localization length of the corner modes shows up as a wrong
    integer, not as a fraction: compare two sizes"""
    boundary = check_boundary(boundary)
    if h.dimensionality!=2:
        raise ValueError("the multipole chiral number needs a "
          +"two-dimensional periodic Hamiltonian, and this one has "
          +"dimensionality "+str(h.dimensionality))
    if cells is None or len(cells)!=2 or min(cells)<4 or \
            np.max(np.abs(np.array(cells)-np.round(cells)))>0:
        raise ValueError("cells has to be (Lx,Ly), the number of unit "
          +"cells of the sample, two integers of at least 4 (with the "
          +"corner as the origin a smaller sample gives a wrong integer "
          +"even for corner modes on one site), got "+str(cells))
    cells = (int(cells[0]),int(cells[1]))
    idx = cell_indices(h,cells)
    S = _chiral_matrix(h,chiral)
    if boundary=="periodic": m = h.get_hk_gen()([0.,0.,0.])
    else: # only the hoppings inside the sample, the corner modes sit at
        m = h.intra # zero energy and are allowed, so no tol
        tol = None
    m = np.array(algebra.todense(m),dtype=complex)
    return multipole_chiral_number_matrix(m,S,idx,cells,tol=tol)
