"""The matrix of the ultracell at a Bloch momentum of the ultracell.

    H_{q m, q' m'} = <B_{q,m}| H_S(k_S) |B_{q',m'}>,

the Rayleigh-Ritz matrix of the modulated supercell H_S in the folded Bloch
basis of basis.py. The clean part of H_S gives eps_m(p_q) on the diagonal,
since the basis states are its eigenstates, and the modulation couples the
folded momenta that differ by one of its harmonics. Nothing assumes the
modulation is onsite: hoppings, spin-orbit coupling or pairing that change
from cell to cell are projected the same way.

The matrix is built one column block q' at a time: H_S applied to the
basis vectors of q' (a sparse product), then an FFT over the cells gives
their components at every q at once, which costs N log N instead of the N
separate projections.
"""

import numpy as np
from scipy.linalg import eigh


def ultracell_matrix(basis,hk,kS,E,U,valid):
    """The ultracell matrix at kS, restricted to the kept states, from the
    supercell Bloch Hamiltonian hk = H_S(k_S) and the states of
    basis.states(kS)"""
    N,bmax = valid.shape
    H = np.zeros((N*bmax,N*bmax),dtype=np.complex128)
    for iq in range(N): # column block of the folded momentum q'
        if not np.any(valid[iq]): continue
        B = basis.basis_block(kS,U,iq) # (norbs,bmax)
        Y = np.asarray(hk@B) # H_S applied to the basis vectors of q'
        Yq = basis.from_supercell(kS,Y) # (N,norb,bmax), components at every q
        blocks = np.einsum("qam,qan->qmn",np.conjugate(U),Yq) # U_q^dag Y_q
        H[:,iq*bmax:(iq+1)*bmax] = blocks.reshape(N*bmax,bmax)
    v = valid.ravel()
    H = H[np.ix_(v,v)]
    return 0.5*(H + np.conjugate(H.T)) # Hermitian up to rounding


def eigenstates(basis,hk,kS):
    """Eigenvalues and eigenvectors of the ultracell at kS. The vectors are
    returned padded, (N,bmax,nstates), as coefficients c_{q,m} of the
    folded Bloch states, together with the unit-cell vectors U of
    basis.states, so that phi_q = U_q c_q"""
    E,U,valid = basis.states(kS)
    H = ultracell_matrix(basis,hk,kS,E,U,valid)
    es,vs = eigh(H)
    N,bmax = valid.shape
    C = np.zeros((N*bmax,len(es)),dtype=np.complex128)
    C[valid.ravel()] = vs
    return es,C.reshape(N,bmax,len(es)),U


def real_space_states(basis,kS,C,U):
    """The eigenvectors on the orbitals of the supercell, (norbs,nstates),
    one FFT over the folded momenta per state"""
    phi = np.einsum("qam,qmj->qaj",U,C) # components per folded momentum
    return basis.to_supercell(kS,phi)
