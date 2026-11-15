"""The matrix of the ultracell at a Bloch momentum of the ultracell.

    H_{q m, q' m'} = <B_{q,m}| H_S(k_S) |B_{q',m'}>,

the Rayleigh-Ritz matrix of the modulated supercell H_S in the folded Bloch
basis of basis.py. The clean part of H_S gives eps_m(p_q) on the diagonal,
since the basis states are its eigenstates, and the modulation couples the
folded momenta that differ by one of its harmonics. Nothing assumes the
modulation is onsite: hoppings, spin-orbit coupling or pairing that change
from cell to cell are projected the same way.

The matrix is built one group of column blocks q' at a time: H_S applied
to the basis vectors of those q' (a sparse product), then an FFT over the
cells gives their components at every q at once, which costs N log N
instead of the N separate projections.
"""

import numpy as np
from scipy.linalg import eigh

# the states are taken to the supercell orbitals in chunks of about this
# many entries, so that the memory of the real-space states stays bounded
chunk_entries = 2**23


def ultracell_matrix(basis,hk,kS,E,U,valid):
    """The ultracell matrix at kS, on the kept states in the order of
    valid.ravel() (folded momentum first, then level), from the supercell
    Bloch Hamiltonian hk = H_S(k_S) and the states of basis.states(kS)"""
    N,bmax = valid.shape
    counts = np.sum(valid,axis=1) # kept at each momentum, the first ones
    offsets = np.concatenate([[0],np.cumsum(counts)])
    rows = valid.ravel()
    # column order, the one of LAPACK, so that eigh can work on H in place
    H = np.zeros((offsets[-1],offsets[-1]),dtype=np.complex128,order="F")
    for g0,g1 in momentum_groups(basis,counts): # consecutive momenta q'
        B = np.hstack([basis.basis_block(kS,U,iq)[:,0:counts[iq]]
            for iq in range(g0,g1)]) # their basis vectors, (norbs,nb)
        Y = np.asarray(hk@B) # H_S applied to the basis vectors
        Yq = basis.from_supercell(kS,Y) # (N,norb,nb), components at every q
        blocks = np.einsum("qam,qan->qmn",np.conjugate(U),Yq) # U_q^dag Y_q
        H[:,offsets[g0]:offsets[g1]] = blocks.reshape(N*bmax,-1)[rows]
    # Hermitian up to rounding; in place, so that only one more matrix of
    # this size is alive (the conjugate transpose is built first)
    H += np.conjugate(H.T)
    H *= 0.5
    return H


def momentum_groups(basis,counts):
    """Ranges (g0,g1) of consecutive folded momenta whose kept states, at
    most chunk_entries/norbs of them unless one momentum has more, are
    built together; momenta with no kept state are left out"""
    size = max(1,chunk_entries//(basis.N*basis.norb))
    groups,g0,nb = [],None,0
    for iq,b in enumerate(counts):
        if b==0: # nothing kept, the group ends here
            if g0 is not None: groups.append((g0,iq))
            g0,nb = None,0
            continue
        if g0 is not None and nb+b>size: # full
            groups.append((g0,iq)) ; g0,nb = None,0
        if g0 is None: g0 = iq
        nb += b
    if g0 is not None: groups.append((g0,len(counts)))
    return groups


def eigenstates(basis,hk,kS):
    """Eigenvalues and eigenvectors of the ultracell at kS. The vectors are
    the coefficients of the kept states, (nkept,nstates) in the order of
    valid.ravel(), returned with the unit-cell vectors U and the mask
    valid of basis.states, so that phi_q = U_q c_q"""
    E,U,valid = basis.states(kS)
    H = ultracell_matrix(basis,hk,kS,E,U,valid)
    es,vs = eigh(H,overwrite_a=True) # H is not used after, no copy of it
    return es,vs,U,valid


def eigenvalues(basis,hk,kS):
    """Eigenvalues of the ultracell at kS, without the eigenvectors"""
    E,U,valid = basis.states(kS)
    H = ultracell_matrix(basis,hk,kS,E,U,valid)
    return eigh(H,eigvals_only=True,overwrite_a=True) # no copy of H


def real_space_states(basis,kS,vs,U,valid):
    """The eigenvectors vs (nkept,nstates) on the orbitals of the
    supercell, (norbs,nstates), one FFT over the folded momenta per state"""
    N,bmax = valid.shape
    C = np.zeros((N*bmax,vs.shape[1]),dtype=np.complex128)
    C[valid.ravel()] = vs
    phi = np.einsum("qam,qmj->qaj",U,C.reshape(N,bmax,vs.shape[1]))
    return basis.to_supercell(kS,phi)


def state_chunks(basis,nstates):
    """Slices of the nstates states, chunk_entries/norbs states each, at
    least one"""
    norbs = basis.N*basis.norb
    size = max(1,chunk_entries//norbs)
    return [slice(i,min(i+size,nstates)) for i in range(0,nstates,size)]
