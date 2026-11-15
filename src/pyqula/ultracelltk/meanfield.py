"""The ultracell as a density-matrix backend of the mean field,
integration="ultracell" of scftk.densitydensity.

The self-consistent loop of scftk works on the supercell Hamiltonian, the
modulated supercell plus the mean field, and needs three things from a
backend: the density matrix at the entries the mean field reads, the Fermi
level that holds a filling, and the band energy. This module computes the
three from the ultracell states instead of the full diagonalization of the
supercell. The states are written on the supercell orbitals, so the density
matrix comes out of the same kernel the "ed" backend uses
(densitymatrix._accumulate_dm_batch), with the same conventions.

The supercell Hamiltonian carries the folded basis as the attribute
ultracell_basis, which u.get_mean_field_hamiltonian() sets and which the
copies of the loop keep. The basis is the one of the clean unit cell,
rebuilt at every call from the lattice-periodic part of the current
Hamiltonian when basis.rebuild is set, the default of
u.get_mean_field_hamiltonian(), or frozen to the clean unit cell otherwise
(see basis.FoldedBasis.states and future_development/ultra_long_range.md,
decision 5).
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse, csr_matrix

from .basis import pad3, dense
from .matrix import ultracell_matrix, real_space_states


def get_basis(h):
    """The folded basis that h carries"""
    b = getattr(h,"ultracell_basis",None)
    if b is None:
        raise ValueError("integration=\"ultracell\" needs the supercell "
                "Hamiltonian of an ultracell, which "
                "u.get_mean_field_hamiltonian() builds; build "
                "u = h.get_ultracell(hs) and call it there")
    if h.has_eh:
        raise NotImplementedError("the mean field of the ultracell is not "
                "implemented for a Hamiltonian with the electron-hole "
                "(Nambu) degree of freedom")
    if h.non_hermitian:
        raise NotImplementedError("the mean field of the ultracell is only "
                "implemented for a Hermitian Hamiltonian")
    return b


def sparse_hk(hkgen,k):
    """The Bloch Hamiltonian at k as a sparse matrix, since it is only
    applied to the basis vectors"""
    m = hkgen(pad3(k))
    if issparse(m): return csr_matrix(m)
    return csr_matrix(np.asarray(m))


def solve(h,hkgen,k):
    """Eigenvalues of the ultracell of h at k, and its eigenvectors on the
    supercell orbitals, (norbs,nstates)"""
    b = get_basis(h)
    hk = sparse_hk(hkgen,k)
    E,U,valid = b.states(k,hk)
    H = ultracell_matrix(b,hk,k,E,U,valid)
    es,vs = eigh(H)
    N,bmax = valid.shape
    C = np.zeros((N*bmax,len(es)),dtype=np.complex128)
    C[valid.ravel()] = vs
    Phi = real_space_states(b,k,C.reshape(N,bmax,len(es)),U)
    return es,np.ascontiguousarray(Phi)


def get_dm(h,v,nk=1,T=None,full=False):
    """Density matrix of the supercell h in its ultracell states, one
    (n,n) matrix per direction of v (a dict, or a list of directions with
    full=True), with every state below zero energy occupied with the
    Fermi-Dirac weight of temperature T, as integration="ed" builds it.
    Without full, only the entries the mean field reads are computed"""
    from ..densitymatrix import _accumulate_dm_batch, nonzero_smearing, delta_dm
    from ..scftk.densitydensity import dm_sparse_pairs
    get_basis(h) # check that h is the supercell of an ultracell
    if T is None: T = delta_dm
    if T==0.: T = 1e-15 # as densitymatrix.full_dm does
    delta = nonzero_smearing(T)
    ds = [(0,0,0)] + [tuple(d) for d in v if tuple(d)!=(0,0,0)]
    n = h.intra.shape[0]
    if full: # every entry of every direction, with the dense kernel
        allidx = np.arange(n)
        pairs = {d: (allidx,allidx) for d in ds}
        threshold = -1
    else:
        pairs = dm_sparse_pairs(v,ds,n,has_spin=h.has_spin)
        threshold = 0.01*n*n # as densitymatrix.full_dm_accumulate_sparse
    hkgen = h.get_hk_gen()
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # the mesh of "ed"
    outd = {d: np.zeros((n,n),dtype=np.complex128) for d in pairs}
    for k in ks:
        es,Phi = solve(h,hkgen,k)
        _accumulate_dm_batch(outd,pairs,threshold,es[None,:],Phi[None,:,:],
                np.array(k,dtype=np.float64)[None,:],delta)
    for d in outd: outd[d] /= len(ks)
    return outd


def eigenvalues(h,nk=1):
    """Ultracell eigenvalues of h on the mesh of spectrum.eigenvalues, and
    the number of states of the supercell per kpoint"""
    from ..klist import kmesh
    hkgen = h.get_hk_gen()
    out = [] ; ks = kmesh(h.dimensionality,nk=nk)
    b = get_basis(h)
    for k in ks:
        hk = sparse_hk(hkgen,k)
        E,U,valid = b.states(k,hk)
        out.append(eigh(ultracell_matrix(b,hk,k,E,U,valid),eigvals_only=True))
    return out,h.intra.shape[0]


def get_fermi4filling(h,filling,nk=1,T=0.):
    """Fermi level that holds filling, a fraction of all the states of the
    supercell, from the ultracell eigenvalues. The states above emax are
    not in the basis and count as empty, so the filling must leave the
    Fermi level below the highest kept state"""
    from ..spectrum import get_fermi_energy_T
    esk,n = eigenvalues(h,nk=nk)
    kept = sum(len(e) for e in esk)
    total = n*len(esk)
    ntarget = int(round(total*filling))
    if ntarget>=kept:
        raise ValueError("filling="+str(filling)+" needs "+str(ntarget)
                +" occupied states, and the ultracell keeps "+str(kept)
                +" on this mesh; build it with a larger emax")
    top = max(np.max(e) for e in esk) + 100. # above every kept state
    es = np.concatenate(esk + [np.full(total-kept,top)])
    return get_fermi_energy_T(es,filling,T=T)


def band_energy(h,nk=1):
    """Sum of the ultracell eigenvalues below zero energy, averaged over the
    mesh of spectrum.total_energy, its counterpart for "ed\""""
    esk,n = eigenvalues(h,nk=nk)
    return np.mean([np.sum(e[e<0.]) for e in esk])
