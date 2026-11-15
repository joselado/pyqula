"""The ultracell as a density-matrix backend of the mean field,
integration="ultracell" of scftk.densitydensity.

The self-consistent loop of scftk works on the supercell Hamiltonian, the
modulated supercell plus the mean field, and needs three things from a
backend: the density matrix at the entries the mean field reads, the Fermi
level that holds a filling, and the band energy. This module computes the
three from the ultracell states instead of the full diagonalization of the
supercell. The states are written on the supercell orbitals, and the density
matrix is summed from them with the convention of the "ed" backend (the
transpose of rho), at the entries the mean field reads, as dense matrices
for scftk.densitydensity (get_dm) or sparse ones for the sparse engines
(get_dm_sparse).

The supercell Hamiltonian carries the folded basis as the attribute
ultracell_basis, which u.get_mean_field_hamiltonian() sets and which the
copies of the loop keep. The basis is the one of the clean unit cell,
rebuilt at every call from the lattice-periodic part of the current
Hamiltonian when basis.rebuild is set, the default of
u.get_mean_field_hamiltonian(), or frozen to the clean unit cell otherwise
(see basis.FoldedBasis.rebuilt and future_development/ultra_long_range.md,
decision 5).

Each iteration diagonalizes the ultracell once per kpoint. The loop finds
the Fermi level on h and then asks for the density matrix of h shifted by
it, and a uniform shift leaves the eigenvectors as they are, so the search
keeps its states on h (get_fermi4filling(keep=True)), and get_dm and
band_energy take them as long as h has changed by a shift only. The loop
releases them (release_states) before the Hamiltonian leaves it.
"""

import numpy as np
from copy import copy
from scipy.linalg import eigh
from scipy.sparse import issparse, csr_matrix
from numba import jit, prange

from .basis import pad3
from .matrix import ultracell_matrix, real_space_states

# above this the states of the Fermi search are not kept for the density
# matrix, which then diagonalizes again, as in
# densitymatrix.full_dm_accumulate_sparse_with_fermi
max_memory_gb = 2.0

# a state above the Fermi level by this many times the smearing has an
# occupation below exp(-50), 2e-22, and is left out of the density matrix
empty_ratio = 50.


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


def loop_basis(h):
    """The basis of the ultracell of h at this iteration: the clean one, or
    the one rebuilt from the lattice-periodic part of h"""
    b = get_basis(h)
    return b.rebuilt(h) if b.rebuild else b


def sparse_hk_gen(h):
    """The Bloch Hamiltonian of h as a function of k returning a sparse
    matrix, since it is only applied to the basis vectors; the hoppings
    are made sparse once, whether h is dense or sparse"""
    hop = [(d,csr_matrix(m)) for (d,m) in h.get_multihopping().get_dict().items()]
    hop = [(d,m) for (d,m) in hop if m.nnz>0]
    g = h.geometry
    def f(k):
        k = pad3(k)
        return sum([m*g.bloch_phase(d,k) for (d,m) in hop],
                csr_matrix(h.intra.shape,dtype=np.complex128))
    return f


class MeshStates():
    """The ultracell eigenstates of h at every kpoint of a mesh: per
    kpoint the eigenvalues, the eigenvectors in the kept states and the
    unit-cell vectors of the basis"""
    def __init__(self,h,nk=1):
        self.basis = loop_basis(h)
        hkgen = sparse_hk_gen(h)
        self.ks = np.array(h.geometry.get_kmesh(nk=nk)) # the mesh of "ed"
        self.n = h.intra.shape[0] # orbitals of the supercell
        self.es,self.vs,self.Us,self.valid = [],[],[],[]
        for k in self.ks:
            E,U,valid = self.basis.states(k)
            H = ultracell_matrix(self.basis,hkgen(k),k,E,U,valid)
            es,vs = eigh(H)
            self.es.append(es) ; self.vs.append(vs)
            self.Us.append(U) ; self.valid.append(valid)
        self.shift = 0. # added to every eigenvalue, see shifted()
    def nbytes(self):
        return sum(v.nbytes + U.nbytes for (v,U) in zip(self.vs,self.Us))
    def eigenvalues(self):
        return [es + self.shift for es in self.es]
    def real_space(self,ik,emax=None):
        """Eigenvalues at kpoint ik and the eigenvectors on the supercell
        orbitals, (norbs,nstates), only those up to emax if given"""
        es = self.es[ik] + self.shift
        vs = self.vs[ik]
        if emax is not None:
            keep = es<=emax ; es,vs = es[keep],vs[:,keep]
        valid = self.valid[ik]
        N,bmax = valid.shape
        C = np.zeros((N*bmax,len(es)),dtype=np.complex128)
        C[valid.ravel()] = vs
        C = C.reshape(N,bmax,len(es))
        Phi = real_space_states(self.basis,self.ks[ik],C,self.Us[ik])
        return es,np.ascontiguousarray(Phi)


def matrix_stamp(h):
    """The matrices of h, to tell later whether h changed"""
    return (h.intra,[t.m for t in h.hopping])


def uniform_shift(h,stamp):
    """The constant c with h = h(stamp) + c, or None if h changed in any
    other way"""
    from scipy.sparse import identity
    intra0,hops0 = stamp
    hops = [t.m for t in h.hopping]
    if len(hops)!=len(hops0) or any(a is not b for a,b in zip(hops,hops0)):
        return None # a hopping was replaced
    d = h.intra - intra0
    if issparse(d):
        c = np.mean(d.diagonal()).real
        rest = (d - c*identity(d.shape[0],format="csr")).tocoo().data
    else:
        rest = np.array(d)
        c = np.mean(np.diag(rest)).real
        rest[np.diag_indices_from(rest)] -= c
    if rest.size>0 and np.max(np.abs(rest))>1e-10: return None
    return c


def keep_states(h,states):
    """Keep the states on h for the next call on the same h, unless they
    take more than max_memory_gb"""
    if states.nbytes()<max_memory_gb*1e9:
        h._ultracell_states = (matrix_stamp(h),states)


def kept_states(h,nk):
    """The states h keeps from an earlier call on this mesh, if h has only
    been shifted by a constant since, with their eigenvalues shifted to
    match; None otherwise"""
    kept = h.__dict__.get("_ultracell_states",None)
    if kept is None: return None
    stamp,states = kept
    c = uniform_shift(h,stamp)
    if c is None or not np.array_equal(states.ks,
            np.array(h.geometry.get_kmesh(nk=nk))): # not those of this mesh
        release_states(h)
        return None
    states = copy(states) # the same vectors, the eigenvalues of h
    states.shift = states.shift + c
    keep_states(h,states) # stamped with the matrices of h
    return states


def release_states(h):
    """Drop the states h keeps, so that a Hamiltonian that leaves the loop
    does not carry them"""
    h.__dict__.pop("_ultracell_states",None)


def mesh_states(h,nk):
    """The ultracell states of h on the mesh, those h keeps or new ones,
    which h then keeps"""
    states = kept_states(h,nk)
    if states is None:
        states = MeshStates(h,nk=nk)
        keep_states(h,states)
    return states


@jit(nopython=True,parallel=True,cache=True)
def pair_sums(Phi,occ,rows,cols):
    """sum_s occ[s] conj(Phi[i,s]) Phi[j,s] for every pair (i,j) =
    (rows[p],cols[p]), without the (npairs,nstates) temporaries"""
    out = np.zeros(len(rows),dtype=np.complex128)
    ns = Phi.shape[1]
    for p in prange(len(rows)):
        i = rows[p] ; j = cols[p]
        acc = 0j
        for s in range(ns): acc += occ[s]*np.conj(Phi[i,s])*Phi[j,s]
        out[p] = acc
    return out


def smearing(T):
    """The smearing of the occupations for a temperature T, as
    integration="ed" takes it"""
    from ..densitymatrix import nonzero_smearing, delta_dm
    if T is None: T = delta_dm
    return nonzero_smearing(T)


def occupied(states,ik,delta):
    """Eigenvalues, Fermi-Dirac occupations and eigenvectors on the
    supercell orbitals of the occupied states at kpoint ik"""
    es,Phi = states.real_space(ik,emax=empty_ratio*delta)
    return es,1./(1.+np.exp(es/delta)),Phi


def dm_entries(states,needed,delta):
    """The density matrix at the entries needed, {d: (rows,cols)}, as
    {d: values}, with the convention of integration="ed",
    dm[d][i,j] = (1/Nk) sum_k exp(2 pi i k.d) sum_s f_s conj(phi_s(i)) phi_s(j),
    the transpose of rho. At one kpoint the sum over the states is the same
    for every direction, so it is taken once on the union of the pairs"""
    n = states.n
    keys = {d: np.asarray(r,dtype=np.int64)*n + np.asarray(c,dtype=np.int64)
            for d,(r,c) in needed.items()}
    union = np.unique(np.concatenate([np.zeros(0,dtype=np.int64)]
        + list(keys.values())))
    ur,uc = union//n,union%n
    where = {d: np.searchsorted(union,kd) for d,kd in keys.items()}
    vals = {d: np.zeros(len(kd),dtype=np.complex128) for d,kd in keys.items()}
    for ik,k in enumerate(states.ks):
        es,occ,Phi = occupied(states,ik,delta)
        s = pair_sums(Phi,occ,ur,uc)
        for d in vals:
            vals[d] += np.exp(2j*np.pi*np.dot(k,pad3(d)))*s[where[d]]
    for d in vals: vals[d] /= len(states.ks)
    return vals


def dm_full(states,ds,delta):
    """Every entry of the density matrix in the directions ds, dense,
    with one product per kpoint shared by every direction"""
    n = states.n
    out = {d: np.zeros((n,n),dtype=np.complex128) for d in ds}
    for ik,k in enumerate(states.ks):
        es,occ,Phi = occupied(states,ik,delta)
        M = (np.conjugate(Phi)*occ[None,:])@Phi.T
        for d in ds: out[d] += np.exp(2j*np.pi*np.dot(k,pad3(d)))*M
    for d in out: out[d] /= len(states.ks)
    return out


def get_dm(h,v,nk=1,T=None,full=False):
    """Density matrix of the supercell h in its ultracell states, one
    (n,n) matrix per direction of v (a dict, or a list of directions with
    full=True), with every state below zero energy occupied with the
    Fermi-Dirac weight of temperature T, as integration="ed" builds it.
    Without full, only the entries the mean field reads are computed"""
    from ..scftk.densitydensity import dm_sparse_pairs
    get_basis(h) # check that h is the supercell of an ultracell
    delta = smearing(T)
    ds = [(0,0,0)] + [tuple(d) for d in v if tuple(d)!=(0,0,0)]
    n = h.intra.shape[0]
    states = mesh_states(h,nk) # those of the Fermi search, if kept
    if full: return dm_full(states,ds,delta)
    pairs = dm_sparse_pairs(v,ds,n,has_spin=h.has_spin)
    vals = dm_entries(states,pairs,delta)
    outd = {d: np.zeros((n,n),dtype=np.complex128) for d in pairs}
    for d,(rows,cols) in pairs.items(): outd[d][rows,cols] = vals[d]
    return outd


def get_dm_sparse(h,needed,nk=1,T=None):
    """Density matrix of the supercell h in its ultracell states at the
    entries needed, {d: (rows,cols)}, as {d: csr (n,n)}, with the
    convention of get_dm; the counterpart of
    kpmtk.densitymatrix_kpm.get_dm_kpm_sparse"""
    get_basis(h)
    n = h.intra.shape[0]
    vals = dm_entries(mesh_states(h,nk),needed,smearing(T))
    return {d: csr_matrix((vals[d],(np.asarray(r),np.asarray(c))),
        shape=(n,n),dtype=np.complex128) for d,(r,c) in needed.items()}


def fermi_level(esk,n,filling,T=0.):
    """Fermi level that holds filling, a fraction of all the n states of
    the supercell per kpoint, from the ultracell eigenvalues esk, one array
    per kpoint. The states above emax are not in the basis and count as
    empty, so the filling must leave the Fermi level below the highest
    kept state"""
    from ..spectrum import get_fermi_energy_T
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


def get_fermi4filling(h,filling,nk=1,T=0.,keep=False):
    """Fermi level of the supercell h that holds filling, a fraction of all
    the states of the supercell, from the ultracell eigenvalues. With keep,
    h keeps the states for the density matrix that the loop asks next, on
    h shifted by this level, unless they take more than max_memory_gb"""
    states = MeshStates(h,nk=nk)
    fermi = fermi_level(states.eigenvalues(),states.n,filling,T=T)
    if keep: keep_states(h,states)
    return fermi


def band_energy(h,nk=1):
    """Sum of the ultracell eigenvalues below zero energy, averaged over the
    mesh of spectrum.total_energy, its counterpart for "ed\""""
    esk = mesh_states(h,nk).eigenvalues()
    return np.mean([np.sum(e[e<0.]) for e in esk])
