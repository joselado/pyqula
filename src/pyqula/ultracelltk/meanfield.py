"""The ultracell as a backend of the mean field, integration="ultracell".

The self-consistent loops of scftk work on the supercell Hamiltonian, the
modulated supercell plus the mean field, and need three things from a
backend: the density matrix at the entries the mean field reads, the Fermi
level that holds a filling, and the band energy. SparseBackend computes the
three from the ultracell states instead of the full diagonalization of the
supercell, with one diagonalization of the ultracell per kpoint and
iteration. The states are written on the supercell orbitals, and the density
matrix is summed from them with the convention of the "ed" backend (the
transpose of rho), as csr matrices at the entries asked for, so that the
loops build no dense matrix of the supercell. A spinful supercell runs on
the sparse engine of scftk.spinspin (VJinteraction), which has the
spin-spin exchange, and a spinless one on the sparse loop of
scftk.densitydensity_kpm.

The supercell Hamiltonian carries the folded basis as the attribute
ultracell_basis, which u.get_mean_field_hamiltonian() sets and which the
copies of the loop keep. The basis is the one of the clean unit cell,
rebuilt at every iteration from the lattice-periodic part of the current
Hamiltonian when basis.rebuild is set, the default of
u.get_mean_field_hamiltonian(), or frozen to the clean unit cell otherwise
(see basis.FoldedBasis.rebuilt and future_development/ultra_long_range.md,
decision 5).
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from numba import jit, prange

from .basis import pad3
from .matrix import ultracell_matrix, real_space_states, state_chunks

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
            # H is in column order and not used after, so eigh works on
            # it in place instead of on a copy
            es,vs = eigh(H,overwrite_a=True)
            del H
            self.es.append(es) ; self.vs.append(vs)
            self.Us.append(U) ; self.valid.append(valid)
        self.shift = 0. # added to every eigenvalue, see shifted()
    def eigenvalues(self):
        return [es + self.shift for es in self.es]
    def real_space(self,ik,vs):
        """The eigenvectors vs of kpoint ik, columns of self.vs[ik], on
        the supercell orbitals, (norbs,nstates)"""
        return np.ascontiguousarray(real_space_states(self.basis,self.ks[ik],
            vs,self.Us[ik],self.valid[ik]))


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
    """The Fermi-Dirac occupations and the eigenvectors on the supercell
    orbitals of the occupied states at kpoint ik, a chunk of states at a
    time, so that the real-space states never take more than a chunk"""
    es = states.es[ik] + states.shift
    # eigh sorts the eigenvalues, so the occupied states are the first
    # ones and a slice of the eigenvectors is a view, not a copy
    nocc = int(np.sum(es<=empty_ratio*delta))
    occ,vs = 1./(1.+np.exp(es[0:nocc]/delta)),states.vs[ik][:,0:nocc]
    for sl in state_chunks(states.basis,len(occ)):
        yield occ[sl],states.real_space(ik,vs[:,sl])


def dm_entries(states,needed,delta):
    """The density matrix at the entries needed, {d: (rows,cols)}, as
    {d: values}, with the convention of integration="ed",
    dm[d][i,j] = (1/Nk) sum_k exp(2 pi i k.d) sum_s f_s conj(phi_s(i)) phi_s(j),
    the transpose of rho. At one kpoint the sum over the states is the same
    for every direction, so it is taken once on the union of the pairs.
    At one kpoint the sum of (j,i) is the conjugate of the sum of (i,j),
    and the entry at d and its Fock partner at -d ask for both, so the
    sums are taken on the pairs with i<=j and the others are conjugated"""
    n = states.n
    rc = {d: (np.asarray(r,dtype=np.int64),np.asarray(c,dtype=np.int64))
            for d,(r,c) in needed.items()}
    keys = {d: np.minimum(r,c)*n + np.maximum(r,c) for d,(r,c) in rc.items()}
    lower = {d: r>c for d,(r,c) in rc.items()} # conjugate of (c,r)
    union = np.unique(np.concatenate([np.zeros(0,dtype=np.int64)]
        + list(keys.values())))
    ur,uc = union//n,union%n
    where = {d: np.searchsorted(union,kd) for d,kd in keys.items()}
    vals = {d: np.zeros(len(kd),dtype=np.complex128) for d,kd in keys.items()}
    for ik,k in enumerate(states.ks):
        s = np.zeros(len(union),dtype=np.complex128)
        for occ,Phi in occupied(states,ik,delta):
            s = s + pair_sums(Phi,occ,ur,uc)
        for d in vals:
            sd = s[where[d]] # a copy
            sd[lower[d]] = np.conjugate(sd[lower[d]])
            vals[d] += np.exp(2j*np.pi*np.dot(k,pad3(d)))*sd
    for d in vals: vals[d] /= len(states.ks)
    return vals


def dm_full(states,ds,delta):
    """Every entry of the density matrix in the directions ds, dense,
    with one product per kpoint shared by every direction"""
    n = states.n
    out = {d: np.zeros((n,n),dtype=np.complex128) for d in ds}
    for ik,k in enumerate(states.ks):
        M = np.zeros((n,n),dtype=np.complex128)
        for occ,Phi in occupied(states,ik,delta):
            M = M + (np.conjugate(Phi)*occ[None,:])@Phi.T
        for d in ds: out[d] += np.exp(2j*np.pi*np.dot(k,pad3(d)))*M
    for d in out: out[d] /= len(states.ks)
    return out


class SparseBackend():
    """The ultracell backend of the sparse engines of the mean field: the
    density matrix at the entries the mean field reads, as csr matrices,
    with the Fermi level that holds a filling, and the band energy, from
    one diagonalization of the ultracell per kpoint and iteration. It
    keeps the states of the last Hamiltonian it solved, for the density
    matrix and the band energy of that same Hamiltonian.

    The spinful loop (scftk.spinspin._run_anisotropic_scf) calls
    dm_with_fermi, or dm after its own shift by mu. The spinless loop
    (scftk.densitydensity_kpm.generic_densitydensity_kpm) takes it as its
    fermi object, the interface of kpmtk.densitymatrix_kpm.LaggedFermi:
    shift(h) finds the Fermi level of h at self.filling exactly and shifts
    h by it, so its error is zero, and dm(h) then takes the same states"""
    def __init__(self,nk=1,T=None,filling=None):
        self.nk = nk
        # the occupations and the Fermi search at the same smearing, as
        # densitymatrix.full_dm_accumulate_sparse_with_fermi does for the
        # spinful "ed", so that a level at the Fermi energy holds the
        # charge the filling asks for and not half of the level
        self.delta = smearing(T)
        self.filling = filling # for shift()
        self.error = 0. # the Fermi level is exact, see shift()
        self.release()
    def release(self):
        """Drop the states kept"""
        self.states,self.h = None,None
    def solve(self,h):
        self.states,self.h = MeshStates(h,nk=self.nk),h
        return self.states
    def entries(self,needed):
        n = self.states.n
        vals = dm_entries(self.states,needed,self.delta)
        return {d: csr_matrix((vals[d],(np.asarray(r),np.asarray(c))),
            shape=(n,n),dtype=np.complex128)
            for d,(r,c) in needed.items()}
    def dm_with_fermi(self,h,needed,filling):
        """The density matrix of h shifted by its Fermi level at filling,
        and that level; the caller shifts h by it, as for
        densitymatrix.full_dm_accumulate_sparse_with_fermi"""
        states = self.solve(h)
        fermi = fermi_level(states.eigenvalues(),states.n,filling,
                T=self.delta)
        states.shift = -fermi # the eigenvalues of the shifted h
        return self.entries(needed),fermi
    def dm(self,h,needed):
        """The density matrix of h, already shifted by its chemical
        potential, from the states of shift() if they were those of h.
        Otherwise the caller shifted h by a fixed mu, which with a cut
        must lie below the highest kept state, as fermi_level requires
        of a filling"""
        if h is not self.h: # a fixed chemical potential, at zero energy
            states = self.solve(h)
            require_kept_above(states.eigenvalues(),states.n,self.delta,
                    states.basis.emax,"the chemical potential")
        return self.entries(needed)
    def shift(self,h):
        """h shifted by its Fermi level at self.filling, which h.fermi
        keeps; the states stay for dm(h)"""
        states = self.solve(h)
        fermi = fermi_level(states.eigenvalues(),states.n,self.filling,
                T=self.delta)
        states.shift = -fermi # the eigenvalues of the shifted h
        h.fermi = fermi
        h.shift_fermi(-fermi) # the eigenvectors are unaffected
        return h
    def wants_trace(self,h): return False # no lagged level to move
    def update(self,h,trace): pass
    def band_energy(self,h):
        """Sum of the eigenvalues of h below zero energy, the occupied
        states at T=0, averaged over the mesh, as h.get_total_energy() for
        "ed"; from the states of the last density matrix if it was of h"""
        states = self.states if h is self.h else MeshStates(h,nk=self.nk)
        return np.mean([np.sum(e[e<0.]) for e in states.eigenvalues()])


def require_kept_above(esk,n,delta,emax,what):
    """With a cut, the Fermi level must lie below the highest kept state,
    as fermi_level requires of a filling: the states above emax are not
    in the basis and count as empty, and when every kept state is full
    the states cut away can lie below the Fermi level too. esk are the
    ultracell eigenvalues per kpoint, measured from the Fermi level, of a
    supercell with n states per kpoint, occupied with the smearing delta;
    a state is full empty_ratio smearings below the Fermi level. The
    count is over the whole mesh, as in fermi_level, since a kpoint can
    keep no state at all"""
    kept = sum(len(e) for e in esk)
    if kept==n*len(esk): return # every state kept, nothing cut away
    tops = [np.max(e) for e in esk if len(e)>0]
    if len(tops)>0 and max(tops)>-empty_ratio*delta: return
    where = "" if len(tops)==0 else (" (the highest is "
            +str(-max(tops))+" below it)")
    raise ValueError(what+" is above every state the ultracell keeps on "
            "this mesh"+where+", so the states that emax="+str(emax)
            +" cuts away can lie below it and be occupied; build the "
            "ultracell with a larger emax")


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
    if ntarget>=kept and kept<total: # a cut, and states above it needed
        raise ValueError("filling="+str(filling)+" needs "+str(ntarget)
                +" occupied states, and the ultracell keeps "+str(kept)
                +" on this mesh; build it with a larger emax")
    # above every kept state; a kpoint where the cut keeps nothing has none
    top = max(np.max(e) for e in esk if len(e)>0) + 100.
    es = np.concatenate(esk + [np.full(total-kept,top)])
    return get_fermi_energy_T(es,filling,T=T)
