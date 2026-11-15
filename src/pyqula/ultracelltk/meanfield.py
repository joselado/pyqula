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

With a window (emin,emax) the states of the basis below emin are frozen:
they are full, they do not take part in the ultracell matrix, and their
density matrix is that of the unit cell tiled on the supercell
(frozen_entries), which only enters the lattice-periodic part of the mean
field. The paper has no frozen states; they are an extension of ours. The
frozen states and the kept ones come from the same diagonalization of the
unit cell, so the frozen density follows the basis: computed from the
clean states with the clean basis, and from the rebuilt ones at every
iteration otherwise. The Fermi search counts them as full levels below
every kept one, and the band energy adds their energies in the
lattice-periodic part of the Hamiltonian.

A Nambu (BdG) supercell takes the cut |E|<=emax from the zero of the BdG
Hamiltonian, on the BdG states of the lattice-periodic part of the shifted
Hamiltonian, counted again by energy at every iteration (basis.center).
The gap equation sums over every BdG state, so the frozen ones below -emax
carry their anomalous block too, and a uniform paired state is exact for
any emax. Particle-hole symmetry maps kS to -kS, the mesh is closed under
it, and the symmetric window maps the frozen states at p to the dropped
ones at -p, so the kept space is closed over the mesh. Two limits are
worth knowing. The frozen states do not respond, so the modulated part of
the pairing converges only slowly, about logarithmically, with emax. And
the count by energy can change the number of kept states by a multiplet
between iterations.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from numba import jit, prange

from .basis import pad3, HoppingSum, eigh_momenta
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


def periodic_part(basis,h):
    """The lattice-periodic part of h as a HoppingSum, where the energies
    of the frozen states of a clean basis are read (frozen_levels). None
    when there is no lower edge, or for a rebuilt basis, whose frozen
    states are eigenstates of that part already"""
    if basis.emin is None or basis.hopp is not None: return None
    return HoppingSum(basis.periodic_hoppings(h),basis.norb,
            dim=basis.dimensionality)


def frozen_levels(basis,kS,Ef,Uf,validf,hp):
    """The energies of the frozen states at kS in the lattice-periodic part
    H_P of the Hamiltonian, the diagonal of Uf^dag H_P(p_q) Uf, as one
    array. They are the frozen levels themselves for a rebuilt basis
    (hp=None), and, for the clean one, their expectation values in hp, the
    periodic_part of the Hamiltonian, which a uniform Hartree shift or a
    uniform moment moves away from the clean levels. The block of each
    folded momentum on itself is H_P(p_q), so these are also the diagonal
    of the supercell Hamiltonian in the frozen states"""
    if hp is None: return Ef[validf]
    Hp = hp(basis.folded_momenta(kS))
    e = np.real(np.einsum("qam,qab,qbm->qm",np.conjugate(Uf),Hp,Uf))
    return e[validf]


def electron_trace(hk):
    """Tr h_e(k), the electron diagonal of the Bloch Hamiltonian hk of a
    Nambu supercell, whose electrons are the orbitals 4i and 4i+1"""
    d = np.asarray(hk.diagonal()).ravel()
    return float(np.real(np.sum(d[np.arange(len(d))%4<2])))


def mesh_levels(basis,hsum,ks):
    """Every level of the unit-cell Bloch Hamiltonian hsum (a HoppingSum)
    at the folded momenta of every kS of the mesh ks, as one array"""
    return np.concatenate([eigh_momenta(hsum,basis.folded_momenta(k))[0]
        .ravel() for k in ks])


class MeshStates():
    """The ultracell eigenstates of h at every kpoint of a mesh: per
    kpoint the eigenvalues, the eigenvectors in the kept states and the
    unit-cell vectors of the basis, and the frozen states below the
    window, their vectors and their energies (frozen_levels), from the
    same diagonalization of the unit cell as the kept ones. For a Nambu h
    it also keeps Tr h_e of every kpoint, for the band energy"""
    def __init__(self,h,nk=1):
        self.basis = loop_basis(h)
        hkgen = sparse_hk_gen(h)
        self.ks = np.array(h.geometry.get_kmesh(nk=nk)) # the mesh of "ed"
        self.n = h.intra.shape[0] # orbitals of the supercell
        self.has_eh = h.has_eh
        self.es,self.vs,self.Us,self.valid = [],[],[],[]
        self.frozen,self.ef,self.trhe = [],[],[]
        hp = periodic_part(self.basis,h)
        for k in self.ks:
            E,U,valid,Ef,Uf,validf = self.basis.states(k,frozen=True)
            hk = hkgen(k)
            if h.has_eh: self.trhe.append(electron_trace(hk))
            H = ultracell_matrix(self.basis,hk,k,E,U,valid)
            # H is in column order and not used after, so eigh works on
            # it in place instead of on a copy
            es,vs = eigh(H,overwrite_a=True)
            del H
            self.es.append(es) ; self.vs.append(vs)
            self.Us.append(U) ; self.valid.append(valid)
            self.frozen.append(Uf) # padded with zero columns
            self.ef.append(frozen_levels(self.basis,k,Ef,Uf,validf,hp))
        self.nfrozen = sum(len(e) for e in self.ef) # on the whole mesh
        self.shift = 0. # added to every eigenvalue, see shifted()
    def eigenvalues(self):
        return [es + self.shift for es in self.es]
    def frozen_eigenvalues(self):
        """The energies of the frozen states, as eigenvalues()"""
        return [ef + self.shift for ef in self.ef]
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
    if states.nfrozen>0: # the frozen states below the window, all full
        fz = frozen_entries(states.basis,states.ks,states.frozen,rc)
        for d in vals: vals[d] += fz[d]
    return vals


def dm_full(states,ds,delta):
    """Every entry of the density matrix in the directions ds, dense,
    with one product per kpoint shared by every direction, the frozen
    states included"""
    n = states.n
    out = {d: np.zeros((n,n),dtype=np.complex128) for d in ds}
    for ik,k in enumerate(states.ks):
        M = np.zeros((n,n),dtype=np.complex128)
        for occ,Phi in occupied(states,ik,delta):
            M = M + (np.conjugate(Phi)*occ[None,:])@Phi.T
        for d in ds: out[d] += np.exp(2j*np.pi*np.dot(k,pad3(d)))*M
    for d in out: out[d] /= len(states.ks)
    if states.nfrozen>0: # every entry of the frozen density matrix
        r,c = [x.ravel() for x in np.indices((n,n))]
        fz = frozen_entries(states.basis,states.ks,states.frozen,
                {d: (r,c) for d in ds})
        for d in out: out[d] += fz[d].reshape(n,n)
    return out


def frozen_entries(basis,ks,frozen,needed):
    """The density matrix of the frozen states at the entries needed,
    {D: (rows,cols)}, as {D: values}, in the convention of dm_entries.
    frozen holds the frozen unit-cell vectors Uf (N,norb,fmax) of every kS
    of the mesh ks, padded with zero columns (basis.states). The frozen
    states are full, so their density matrix is that of the unit cell
    tiled on the supercell,

        dmf[D][i,j] = 1/(Nk N) sum_{kS,q} exp(2 pi i p_q.d_ij)
                      rho0(p_q)[alpha_i,alpha_j],
        rho0(p)[a,b] = sum_{m frozen} conj(u_m(p)[a]) u_m(p)[b],
        d_ij = R[j] - R[i] + D M,

    which is the sum over the frozen basis vectors of every kS, since
    exp(2 pi i kS.D) = exp(2 pi i p_q.(D M)), and the inverse map of
    FoldedBasis.periodic_hoppings, with the same d. The sum runs on the
    folded momenta of the mesh of the ultracell, so it is the frozen part
    of the same density matrix as the kept states. It is computed once
    per distinct displacement d, which the entries share"""
    keys = list(needed)
    ds = [] # the displacement of every entry, direction by direction
    for D in keys:
        r,c = [np.asarray(x,dtype=np.int64) for x in needed[D]]
        d = basis.R[c] - basis.R[r] + (pad3(D)@basis.M)[None,:]
        ds.append(np.round(d).astype(int).reshape(-1,3))
    dall = np.concatenate([np.zeros((0,3),dtype=int)] + ds)
    if len(dall)==0: return {D: np.zeros(0,dtype=np.complex128) for D in keys}
    dist,inv = np.unique(dall,axis=0,return_inverse=True)
    inv = np.ravel(inv)
    norb = basis.norb
    rho = np.zeros((len(dist),norb*norb),dtype=np.complex128)
    for k,Uf in zip(ks,frozen):
        if Uf.shape[2]==0: continue # nothing frozen at this kS
        rho0 = np.einsum("qam,qbm->qab",np.conjugate(Uf),Uf)
        ph = np.exp(2j*np.pi*(dist@basis.folded_momenta(k).T)) # (nd,N)
        rho += ph@rho0.reshape(basis.N,norb*norb)
    rho = rho.reshape(len(dist),norb,norb)/(len(ks)*basis.N)
    out,i0 = {},0
    for D,d in zip(keys,ds):
        r,c = [np.asarray(x,dtype=np.int64) for x in needed[D]]
        out[D] = rho[inv[i0:i0+len(d)],basis.alpha[r],basis.alpha[c]]
        i0 += len(d)
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
    h by it, so its error is zero, and dm(h) then takes the same states.
    A Nambu h is not shifted rigidly by a chemical potential, so the
    spinful loop shifts it first and calls dm(h), with the Fermi level of
    superscf.BdGFermiLevel, whose search and compressibility are
    normal_fermi and normal_compressibility"""
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
        fermi = self.fermi(self.solve(h),filling)
        return self.entries(needed),fermi
    def fermi(self,states,filling):
        """The Fermi level of the states at filling, by which their
        eigenvalues are then shifted; the frozen states must lie below it"""
        if states.has_eh:
            raise NotImplementedError("a Nambu Hamiltonian is not shifted "
                    "rigidly by a chemical potential (-mu on the electrons, "
                    "+mu on the holes), so its states cannot be shifted "
                    "after the search; shift it first, with "
                    "superscf.BdGFermiLevel or h.shift_fermi(-mu), and call "
                    "dm(h)")
        fermi = fermi_level(states.eigenvalues(),states.n,filling,
                T=self.delta,nfrozen=states.nfrozen)
        states.shift = -fermi # the eigenvalues of the shifted h
        require_frozen_below(states.frozen_eigenvalues(),self.delta,
                states.basis.emin,"the Fermi level of filling="
                +str(filling))
        return fermi
    def dm(self,h,needed):
        """The density matrix of h, already shifted by its chemical
        potential, from the states of shift() if they were those of h.
        Otherwise the caller shifted h by a fixed mu, which with a cut
        must lie below the highest kept state and above the frozen ones,
        as fermi_level requires of a filling"""
        if h is not self.h: # a fixed chemical potential, at zero energy
            states = self.solve(h)
            require_kept_above(states.eigenvalues(),states.n,self.delta,
                    states.basis.emax,"the chemical potential",
                    nfrozen=states.nfrozen)
            require_frozen_below(states.frozen_eigenvalues(),self.delta,
                    states.basis.emin,"the chemical potential")
        return self.entries(needed)
    def shift(self,h):
        """h shifted by its Fermi level at self.filling, which h.fermi
        keeps; the states stay for dm(h)"""
        fermi = self.fermi(self.solve(h),self.filling)
        h.fermi = fermi
        h.shift_fermi(-fermi) # the eigenvectors are unaffected
        return h
    def wants_trace(self,h): return False # no lagged level to move
    def update(self,h,trace): pass
    def band_energy(self,h):
        """Sum of the eigenvalues of h below zero energy, the occupied
        states at T=0, averaged over the mesh, as h.get_total_energy() for
        "ed"; from the states of the last density matrix if it was of h.
        The frozen states are full and add Tr[P_f H_P], their energies in
        the lattice-periodic part of h (frozen_levels), so that the band
        energy is Tr[rho H] for the density matrix of dm_entries. For a
        Nambu h it is (S + Tr h_e)/2 at every kpoint, as
        spectrum.total_energy, with S that sum and Tr h_e the electron
        diagonal of the Bloch Hamiltonian of the supercell"""
        states = self.states if h is self.h else MeshStates(h,nk=self.nk)
        s = [np.sum(e[e<0.]) + np.sum(ef) for e,ef in
            zip(states.eigenvalues(),states.frozen_eigenvalues())]
        if states.has_eh: s = [(x + t)/2. for x,t in zip(s,states.trhe)]
        return np.mean(s)
    def electrons(self,h):
        """The electron sector of a Nambu h, a copy without the Nambu
        degree of freedom (sparse when h is), and its folded basis, the
        electron one u.get_mean_field_hamiltonian() keeps as basis.normal"""
        b = getattr(get_basis(h),"normal",None)
        if b is None:
            raise ValueError("the Fermi level of a Nambu ultracell needs the "
                    "folded basis of its electrons, which "
                    "u.get_mean_field_hamiltonian() builds")
        hn = h.copy() ; hn.remove_nambu()
        return hn,b
    def normal_fermi(self,h,filling):
        """The Fermi level of the normal state of a Nambu h at filling, the
        search(h,filling) of superscf.BdGFermiLevel, as
        spectrum.get_fermi4filling of its electrons for "ed". The electrons
        are solved in their own folded basis. With a cut it is that of the
        lattice-periodic part of the electrons, with the window centered at
        the Fermi level of that part, found from all its levels, and the
        half-width of the BdG cut. Only the eigenvalues of the ultracell are
        computed, and the states this backend keeps are left alone"""
        hn,b = self.electrons(h)
        b = b.rebuilt(hn) # with a cut, the states of the periodic part
        ks = np.array(hn.geometry.get_kmesh(nk=self.nk))
        if b.hopp is not None: # the window around its Fermi level
            from ..spectrum import get_fermi_energy_T
            b.center = get_fermi_energy_T(mesh_levels(b,b.hopp,ks),filling,
                    T=self.delta)
        hkgen = sparse_hk_gen(hn)
        esk,nfrozen = [],0
        for k in ks:
            E,U,valid,Ef,Uf,validf = b.states(k,frozen=True)
            H = ultracell_matrix(b,hkgen(k),k,E,U,valid)
            esk.append(eigh(H,eigvals_only=True,overwrite_a=True))
            nfrozen += int(np.sum(validf))
        return fermi_level(esk,hn.intra.shape[0],filling,T=self.delta,
                nfrozen=nfrozen)
    def normal_compressibility(self,h,filling,df=0.05):
        """dN/dmu of the normal state of a Nambu h around filling, the
        compressibility of superscf.BdGFermiLevel, with the finite
        difference of BdGFermiLevel.compressibility taken on all the
        levels of the lattice-periodic part of the electrons on the folded
        mesh: the window of normal_fermi may not hold filling +- df"""
        from ..spectrum import get_fermi_energy_T
        df = min(df,filling/2.,(1.-filling)/2.)
        if df<=0.: return 1.0 # empty or full, nothing to hold
        hn,b = self.electrons(h)
        hp = HoppingSum(b.periodic_hoppings(hn),b.norb,dim=b.dimensionality)
        es = mesh_levels(b,hp,np.array(hn.geometry.get_kmesh(nk=self.nk)))
        dmu = get_fermi_energy_T(es,filling+df,T=self.delta) \
                - get_fermi_energy_T(es,filling-df,T=self.delta)
        dmu = max(dmu,1e-6) # a degenerate level: a small, safe step
        return 2.*df*hn.intra.shape[0]/dmu


def require_kept_above(esk,n,delta,emax,what,nfrozen=0):
    """With a cut, the Fermi level must lie below the highest kept state,
    as fermi_level requires of a filling: the states above emax are not
    in the basis and count as empty, and when every kept state is full
    the states cut away can lie below the Fermi level too. esk are the
    ultracell eigenvalues per kpoint, measured from the Fermi level, of a
    supercell with n states per kpoint, occupied with the smearing delta;
    a state is full empty_ratio smearings below the Fermi level. The
    count is over the whole mesh, as in fermi_level, since a kpoint can
    keep no state at all; nfrozen is the number of frozen states on the
    mesh, which are in the basis but not in esk"""
    kept = sum(len(e) for e in esk)
    if emax is None or kept+nfrozen==n*len(esk): return # nothing cut above
    tops = [np.max(e) for e in esk if len(e)>0]
    if len(tops)>0 and max(tops)>-empty_ratio*delta: return
    where = "" if len(tops)==0 else (" (the highest is "
            +str(-max(tops))+" below it)")
    raise ValueError(what+" is above every state the ultracell keeps on "
            "this mesh"+where+", so the states that emax="+str(emax)
            +" cuts away can lie below it and be occupied; build the "
            "ultracell with a larger emax")


def require_frozen_below(efk,delta,emin,what):
    """The frozen states below emin are full, so the Fermi level must lie
    above every one of them by empty_ratio smearings, where the occupation
    of a state differs from one by less than exp(-50). efk are the
    energies of the frozen states per kpoint (frozen_levels), measured from
    the Fermi level"""
    tops = [np.max(e) for e in efk if len(e)>0]
    if len(tops)==0 or max(tops)<-empty_ratio*delta: return
    raise ValueError(what+" is not above every frozen state of the "
            "ultracell (the highest is "+str(max(tops))+" from it), and the "
            "states below emin="+str(emin)+" count as full; build the "
            "ultracell with a lower emin")


def fermi_level(esk,n,filling,T=0.,nfrozen=0):
    """Fermi level that holds filling, a fraction of all the n states of
    the supercell per kpoint, from the ultracell eigenvalues esk, one array
    per kpoint, and the number nfrozen of frozen states on the whole mesh.
    The states above emax are not in the basis and count as empty, so the
    filling must leave the Fermi level below the highest kept state; the
    frozen states below emin count as full, so the filling must need more
    states than them. Both are padded, the frozen ones below every kept
    state and the dropped ones above, for the search"""
    from ..spectrum import get_fermi_energy_T
    kept = sum(len(e) for e in esk)
    total = n*len(esk)
    ntarget = int(round(total*filling))
    if ntarget>=kept+nfrozen and kept+nfrozen<total: # states above needed
        raise ValueError("filling="+str(filling)+" needs "+str(ntarget)
                +" occupied states, and the ultracell keeps "+str(kept)
                +" and freezes "+str(nfrozen)+" below them on this mesh; "
                "build it with a larger emax")
    if nfrozen>0 and ntarget<=nfrozen: # the frozen states hold too much
        raise ValueError("filling="+str(filling)+" needs "+str(ntarget)
                +" occupied states, and the ultracell freezes "+str(nfrozen)
                +" full states below emin on this mesh; build it with a "
                "lower emin")
    # beyond every kept state; a kpoint where the cut keeps nothing has none
    kept_es = [e for e in esk if len(e)>0]
    top = max(np.max(e) for e in kept_es) + 100.
    bottom = min(np.min(e) for e in kept_es) - 100.
    es = np.concatenate(esk + [np.full(nfrozen,bottom),
        np.full(total-kept-nfrozen,top)])
    return get_fermi_energy_T(es,filling,T=T)
