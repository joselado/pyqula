"""The folded Bloch basis of the ultracell method.

The ultracell is a supercell of n1 x n2 x n3 unit cells. At a Bloch momentum
k_S of the ultracell (reduced coordinates of the ultracell), the clean
supercell is diagonalized by the Bloch states of the unit cell at the N
folded momenta

    p_q = (k_S + q)/n   (componentwise),   q in Z_n1 x Z_n2 x Z_n3,

written on the orbitals of the supercell as

    B_{q,m}(R,alpha) = N^(-1/2) exp(2 pi i p_q.R) u_m(p_q)[alpha],

with u_m(p) the eigenvectors of the unit cell's h.get_hk_gen() at p, in
pyqula's lattice gauge. The method keeps, at each folded momentum, the
states with energy up to emax, and solves the modulated supercell in that
basis (Rayleigh-Ritz). With every state kept the basis spans the whole
supercell and the result is exact.

This module holds the bookkeeping: which cell R and which unit-cell orbital
alpha every orbital of the supercell is, the unit-cell states at the folded
momenta, and the two transforms between a vector on the supercell orbitals
and its components per folded momentum, done with an FFT over the cells.
The method is that of T. Mueller, S. Sharma, E. K. U. Gross and
J. K. Dewhurst, Phys. Rev. Lett. 125, 256402 (2020), arXiv:2008.12573, in
its direct form, with the unit-cell states at every folded momentum.
"""

import numpy as np
from copy import copy

from .. import algebra


def dense(m):
    """A dense ndarray from a dense or sparse matrix"""
    return np.asarray(algebra.todense(m))


def pad3(k):
    """A kpoint padded to three components"""
    kv = np.zeros(3)
    ki = np.asarray(k,dtype=float).ravel()
    kv[0:len(ki)] = ki[0:3]
    return kv


def diagonal_shape(M):
    """The three integers of a diagonal supercell matrix, padded to 3x3"""
    M = np.array(M,dtype=int)
    M3 = np.identity(3,dtype=int)
    M3[0:M.shape[0],0:M.shape[1]] = M
    if np.any(M3!=np.diag(np.diag(M3))):
        raise NotImplementedError("the ultracell is only implemented for a "
                "diagonal supercell, n1 x n2 x n3 copies of the unit cell "
                "along its own lattice vectors; this supercell has the "
                "matrix "+str(M3.tolist()))
    n = np.diag(M3).astype(int)
    if np.any(n<1):
        raise ValueError("the supercell matrix has a non-positive diagonal "
                +str(n.tolist()))
    return n


def cut_window(e,emax,tol=1e-7):
    """Indexes of the sorted levels e kept by the cut e <= emax, completed
    so that a degenerate level is kept whole: a level split by the cut
    would leave an arbitrary part of its subspace in the basis"""
    if emax is None: return np.arange(len(e))
    return np.arange(complete_level(e,int(np.sum(e<=emax)),tol=tol))


def complete_level(e,nkeep,tol=1e-7):
    """The number of sorted levels e to keep, nkeep raised so that the last
    kept level is not split from the next one by the cut"""
    while 0<nkeep<len(e) and abs(e[nkeep]-e[nkeep-1])<tol*max(1.,abs(e[nkeep])):
        nkeep += 1 # the next level is degenerate with the last one kept
    return nkeep


class HoppingSum():
    """The Bloch Hamiltonian sum_d t_d exp(2 pi i p.d) of a set of hoppings
    {d: (norb,norb)} of a unit cell, evaluated at many momenta at once.
    Only the first dim components of d and p enter, as in
    geometrytk.bloch.bloch_phase"""
    def __init__(self,hop,norb,dim=3):
        keys = list(hop)
        self.ds = np.array([pad3(d) for d in keys]).reshape(len(keys),3)
        self.ds[:,dim:] = 0.
        self.ms = np.array([dense(hop[d]) for d in keys],
                dtype=np.complex128).reshape(len(keys),norb,norb)
    def __call__(self,ps):
        """The Hermitian matrices at the momenta ps, (np,norb,norb)"""
        ph = np.exp(2j*np.pi*(np.asarray(ps,dtype=float)@self.ds.T))
        H = np.einsum("pd,dab->pab",ph,self.ms)
        return 0.5*(H + np.conjugate(np.swapaxes(H,1,2)))


def eigh_momenta(hsum,ps,chunk=2**22):
    """Eigenvalues (np,norb) and eigenvectors (np,norb,norb) of hsum at the
    momenta ps, diagonalized in batches of at most chunk matrix entries"""
    ps = np.asarray(ps)
    norb = hsum.ms.shape[1]
    nb = max(1,chunk//(norb*norb)) # momenta per batch
    es,ws = [],[]
    for i in range(0,len(ps),nb):
        e,w = np.linalg.eigh(hsum(ps[i:i+nb]))
        es.append(e) ; ws.append(w)
    return np.concatenate(es),np.concatenate(ws)


class FoldedBasis():
    """Bookkeeping of the folded Bloch basis of a supercell hs of h"""
    def __init__(self,h,hs,emax=None):
        from ..unfolding import get_supercell_map
        if h.has_spin!=hs.has_spin or h.has_eh!=hs.has_eh:
            raise ValueError("the supercell and the unit cell must have the "
                    "same Hilbert space, got has_spin="+str(hs.has_spin)
                    +" and has_eh="+str(hs.has_eh)+" for the supercell, "
                    "has_spin="+str(h.has_spin)+" and has_eh="+str(h.has_eh)
                    +" for the unit cell")
        if h.dimensionality!=hs.dimensionality:
            raise ValueError("the supercell is "+str(hs.dimensionality)
                    +"-dimensional and the unit cell "+str(h.dimensionality)
                    +"-dimensional; they must be the same")
        if h.dimensionality==0:
            raise ValueError("the ultracell needs a periodic unit cell, got "
                    "a zero-dimensional Hamiltonian")
        M,replicas,primal = get_supercell_map(hs.geometry,h.geometry)
        self.n = diagonal_shape(M) # cells along each lattice vector
        self.N = int(np.prod(self.n)) # number of cells
        self.dimensionality = h.dimensionality
        self.norb = h.intra.shape[0] # orbitals of the unit cell
        nat0 = len(h.geometry.r) # sites of the unit cell
        if self.norb%nat0!=0:
            raise ValueError("the unit cell has "+str(self.norb)+" orbitals "
                    "on "+str(nat0)+" sites, not a whole number per site")
        f = self.norb//nat0 # orbitals per site (spin, electron-hole)
        nats = len(hs.geometry.r) # sites of the supercell
        norbs = hs.intra.shape[0] # orbitals of the supercell
        if norbs!=self.N*self.norb or nats!=self.N*nat0:
            raise ValueError("the supercell must hold every orbital of every "
                    "one of its "+str(self.N)+" cells, "
                    +str(self.N*self.norb)+" orbitals, and it has "
                    +str(norbs)+"; model a missing site with a large onsite "
                    "energy instead of removing it")
        replicas = np.array(replicas,dtype=int).reshape(nats,-1)
        rep3 = np.zeros((nats,3),dtype=int)
        rep3[:,0:replicas.shape[1]] = replicas[:,0:3]
        primal = np.array(primal,dtype=int)
        # per orbital of the supercell: its cell and its unit-cell orbital
        R = np.repeat(rep3,f,axis=0) # actual cell vector of each orbital
        self.alpha = np.repeat(primal*f,f) + np.tile(np.arange(f),nats)
        cell = np.mod(R,self.n[None,:]) # cell inside the ultracell
        self.cell = np.ravel_multi_index(cell.T,tuple(self.n))
        key = self.cell*self.norb + self.alpha
        if len(np.unique(key))!=norbs:
            raise ValueError("the supercell does not hold each orbital of "
                    "each cell exactly once, so it is not a supercell of "
                    "this unit cell")
        self.R = R.astype(float)
        self.site_cell = np.ravel_multi_index(np.mod(rep3,self.n[None,:]).T,
                tuple(self.n)) # cell of each site of the supercell
        self.site_primal = primal # unit-cell site of each site
        self.nsites = nat0
        self.q = np.stack(np.meshgrid(*[np.arange(m) for m in self.n],
            indexing="ij"),axis=-1).reshape(-1,3) # folded momenta, C order
        self.hop0 = HoppingSum(h.get_multihopping().get_dict(),self.norb,
                dim=self.dimensionality) # Bloch Hamiltonian of the unit cell
        if emax is not None:
            try: emax = float(emax)
            except (TypeError,ValueError):
                raise TypeError("emax must be a number or None, got "
                        +repr(emax))
        self.emax = emax
        self.rebuild = False # the clean states, see states()
        self.hopp = None # the lattice-periodic part, see rebuilt()
        self._counts = {} # states kept by the clean cut, per kS
    def folded_momenta(self,kS):
        """The N momenta p_q = (k_S+q)/n, in reduced coordinates of the
        unit cell, one row per q"""
        return (pad3(kS)[None,:] + self.q)/self.n[None,:]
    def rebuilt(self,hs):
        """A copy of this basis whose states are those of the
        lattice-periodic part of the supercell hs, its hoppings averaged
        over the cells. The block of each folded momentum on itself, in
        the orbital Bloch basis, is the Bloch Hamiltonian of that part at
        p_q, the same for every kS, so it is built once per hs. Without a
        cut both bases span every orbital, and the copy keeps the clean
        states"""
        b = copy(self)
        if self.emax is not None:
            b.hopp = HoppingSum(self.periodic_hoppings(hs),self.norb,
                    dim=self.dimensionality)
        return b
    def periodic_hoppings(self,hs):
        """The hoppings of the supercell hs averaged over its N cells, as
        the hoppings of a unit cell, {d: (norb,norb)}. An element of hs
        between orbital i of cell R_i and orbital j of cell R_j + n D of
        the supercell D is a hopping of the unit cell in the direction
        d = R_j + n D - R_i, between their unit-cell orbitals, divided by
        N. It is the lattice-periodic part of hs: a uniform moment, a
        uniform shift, the average of a modulated hopping"""
        from scipy.sparse import coo_matrix
        out = {}
        for D,m in hs.get_multihopping().get_dict().items():
            m = coo_matrix(m)
            if m.nnz==0: continue
            i,j,val = m.row,m.col,m.data
            d = self.R[j] - self.R[i] + self.n[None,:]*pad3(D)[None,:]
            d = np.round(d).astype(int)
            keys,inv = np.unique(d,axis=0,return_inverse=True)
            inv = np.ravel(inv)
            for ik,key in enumerate(keys):
                s = inv==ik
                t = coo_matrix((val[s]/self.N,(self.alpha[i[s]],
                    self.alpha[j[s]])),shape=(self.norb,self.norb))
                key = tuple(key)
                out[key] = out.get(key,0.) + t.toarray()
        return out
    def states(self,kS):
        """The states kept at every folded momentum, padded to the largest
        number kept: energies (N,bmax), vectors (N,norb,bmax), and the mask
        (N,bmax) of the ones that exist. They are the states of the clean
        unit cell, or, for a basis made by rebuilt() with a cut, those of
        the lattice-periodic part of the supercell"""
        if self.hopp is not None: return self.rebuilt_states(kS)
        return self.clean_states(kS)
    def counts(self,kS):
        """Number of states the clean cut keeps at every folded momentum"""
        key = tuple(np.round(pad3(kS),12))
        if key not in self._counts:
            self._counts[key] = np.sum(self.clean_states(kS)[2],axis=1)
        return self._counts[key]
    def rebuilt_states(self,kS):
        """The states of the lattice-periodic part, as many per momentum as
        the cut keeps in the clean unit cell, completed so that a
        degenerate level is kept whole. A lattice-periodic part of the
        modulation or of a mean field (a uniform moment, a uniform shift)
        is then in the basis, and does not have to be built from the states
        the cut leaves out"""
        e,w = eigh_momenta(self.hopp,self.folded_momenta(kS))
        counts = self.counts(kS)
        cs = [complete_level(e[iq],int(counts[iq])) for iq in range(self.N)]
        return self.pad([e[iq,0:c] for iq,c in enumerate(cs)],
                [w[iq,:,0:c] for iq,c in enumerate(cs)])
    def pad(self,es,ws):
        """Pad the states of every folded momentum to the largest number"""
        bmax = max([len(e) for e in es])
        if bmax==0:
            raise ValueError("no state of the unit cell is below emax="
                    +str(self.emax)+", so the basis is empty")
        E = np.zeros((self.N,bmax))
        U = np.zeros((self.N,self.norb,bmax),dtype=np.complex128)
        valid = np.zeros((self.N,bmax),dtype=bool)
        for i in range(self.N):
            b = len(es[i])
            E[i,0:b] = es[i] ; U[i,:,0:b] = ws[i] ; valid[i,0:b] = True
        return E,U,valid
    def clean_states(self,kS):
        """The states of the clean unit cell up to emax at every folded
        momentum, padded as in states()"""
        e,w = eigh_momenta(self.hop0,self.folded_momenta(kS))
        idx = [cut_window(e[iq],self.emax) for iq in range(self.N)]
        return self.pad([e[iq,i] for iq,i in enumerate(idx)],
                [w[iq][:,i] for iq,i in enumerate(idx)])
    def phase(self,kS):
        """exp(2 pi i (k_S/n).R) on every orbital of the supercell"""
        return np.exp(2j*np.pi*(self.R@(pad3(kS)/self.n)))
    def to_supercell(self,kS,phi):
        """From the components phi (N,norb,...) per folded momentum to the
        vector on the supercell orbitals,
        Phi(R,alpha) = N^(-1/2) sum_q exp(2 pi i p_q.R) phi_q(alpha)"""
        rest = phi.shape[2:]
        a = phi.reshape(tuple(self.n)+(self.norb,)+rest)
        a = np.fft.ifftn(a,axes=(0,1,2))*self.N # sum_q exp(+2 pi i q.R/n)
        a = a.reshape((self.N,self.norb)+rest)
        out = a[self.cell,self.alpha]/np.sqrt(self.N)
        ph = self.phase(kS).reshape((-1,)+(1,)*len(rest))
        return ph*out
    def from_supercell(self,kS,Y):
        """From a vector Y (norbs,...) on the supercell orbitals to its
        components per folded momentum,
        Y_q(alpha) = N^(-1/2) sum_R exp(-2 pi i p_q.R) Y(R,alpha)"""
        rest = Y.shape[1:]
        ph = np.conjugate(self.phase(kS)).reshape((-1,)+(1,)*len(rest))
        Z = np.zeros((self.N,self.norb)+rest,dtype=np.complex128)
        Z[self.cell,self.alpha] = ph*Y
        Z = Z.reshape(tuple(self.n)+(self.norb,)+rest)
        Z = np.fft.fftn(Z,axes=(0,1,2)) # sum_R exp(-2 pi i q.R/n)
        return Z.reshape((self.N,self.norb)+rest)/np.sqrt(self.N)
    def basis_block(self,kS,U,iq):
        """The basis vectors of the folded momentum iq on the supercell
        orbitals, (norbs,bmax)"""
        p = self.folded_momenta(kS)[iq]
        ph = np.exp(2j*np.pi*(self.R@p))/np.sqrt(self.N)
        return ph[:,None]*U[iq][self.alpha,:]
