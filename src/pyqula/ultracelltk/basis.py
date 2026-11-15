"""The folded Bloch basis of the ultracell method.

The ultracell is a supercell of N unit cells, with lattice vectors
A_S = M A (rows) for an integer matrix M, n1 x n2 x n3 cells along the
lattice vectors when M is diagonal. At a Bloch momentum k_S of the
ultracell (reduced coordinates of the ultracell), the clean supercell is
diagonalized by the Bloch states of the unit cell at the N folded momenta

    p_q = M^-1 (k_S + q),   q in Z^d / M Z^d,

which is (k_S + q)/n componentwise for a diagonal M. For any M a diagonal
form P M Q = diag(n), with P and Q unimodular, makes the q a grid in the
coordinates P q and the cells R a grid in the coordinates R Q, since
R.M^-1 q = (R Q) diag(1/n) (P q), so both transforms below are FFTs,

written on the orbitals of the supercell as

    B_{q,m}(R,alpha) = N^(-1/2) exp(2 pi i p_q.R) u_m(p_q)[alpha],

with u_m(p) the eigenvectors of the unit cell's h.get_hk_gen() at p, in
pyqula's lattice gauge. The method keeps, at each folded momentum, the
states inside a window (emin,emax), and solves the modulated supercell in
that basis (Rayleigh-Ritz). The states below emin are frozen, occupied and
not solved, and those above emax are dropped. With every state kept the
basis spans the whole supercell and the result is exact.

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


def diagonal_form(M):
    """Unimodular integer matrices P and Q and the positive integers n with
    P M Q = diag(n), for a square nonsingular integer matrix M (a diagonal
    form, not necessarily the Smith form with n_i dividing n_{i+1}, which
    the FFT does not need). The folded momenta of the supercell are then a
    rectangular grid in the coordinates P q, and its cells in R Q"""
    A = [[int(x) for x in row] for row in np.array(M,dtype=int)]
    d = len(A)
    P = [[int(i==j) for j in range(d)] for i in range(d)]
    Q = [[int(i==j) for j in range(d)] for i in range(d)]
    for t in range(d):
        while True:
            nz = [(abs(A[i][j]),i,j) for i in range(t,d) for j in range(t,d)
                    if A[i][j]!=0]
            if len(nz)==0:
                raise ValueError("the supercell matrix "+str(np.array(M)
                    .tolist())+" is singular")
            _,i,j = min(nz) # the smallest entry as the pivot
            A[t],A[i] = A[i],A[t] ; P[t],P[i] = P[i],P[t] # swap rows
            for row in A: row[t],row[j] = row[j],row[t] # swap columns
            for row in Q: row[t],row[j] = row[j],row[t]
            clean = True
            for i in range(t+1,d): # reduce the column below the pivot
                f = A[i][t]//A[t][t]
                A[i] = [a-f*b for a,b in zip(A[i],A[t])]
                P[i] = [a-f*b for a,b in zip(P[i],P[t])]
                clean = clean and A[i][t]==0
            for j in range(t+1,d): # and the row to its right
                f = A[t][j]//A[t][t]
                for row in A: row[j] -= f*row[t]
                for row in Q: row[j] -= f*row[t]
                clean = clean and A[t][j]==0
            if clean: break # what is left is smaller than the pivot
        if A[t][t]<0: A[t] = [-a for a in A[t]] ; P[t] = [-a for a in P[t]]
    return (np.array(P,dtype=int),np.array(Q,dtype=int),
            np.array([A[t][t] for t in range(d)],dtype=int))


def supercell_shape(M,dim):
    """The supercell matrix padded to 3x3, M (A_supercell = M A, rows), and
    its diagonal form P M Q = diag(n) on the first dim directions, padded
    with the identity, as in diagonal_form"""
    M = np.array(M,dtype=int)
    M3 = np.identity(3,dtype=int)
    M3[0:M.shape[0],0:M.shape[1]] = M
    if np.any(M3[0:dim,dim:]!=0) or np.any(M3[dim:,0:dim]!=0) or \
            np.any(M3[dim:,dim:]!=np.identity(3-dim,dtype=int)):
        raise ValueError("the supercell matrix "+str(M3.tolist())+" mixes "
                "the first "+str(dim)+" lattice vectors of a "+str(dim)
                +"-dimensional lattice with the others")
    P,Q = np.identity(3,dtype=int),np.identity(3,dtype=int)
    n = np.ones(3,dtype=int)
    P[0:dim,0:dim],Q[0:dim,0:dim],n[0:dim] = diagonal_form(M3[0:dim,0:dim])
    return M3,P,Q,n


def window_range(e,emin=None,emax=None,tol=1e-7):
    """The range [i0,i1) of the sorted levels e inside the window
    (emin,emax): the i0 levels below emin are frozen, the levels from i1 on
    are dropped, and either side is open when it is None. Both edges are
    tolerant, by tol relative to the edge, and both are completed outward,
    so that a degenerate level split by an edge is kept whole: a level
    split by the cut would leave an arbitrary part of its subspace in the
    basis. For a spectrum with e(-p) = -e(p) reversed, the particle-hole
    symmetry of a BdG Hamiltonian, the window (-x,x) gives at -p the
    mirror of the range at p, i1(-p) = len(e) - i0(p)"""
    i0 = 0 if emin is None else int(np.sum(e<emin-tol*max(1.,abs(emin))))
    i1 = len(e) if emax is None else \
            int(np.sum(e<=emax+tol*max(1.,abs(emax))))
    return lower_level(e,i0,tol=tol),complete_level(e,i1,tol=tol)


def complete_level(e,nkeep,tol=1e-7):
    """The number of sorted levels e to keep, nkeep raised so that the last
    kept level is not split from the next one by the cut"""
    while 0<nkeep<len(e) and abs(e[nkeep]-e[nkeep-1])<tol*max(1.,abs(e[nkeep])):
        nkeep += 1 # the next level is degenerate with the last one kept
    return nkeep


def lower_level(e,nfrozen,tol=1e-7):
    """The number of sorted levels e to freeze, nfrozen lowered so that the
    first kept level is not split from the last frozen one; the mirror of
    complete_level, with the tolerance taken on the level that moves"""
    while 0<nfrozen<len(e) and \
            abs(e[nfrozen]-e[nfrozen-1])<tol*max(1.,abs(e[nfrozen-1])):
        nfrozen -= 1 # the last frozen level is degenerate with the next one
    return nfrozen


def energy_or_none(x,name):
    """x as a float, or None; anything else is a TypeError"""
    if x is None: return None
    try:
        if isinstance(x,str): raise TypeError # float("1") would take it
        return float(x)
    except (TypeError,ValueError):
        raise TypeError(name+" must be a number or None, got "+repr(x))


def parse_window(emax=None,window=None):
    """The window (emin,emax) of an ultracell, from emax= or from
    window=(emin,emax), with None for an open side. emax=x is the window
    (None,x)"""
    if emax is not None and window is not None:
        raise ValueError("give the cut as emax= or as window=(emin,emax), "
                "not both; got emax="+repr(emax)+" and window="+repr(window))
    emin = None
    if window is not None:
        try: emin,emax = window
        except (TypeError,ValueError):
            raise TypeError("window must be a pair (emin,emax) of numbers "
                    "or None, got "+repr(window))
    emin = energy_or_none(emin,"emin")
    emax = energy_or_none(emax,"emax")
    if emin is not None and emax is not None and emin>=emax:
        raise ValueError("the window (emin,emax) needs emin < emax, got "
                +repr((emin,emax)))
    return emin,emax


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
    """Bookkeeping of the folded Bloch basis of a supercell hs of h, with
    the window (emin,emax) of parse_window: at every folded momentum the
    sorted levels split into the frozen ones below emin, the kept ones and
    the dropped ones above emax"""
    def __init__(self,h,hs,emax=None,window=None):
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
        # the supercell matrix and its diagonal form P M Q = diag(n): the
        # cells are a grid of n1 x n2 x n3 in the coordinates R Q, and the
        # folded momenta a grid in P q, so both transforms are FFTs
        self.M,self.P,self.Q,self.n = supercell_shape(M,h.dimensionality)
        self.Minv = np.linalg.inv(self.M.astype(float))
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
        cell = np.mod(R@self.Q,self.n[None,:]) # cell inside the ultracell
        self.cell = np.ravel_multi_index(cell.T,tuple(self.n))
        key = self.cell*self.norb + self.alpha
        if len(np.unique(key))!=norbs:
            raise ValueError("the supercell does not hold each orbital of "
                    "each cell exactly once, so it is not a supercell of "
                    "this unit cell")
        self.R = R.astype(float)
        self.site_cell = np.ravel_multi_index(np.mod(rep3@self.Q,
            self.n[None,:]).T,tuple(self.n)) # cell of each site
        self.site_primal = primal # unit-cell site of each site
        self.nsites = nat0
        self.q = np.stack(np.meshgrid(*[np.arange(m) for m in self.n],
            indexing="ij"),axis=-1).reshape(-1,3) # P q of the folded
        # momenta, in C order
        self.hop0 = HoppingSum(h.get_multihopping().get_dict(),self.norb,
                dim=self.dimensionality) # Bloch Hamiltonian of the unit cell
        self.emin,self.emax = parse_window(emax,window)
        self.window = (self.emin,self.emax)
        self.rebuild = False # the clean states, see states()
        self.hopp = None # the lattice-periodic part, see rebuilt()
        self._ranges = {} # the ranges of the clean window, per kS
    @property
    def has_cut(self):
        """Whether the window leaves out states on either side"""
        return self.emin is not None or self.emax is not None
    def folded_momenta(self,kS):
        """The N momenta p_q = M^-1 (k_S+q) = Q diag(1/n) (P k_S + P q), in
        reduced coordinates of the unit cell, one row per q, which is
        (k_S+q)/n for a diagonal supercell"""
        t = ((self.P@pad3(kS))[None,:] + self.q)/self.n[None,:]
        return t@self.Q.T
    def rebuilt(self,hs):
        """A copy of this basis whose states are those of the
        lattice-periodic part of the supercell hs, its hoppings averaged
        over the cells. The block of each folded momentum on itself, in
        the orbital Bloch basis, is the Bloch Hamiltonian of that part at
        p_q, the same for every kS, so it is built once per hs. Without a
        cut both bases span every orbital, and the copy keeps the clean
        states"""
        b = copy(self)
        if self.has_cut:
            b.hopp = HoppingSum(self.periodic_hoppings(hs),self.norb,
                    dim=self.dimensionality)
        return b
    def periodic_hoppings(self,hs):
        """The hoppings of the supercell hs averaged over its N cells, as
        the hoppings of a unit cell, {d: (norb,norb)}. An element of hs
        between orbital i of cell R_i and orbital j of cell R_j + D M of
        the supercell D is a hopping of the unit cell in the direction
        d = R_j + D M - R_i, between their unit-cell orbitals, divided by
        N. It is the lattice-periodic part of hs: a uniform moment, a
        uniform shift, the average of a modulated hopping"""
        from scipy.sparse import coo_matrix
        out = {}
        for D,m in hs.get_multihopping().get_dict().items():
            m = coo_matrix(m)
            if m.nnz==0: continue
            i,j,val = m.row,m.col,m.data
            d = self.R[j] - self.R[i] + (pad3(D)@self.M)[None,:]
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
    def states(self,kS,frozen=False):
        """The states kept at every folded momentum, padded to the largest
        number kept: energies (N,bmax), vectors (N,norb,bmax), and the mask
        (N,bmax) of the ones that exist. They are the states of the clean
        unit cell, or, for a basis made by rebuilt() with a cut, those of
        the lattice-periodic part of the supercell. With frozen=True the
        frozen states below the window follow, padded the same way, as
        three more arrays (Ef,Uf,validf); they come from the same
        diagonalization as the kept ones, so that the two sets are
        orthogonal and the density matrix of both is a projector"""
        if self.hopp is not None: # the lattice-periodic part, see rebuilt()
            e,w = eigh_momenta(self.hopp,self.folded_momenta(kS))
            ranges = self.rebuilt_ranges(kS,e)
        else:
            e,w = eigh_momenta(self.hop0,self.folded_momenta(kS))
            ranges = [window_range(e[iq],self.emin,self.emax)
                    for iq in range(self.N)]
        out = self.pad([e[iq,i0:i1] for iq,(i0,i1) in enumerate(ranges)],
                [w[iq,:,i0:i1] for iq,(i0,i1) in enumerate(ranges)])
        if not frozen: return out
        return out + self.pad([e[iq,0:i0] for iq,(i0,i1) in enumerate(ranges)],
                [w[iq,:,0:i0] for iq,(i0,i1) in enumerate(ranges)])
    def clean_ranges(self,kS):
        """The ranges [i0,i1) of the window on the levels of the clean unit
        cell at every folded momentum of kS, computed once per kS"""
        key = tuple(np.round(pad3(kS),12))
        if key not in self._ranges:
            e = eigh_momenta(self.hop0,self.folded_momenta(kS))[0]
            self._ranges[key] = [window_range(e[iq],self.emin,self.emax)
                    for iq in range(self.N)]
        return self._ranges[key]
    def rebuilt_ranges(self,kS,e):
        """The ranges of the levels e (N,norb) of the lattice-periodic part
        at the folded momenta of kS: as many frozen and as many kept states
        per momentum as the window holds in the clean unit cell, both edges
        completed outward so that a degenerate level is kept whole. A
        lattice-periodic part of the modulation or of a mean field (a
        uniform moment, a uniform shift) is then in the basis, and does
        not have to be built from the states the cut leaves out"""
        return [(lower_level(e[iq],i0),complete_level(e[iq],i1))
                for iq,(i0,i1) in enumerate(self.clean_ranges(kS))]
    def pad(self,es,ws):
        """Pad the states of every folded momentum to the largest number.
        When the cut keeps no state at any folded momentum of this kS the
        ultracell has no state there, and the arrays have zero width"""
        bmax = max([len(e) for e in es])
        E = np.zeros((self.N,bmax))
        U = np.zeros((self.N,self.norb,bmax),dtype=np.complex128)
        valid = np.zeros((self.N,bmax),dtype=bool)
        for i in range(self.N):
            b = len(es[i])
            E[i,0:b] = es[i] ; U[i,:,0:b] = ws[i] ; valid[i,0:b] = True
        return E,U,valid
    def phase(self,kS):
        """exp(2 pi i (M^-1 k_S).R) on every orbital of the supercell"""
        return np.exp(2j*np.pi*(self.R@(self.Minv@pad3(kS))))
    def to_supercell(self,kS,phi):
        """From the components phi (N,norb,...) per folded momentum to the
        vector on the supercell orbitals,
        Phi(R,alpha) = N^(-1/2) sum_q exp(2 pi i p_q.R) phi_q(alpha)"""
        rest = phi.shape[2:]
        a = phi.reshape(tuple(self.n)+(self.norb,)+rest)
        a = np.fft.ifftn(a,axes=(0,1,2))*self.N # sum_q exp(+2 pi i R.M^-1 q)
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
        Z = np.fft.fftn(Z,axes=(0,1,2)) # sum_R exp(-2 pi i R.M^-1 q)
        Z /= np.sqrt(self.N) # in place, no other array of this size
        return Z.reshape((self.N,self.norb)+rest)
    def basis_block(self,kS,U,iq,ps=None):
        """The basis vectors of the folded momentum iq on the supercell
        orbitals, (norbs,bmax). ps are the folded momenta of kS, which a
        caller that builds many blocks computes once"""
        if ps is None: ps = self.folded_momenta(kS)
        p = ps[iq]
        ph = np.exp(2j*np.pi*(self.R@p))/np.sqrt(self.N)
        return ph[:,None]*U[iq][self.alpha,:]
