import numpy as np
from scipy.sparse import coo_matrix
from copy import deepcopy

# TODO
# - autoannealing, stopping the iterations once a reasonable GS is reached



class LatticeGas():
    def __init__(self,g,filling=0.5): # geometry
        g.nrep = 1
        self.geometry = g # store geometry
        self.nsites = len(g.r) # number of sites
        self.mu = np.zeros(len(g.r)) # chemical potential
        self.den = np.zeros(len(g.r)) # chemical potential
        self.j = np.array([0]) # interactions
        self.pairs = np.array([[0,0]]) # empty list
        self._adjacency = None # lazily-built cache, see _get_adjacency
        self.set_filling(filling)
    def set_filling(self,filling):
        """Set filling of the system"""
        self.den[:] = 0. # initialize all to zero
        N = int(np.round(self.nsites*filling)) # filled sites
        self.den = random_density(len(self.den),N) # random density
    def add_interaction(self,Jij=None,**kwargs):
        h = self.geometry.get_hamiltonian(has_spin=False,tij=Jij)
        m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
        pairs = np.array([m.row,m.col]).transpose() # convert to array
        self.pairs = np.concatenate([self.pairs,pairs]) # add interaction
        self.j = np.concatenate([self.j,m.data]).real # store
        self._adjacency = None # pairs/j changed, invalidate cache
    def _get_adjacency(self):
        """CSR adjacency of self.pairs/self.j, cached since it only
        depends on the (rarely-changing) interaction terms, not on the
        occupation snapshot den"""
        if self._adjacency is None:
            self._adjacency = _build_adjacency(self.nsites,self.pairs,self.j)
        return self._adjacency
    def get_energy(self):
        return energy_numba(self.mu,self.pairs,self.j,self.den)
    def get_local_energy(self,**kwargs):
        return get_local_energy(self,**kwargs)
    def get_local_mu(self,**kwargs):
        return get_local_mu(self,**kwargs)
    def optimize_energy(self,**kwargs):
        """Optimize the energy"""
        x,es = optimize_discrete(self.mu,self.pairs,self.j,self.den,
                adjacency=self._get_adjacency(),**kwargs)
        self.den = x # overwrite
        return es
    def get_correlator(self,**kwargs):
        """Return the nearest neighbor correlators"""
        from .statphystk.correlator import get_nnc
        return get_nnc(self.geometry,self.den,**kwargs)
    def copy(self):
        return deepcopy(self)

from numba import jit

def energy_numba(mu,pairs,js,den):
    if len(pairs)==0:
        pairs = np.array([[0,0]])
        js = np.array([0.])
    return energy_numba_jit(mu,pairs,js,den)


@jit(nopython=True)
def energy_numba_jit(mu,pairs,js,den):
    """Compute the energy of the lattice gas model"""
    nump = len(pairs) # number of pairs
    n = len(mu) # number of sites
    etot = 0. # output energy
    for i in range(n): etot = etot + mu[i]*den[i] # chemical potential
    for ip in range(nump):
        ii = pairs[ip][0]
        jj = pairs[ip][1]
        etot = etot + den[ii]*den[jj]*js[ip] # add contribution
    return etot



def _build_adjacency(n,pairs,js):
    """Build a CSR-style adjacency (rows keyed by pairs[:,0]) so that the
    interaction terms touching a given site can be found in O(degree)
    instead of scanning the full O(n_pairs) pairs array. Relies on the
    same convention as the rest of the module: pairs comes from a
    symmetric coupling matrix, so it always lists both (i,k) and (k,i)
    for every edge, and a site's own row already has full connectivity"""
    order = np.argsort(pairs[:,0],kind="stable")
    idx = pairs[:,1][order].astype(np.int64) # neighbor of each row entry
    jarr = js[order].astype(np.float64)
    ptr = np.searchsorted(pairs[:,0][order],np.arange(n+1)).astype(np.int64)
    return ptr,idx,jarr


@jit(nopython=True)
def _row_sum_excluding(ptr,idx,jarr,den,i,exclude):
    """sum_k J_{i,k}*den[k] over site i's neighbors, skipping `exclude`"""
    s = 0.
    for k in range(ptr[i],ptr[i+1]):
        neighbor = idx[k]
        if neighbor!=exclude:
            s += jarr[k]*den[neighbor]
    return s


@jit(nopython=True)
def swap_delta_energy(mu,ptr,idx,jarr,den,i1,i2):
    """Energy change from swapping den[i1] and den[i2]. Equivalent to
    energy_numba_jit(mu,pairs,js,den_after) - energy_numba_jit(mu,pairs,js,den_before)
    but costs O(degree(i1)+degree(i2)) instead of O(n_pairs): the direct
    i1-i2 edge (if any) cancels exactly under a swap since its
    contribution only depends on the product den[i1]*den[i2], which a
    swap leaves unchanged, so it is excluded from both row sums below"""
    a = den[i1] ; b = den[i2]
    if a==b: return 0. # nothing changes
    sum1 = _row_sum_excluding(ptr,idx,jarr,den,i1,i2)
    sum2 = _row_sum_excluding(ptr,idx,jarr,den,i2,i1)
    return (mu[i1]-mu[i2])*(b-a) + 2.*(b-a)*(sum1-sum2)


def optimize_discrete(mu,pairs,js,x0,temp=0.1,ntries=1e5,info=False,
        resync_every=1000,adjacency=None):
    """Discrete optimization, using a swap method. Energy changes are
    tracked incrementally with swap_delta_energy (O(degree) per swap)
    instead of recomputing the full O(n_pairs) energy on every try;
    the running energy is periodically resynced with a full recompute
    (every `resync_every` iterations) to bound floating-point drift.
    `adjacency`, the (ptr,idx,jarr) CSR tuple from _build_adjacency,
    can be passed in precomputed (e.g. cached across repeated calls
    with the same pairs/js, as LatticeGas.optimize_energy does) to
    skip rebuilding it; it only depends on pairs/js, not on x0"""
    n = len(x0) # number of sites
    inds = np.arange(n) # indexes
    vals = np.unique(x0) # different values
    if len(vals)!=2:
        raise ValueError(f"optimize_discrete needs exactly 2 distinct "
            f"values in the density (got {vals}); check the filling "
            f"is not 0 or 1") # not implemented
    ptr,idx,jarr = adjacency if adjacency is not None else _build_adjacency(n,pairs,js)
    def pick_swap_pair(x):
        ainds = inds[x==vals[0]] # indexes for first value
        binds = inds[x==vals[1]] # indexes for second value
        j1 = np.random.randint(0,len(ainds)) # one random site
        j2 = np.random.randint(0,len(binds)) # one random site
        return ainds[j1],binds[j2]
    nrep = int(ntries) # this many iterations
    x = x0.copy()
    e = energy_numba(mu,pairs,js,x) # current energy, tracked incrementally
    es = np.zeros(nrep) # storage for energies
    for ii in range(nrep): # this many iterations
        if resync_every and ii%resync_every==0: # bound fp drift
            e = energy_numba(mu,pairs,js,x)
        xcand = x.copy()
        delta = 0.
        for _ in range(np.random.randint(1,4)): # make 1,2,3 swaps
            i1,i2 = pick_swap_pair(xcand)
            delta += swap_delta_energy(mu,ptr,idx,jarr,xcand,i1,i2)
            xcand[i1],xcand[i2] = xcand[i2],xcand[i1] # swap
        en = e + delta # new
        if info: print(en)
        if en<=e: # smaller or same
            es[ii] = en # store new energy
            x = xcand ; e = en # overwrite
        else:
            fac = np.exp((e-en)/temp) # acceptance probability
            if np.random.random()<fac:
                es[ii] = en # store new energy
                x = xcand ; e = en # overwrite
            else:
                es[ii] = e # keep old energy
                pass # do nothing
    return x,es # return final state



def _local_terms(LG,ii):
    """Return the (pairs,J,mu) restricted to the interaction terms
    that touch site ii, without copying the whole LatticeGas object"""
    mask = (LG.pairs[:,0]==ii) | (LG.pairs[:,1]==ii) # terms touching ii
    pairs0 = LG.pairs[mask]
    j0 = LG.j[mask]/2.
    mu0 = np.zeros_like(LG.mu)
    mu0[ii] = LG.mu[ii]
    return pairs0,j0,mu0


def get_local_energy(LG,normalize=False):
    """Return the local energy at each site for the current snapshot"""
    def get(ii): # get for site ii
        pairs0,j0,mu0 = _local_terms(LG,ii)
        enii = energy_numba(mu0,pairs0,j0,LG.den)
        if normalize: return enii/np.sum(j0)
        else: return enii
    return np.array([get(ii) for ii in range(len(LG.geometry.r))]) # loop over positions



def get_local_mu(LG,normalize=False):
    """Return the local chemical potential"""
    def get(ii): # get for site ii
        pairs0,j0,mu0 = _local_terms(LG,ii)
        den0 = LG.den.copy()
        den0[ii] = 1.0 # overwrite to return chemical potential
        enii = energy_numba(mu0,pairs0,j0,den0)
        if normalize: return enii/np.sum(j0)
        else: return enii
    return np.array([get(ii) for ii in range(len(LG.geometry.r))]) # loop over positions



def random_density(Ntot,N):
    """Generate an array with N 1's and Ntot-N 0's,
    with the 1's randomly distributed"""
    out = np.zeros(Ntot) # initialize
    inds = np.random.choice(Ntot,N,replace=False) # N random indexes
    out[inds] = 1.0
    return out



