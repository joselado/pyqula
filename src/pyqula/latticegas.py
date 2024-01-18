import numpy as np
from scipy.sparse import coo_matrix
from copy import deepcopy

# TODO
# - more efficient discrete optimizr (computing just the energy correction)
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
    def get_energy(self):
        return energy_jax(self.mu,self.pairs,self.j,self.den)
    def get_local_energy(self,**kwargs):
        return get_local_energy(self,**kwargs)
    def get_local_mu(self,**kwargs):
        return get_local_mu(self,**kwargs)
    def optimize_energy(self,**kwargs):
        """Optimize the energy"""
#        print(self.den) ; exit()
        fun = lambda x: energy_jax(self.mu,self.pairs,self.j,x)
        x,es = optimize_discrete(fun,self.den,**kwargs) # optimize
        self.den = x # overwrite
        return es
    def get_correlator(self,**kwargs):
        """Return the nearest neighbor correlators"""
        from .statphystk.correlator import get_nnc
        return get_nnc(self.geometry,self.den,**kwargs)
    def copy(self):
        return deepcopy(self)

from numba import jit

def energy_jax(mu,pairs,js,den):
    if len(pairs)==0:
        pairs = np.array([[0,0]])
        js = np.array([0.])
    return energy_jax_jit(mu,pairs,js,den)


@jit(nopython=True)
def energy_jax_jit(mu,pairs,js,den):
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



def optimize_discrete(fun,x0,temp=0.1,ntries=1e5,info=False):
    """Discrete optimization, using a swap method"""
    inds = np.array(range(len(x0))) # indexes
    vals = np.unique(x0) # different values
    if len(vals)!=2: 
        print(vals)
        raise # not implemented
    def swap(x):
        x = x.copy()
        ainds = inds[x==vals[0]] # indexes for first value
        binds = inds[x==vals[1]] # indexes for second value
#        i1 = np.random.randint(0,n) # one random site
#        i2 = np.random.randint(0,n) # one random site
        j1 = np.random.randint(0,len(ainds)) # one random site
        j2 = np.random.randint(0,len(binds)) # one random site
        i1 = ainds[j1]
        i2 = binds[j2]
        x1 = x[i1]
        x2 = x[i2]
        x[i2] = x1 # swap
        x[i1] = x2 # swap
        return x
    def nswap(x,n):
        for ii in range(n):
            x = swap(x)
        return x
    n = len(x0) # number of sites
    nrep = ntries # this many iterations
    xold = x0.copy()
    es = np.zeros(int(nrep)) # storage for energies
    for ii in range(int(nrep)): # this many iterations
        x = nswap(xold,np.random.randint(1,4)) # make 1,2,3 swaps
        eo = fun(xold) # old
        en = fun(x) # new
        if info: print(en)
        if en<=eo: # smaller or same
            es[ii] = en # store new energy
            xold = x # overwrite
        else: 
            fac = np.exp((eo-en)/temp) # acceptance probability
            if np.random.random()<fac: 
                es[ii] = en # store new energy
                xold = x # overwrite
            else: 
                es[ii] = eo # keep old energy
                pass # do nothing
    return xold,es # return old one



def get_local_energy(LG,normalize=False):
    """Return the local energy at each site for the current snapshot"""
    def get(ii): # get for site ii
        LG0 = LG.copy() # make a dummy copy
        pairs0 = [] # empty list
        j0 = [] # empty list
        for (p,j) in zip(LG.pairs,LG.j): # take only those for this site
            if ii==p[0] or ii==p[1]: # site ii is here
                pairs0.append(p)
                j0.append(j)
        mu0 = LG.mu*0. # zeros
        mu0[ii] = LG.mu[ii]
        LG0.pairs = np.array(pairs0)
        LG0.j = np.array(j0)/2.
        LG0.mu = mu0
        enii = LG0.get_energy()
        if normalize: return enii/np.sum(LG0.j)
        else: return enii
    return np.array([get(ii) for ii in range(len(LG.geometry.r))]) # loop over positions



def get_local_mu(LG,normalize=False):
    """Return the local chemical potential"""
    def get(ii): # get for site ii
        LG0 = LG.copy() # make a dummy copy
        pairs0 = [] # empty list
        j0 = [] # empty list
        for (p,j) in zip(LG.pairs,LG.j): # take only those for this site
            if ii==p[0] or ii==p[1]: # site ii is here
                pairs0.append(p)
                j0.append(j)
        mu0 = LG.mu*0. # zeros
        mu0[ii] = LG.mu[ii]
        LG0.pairs = np.array(pairs0)
        LG0.j = np.array(j0)/2.
        LG0.den[ii] = 1.0 # overwrite to return chemical potential
        LG0.mu = mu0
        enii = LG0.get_energy()
        if normalize: return enii/np.sum(LG0.j)
        else: return enii
    return np.array([get(ii) for ii in range(len(LG.geometry.r))]) # loop over positions



def random_density(Ntot,N):
    """Generate an array with N 1's and Ntot-N 0's,
    with the 1's randomly distributed"""
    out = np.zeros(Ntot) # initialize
    inds = [i for i in range(Ntot)] # indexes with 1's
    for ii in range(N): # loop
        jj = np.random.randint(0,len(inds)) # random number from 0 to randint
        ind = inds[jj] # index with 1
        out[ind] = 1.0
        del inds[jj] # remove this one
    return np.array(out)



