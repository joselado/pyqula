import numpy as np
from scipy.sparse import coo_matrix



class LatticeGas():
    def __init__(self,g,filling=0.5): # geometry
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
        self.den[0:int(np.round(self.nsites*filling))] = 1.0 # put some to 1
    def add_interaction(self,Jij=None,**kwargs):
        h = self.geometry.get_hamiltonian(has_spin=False,tij=Jij)
        m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
        pairs = np.array([m.row,m.col]).transpose() # convert to array
        self.pairs = np.concatenate([self.pairs,pairs]) # add interaction
        self.j = np.concatenate([self.j,m.data]).real # store
    def get_energy(self):
        return energy_jax(self.mu,self.pairs,self.j,self.den)
    def optimize_energy(self,**kwargs):
        """Optimize the energy"""
#        print(self.den) ; exit()
        fun = lambda x: energy_jax(self.mu,self.pairs,self.j,x)
        x = optimize_discrete(fun,self.den,**kwargs) # optimize
        self.den = x # overwrite
        return self.get_energy()

from numba import jit

@jit(nopython=True)
def energy_jax(mu,pairs,js,den):
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



def optimize_discrete(fun,x0,temp=0.1,ntries=1e5):
    """Discrete optimization, using a swap method"""
    def swap(x):
        x = x.copy()
        i1 = np.random.randint(0,n) # one random site
        i2 = np.random.randint(0,n) # one random site
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
    for ii in range(int(nrep)): # this many iterations
        x = nswap(xold,np.random.randint(1,4)) # make 1,2,3 swaps
        eo = fun(xold) # old
        en = fun(x) # new
#        print(eo/len(x))
        if en<eo: xold = x # overwrite
        else: 
            fac = np.exp((eo-en)/temp) # acceptance probability
            if np.random.random()<fac: xold = x # overwrite
            else: pass # do nothing
    return xold # return old one







