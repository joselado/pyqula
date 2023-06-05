import numpy as np



class LatticeGas():
    def __init__(self,g,filling=0.5): # geometry
        self.geometry = g # store geometry
        self.nsites = len(g.r) # number of sites
        self.mu = np.zeros(len(g.r)) # chemical potential
        self.den = np.zeros(len(g.r)) # chemical potential
        self.j = np.array([0]) # interactions
        self.pairs = np.array([[0,0]]) # empty list
    def set_filling(self,filling):
        """Set filling of the system"""
        self.den[:] = 0. # initialize all to zero
        self.den[0:int(np.round(self.nsites*filling))] = 1.0 # put some to 1
    def add_interaction(self,Jij=None,**kwargs):
        h = self.geometry.get_hamiltonian(has_spin=False,tij=Jij)
        m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
        pairs = np.array([m.row,m.col]).transpose() # convert to array
        self.pairs = np.concatenate([self.pairs,pairs]) # add interaction
        self.j = np.concatenate([self.j,m.data]) # store
    def get_energy(self):
        return energy_jax(self.mu,self.pairs,self.js,self.den)



@jit(nopython=True)
def energy_jax(mu,pairs,js,den):
    """Compute the energy of the lattice gas model"""
    nump = len(pairs) # number of pairs
    n = len(mu) # number of sites
    etot = 0. # output energy
    for i in range(n): etot = etot + mu[i]*den[i] # chemical potential
    for ip in len(nump):
        ii = pairs[ip][0]
        jj = pairs[ip][1]
        etot = etot + den[ii]*den[jj]*js[ip] # add contribution
    return etot



