import numpy as np

def modify_hamiltonian_matrices(self,f0,use_geometry=False):
    """Apply a certain function to all the matrices"""
    # wrapper function
    if use_geometry: # use geometry
        g = self.geometry # get the geometry
        def f(m,dr): 
            return f0(m,g.r,g.replicas(d=-np.array(dr))) # use r1 and r2
    else:
        def f(m,dr): return f0(m) # do not use geometry
    # apply the function
    self.intra = f(self.intra,[0,0,0]) # modify intracell
    if self.dimensionality==0: return  # zero dimensional systems
    self.turn_multicell() # multicell for all
    if self.is_multicell: # for multicell hamiltonians
      for i in range(len(self.hopping)): # loop over hoppings
        # modify Hamiltonian matrix
        self.hopping[i].m = f(self.hopping[i].m,self.hopping[i].dir) 
    else: # conventional way, now disabled
      raise


from .. import algebra

def get_dense(self):
    """ Transforms the hamiltonian into a sparse hamiltonian"""
    def f(m):
        return algebra.todense(m)
    h = self.copy() # make a copy
    h.modify_hamiltonian_matrices(f) # modify the matrices
    h.is_sparse = False # sparse flag to true
    if not self.is_multicell:  h = h.get_no_multicell() # no mult mode
    return h
