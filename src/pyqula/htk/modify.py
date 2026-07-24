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
    if not self.is_sparse and self.is_multicell:
        # Already dense and already multicell: self.copy() below is a
        # full recursive deepcopy (geometry included), whose cost comes
        # from the sheer number of nested Python objects it traverses,
        # not from array size -- and it exists purely so that a caller
        # mutating the result's matrices in place (e.g.
        # merge.merge_channels does `h.intra[i,j] = ...` after
        # get_dense()) doesn't corrupt self. A shallow copy plus fresh
        # copies of just the matrix attributes gives that same safety
        # far more cheaply. Restricted to the already-multicell case,
        # where multicell.turn_multicell(self) (which
        # modify_hamiltonian_matrices would otherwise call) is a no-op
        # returning self unchanged -- i.e. the slow path's hopping list
        # would itself only be independent because self.copy() already
        # made it so, so building a fresh one here directly is
        # equivalent, not an approximation. This path is hit once per
        # energy in hot loops like the LocalProbe Keldysh sideband
        # sweep, where the full deepcopy dominated the profile for
        # nothing (dense in, dense out).
        from copy import copy as _shallow_copy
        from ..multicell import Hopping
        h = _shallow_copy(self)
        h.intra = algebra.todense(self.intra)
        h.hopping = [Hopping(d=t.dir, m=algebra.todense(t.m))
                     for t in self.hopping]
        return h
    def f(m):
        return algebra.todense(m)
    h = self.copy() # make a copy
    h.modify_hamiltonian_matrices(f) # modify the matrices
    h.is_sparse = False # sparse flag to true
    if not self.is_multicell:  h = h.get_no_multicell() # no mult mode
    return h
