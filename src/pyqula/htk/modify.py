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
    if not self.is_sparse:
        # Already dense: self.copy() below (in the general path) is a
        # full recursive deepcopy (geometry included), whose cost comes
        # from the sheer number of nested Python objects it traverses,
        # not from array size -- and it exists purely so that a caller
        # mutating the result's matrices in place (e.g.
        # merge.merge_channels does `h.intra[i,j] = ...` after
        # get_dense()) doesn't corrupt self. A shallow copy plus fresh
        # copies of just the matrix attributes gives that same safety
        # far more cheaply -- and, for a non-multicell Hamiltonian, also
        # avoids the general path's unnecessary round trip through a
        # multicell representation and back (self.turn_multicell() then
        # h.get_no_multicell()). This is hit once per energy in hot
        # loops like the LocalProbe Keldysh sideband sweep, where that
        # combination dominated the profile for nothing (dense in,
        # dense out).
        from copy import copy as _shallow_copy
        if self.is_multicell:
            # multicell.turn_multicell(self) (which
            # modify_hamiltonian_matrices would otherwise call) is a
            # no-op returning self unchanged when already multicell --
            # i.e. the slow path's hopping list would itself only be
            # independent because self.copy() already made it so, so
            # building a fresh one here directly is equivalent, not an
            # approximation.
            from ..multicell import Hopping
            h = _shallow_copy(self)
            h.intra = algebra.todense(self.intra)
            h.hopping = [Hopping(d=t.dir, m=algebra.todense(t.m))
                         for t in self.hopping]
            return h
        elif self.dimensionality==0:
            h = _shallow_copy(self)
            h.intra = algebra.todense(self.intra)
            return h
        elif self.dimensionality==1:
            h = _shallow_copy(self)
            h.intra = algebra.todense(self.intra)
            h.inter = algebra.todense(self.inter)
            return h
        elif self.dimensionality==2:
            h = _shallow_copy(self)
            h.intra = algebra.todense(self.intra)
            for attr in ("tx","ty","txy","txmy"):
                setattr(h, attr, algebra.todense(getattr(self,attr)))
            return h
        # dimensionality==3, non-multicell: falls through to the general,
        # always-correct path below -- not exercised by the hot paths
        # this targets, not worth risking a hand-rolled shortcut for.
    def f(m):
        return algebra.todense(m)
    h = self.copy() # make a copy
    h.modify_hamiltonian_matrices(f) # modify the matrices
    h.is_sparse = False # sparse flag to true
    if not self.is_multicell:  h = h.get_no_multicell() # no mult mode
    return h
