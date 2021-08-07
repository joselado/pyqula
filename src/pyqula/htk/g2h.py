import numpy as np

from .. import specialhopping

def get_hamiltonian(self,fun=None,has_spin=True,ts=None,
                        is_sparse=False,spinful_generator=False,nc=2,
                        is_multicell=False,mgenerator=None,**kwargs):
    """ Create the hamiltonian for this geometry. By default, it assumes
    first neighbor hopping
        - has_spin = True, whether if to include spin degree of freedom
        - is_sparse = False, use sparse representation
        - ts = None, list of 1st,2nd,3rd neighbor hoppings
    """
    if ts is not None: # if hoppings given, overwrite mgenerator
        mgenerator = specialhopping.neighbor_hopping_matrix(self,ts)
        is_multicell = True # overwrite
        nc = len(ts) + 1 # overwrite
    if self.dimensionality==3: is_multicell=True
    from ..hamiltonians import Hamiltonian
    h = Hamiltonian(self)  # create the object
    h.is_sparse = is_sparse
    h.has_spin = has_spin
    h.is_multicell = is_multicell
    if is_multicell:  # workaround for multicell hamiltonians
      from ..multicell import parametric_hopping_hamiltonian
      if mgenerator is not None:
          from ..multicell import parametric_matrix # not implemented
          h = parametric_matrix(h,fm=mgenerator,cutoff=nc)
      else: h = parametric_hopping_hamiltonian(h,fc=fun) # add hopping
      return h
    if fun is None and mgenerator is None: # no function given
      h.first_neighbors()  # create first neighbor hopping
    else: # function or mgenerator given
      if h.dimensionality<3:
        from ..hamiltonians import generate_parametric_hopping
        h = generate_parametric_hopping(h,f=fun,
                  spinful_generator=spinful_generator,
                  mgenerator=mgenerator) # add hopping
      elif h.dimensionality==3:
        if mgenerator is not None: raise # not implemented
        from ..multicell import parametric_hopping_hamiltonian
        h = parametric_hopping_hamiltonian(h,fc=fun,**kwargs) # add hopping
    if not is_sparse: h.turn_dense() # dense Hamiltonian
    return h # return the object

