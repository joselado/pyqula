import numpy as np


def get_hamiltonian(self,fun=None,has_spin=True,
                        is_sparse=False,spinful_generator=False,
                        is_multicell=False,mgenerator=None,**kwargs):
    """ Create the hamiltonian for this geometry. By default, it assumes
    first neighbor hopping
        - has_spin = True, whether if to include spin degree of freedom
        - is_sparse = False, use sparse representation
    """
    if self.dimensionality==3: is_multicell=True
    from ..hamiltonians import hamiltonian
    h = hamiltonian(self)  # create the object
    h.is_sparse = is_sparse
    h.has_spin = has_spin
    h.is_multicell = is_multicell
    if is_multicell:  # workaround for multicell hamiltonians
      from ..multicell import parametric_hopping_hamiltonian
      if mgenerator is not None:
          from ..multicell import parametric_matrix # not implemented
          h = parametric_matrix(h,fm=mgenerator)
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

