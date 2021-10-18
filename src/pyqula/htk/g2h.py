import numpy as np

from .. import specialhopping

def get_hamiltonian(self,tij=None,fun=None,has_spin=True,ts=None,
                        is_sparse=False,spinful_generator=False,nc=2,
                        is_multicell=False,mgenerator=None,**kwargs):
    """ Create the hamiltonian for this geometry. By default, it assumes
    first neighbor hopping
        - has_spin = True, whether if to include spin degree of freedom
        - is_sparse = False, use sparse representation
        - ts = None, list of 1st,2nd,3rd neighbor hoppings
        - tij = None, function that return the spatially dependent hoppings
    """
    ### Perform some initial sanity checks
    if fun is not None: 
        print("fun will be deprecated, use instead tij")
        tij = fun # overwrite
    ## in case tij is an iterable with hoppings
    from ..checkclass import is_iterable
    if is_iterable(tij): # tij is an iterable
        ts = tij # overwrite
        tij = None
    #####
    if ts is not None: # if hoppings given, overwrite mgenerator
        mgenerator = specialhopping.neighbor_hopping_matrix(self,ts)
        is_multicell = True # overwrite
        nc = 2*len(ts) + 1 # overwrite
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
          h = parametric_matrix(h,fm=mgenerator,cutoff=nc,**kwargs)
      else: h = parametric_hopping_hamiltonian(h,fc=tij,**kwargs) # add hopping
      return h
    if tij is None and mgenerator is None: # no function given
      h.first_neighbors()  # create first neighbor hopping
    else: # function or mgenerator given
      if h.dimensionality<3:
        from ..hamiltonians import generate_parametric_hopping
        h = generate_parametric_hopping(h,f=tij,
                  spinful_generator=spinful_generator,
                  mgenerator=mgenerator) # add hopping
      elif h.dimensionality==3:
        if mgenerator is not None: raise # not implemented
        from ..multicell import parametric_hopping_hamiltonian
        h = parametric_hopping_hamiltonian(h,fc=tij,**kwargs) # add hopping
    if not is_sparse: h.turn_dense() # dense Hamiltonian
    return h # return the object

