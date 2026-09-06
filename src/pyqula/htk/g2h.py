import numpy as np

from .. import specialhopping

def get_hamiltonian(self,tij=None,has_spin=True,
                        is_sparse=False,spinful_generator=False,nc=2,
                        non_hermitian = False,
                        is_multicell=False,mgenerator=None,**kwargs):
    """ Create the hamiltonian for this geometry. By default, it assumes
    first neighbor hopping
        - has_spin = True, whether if to include spin degree of freedom
        - is_sparse = False, use sparse representation
        - tij = None, either a function returning the spatially dependent
          hopping, a specialhopping.HoppingGenerator, or a list of the
          1st,2nd,3rd... neighbor hoppings
        - is_multicell = False, store the hoppings as a multicell dictionary
        - mgenerator = None, generator of the full hopping matrix
        - nc = 2, neighbor cutoff for the multicell construction
    Any remaining keyword is forwarded to the multicell hopping builders
    (cutoff, rcut); an unknown one is an error rather than being ignored.
    """
    ### Perform some initial sanity checks
    # `fun` is what this argument used to be called before it was renamed
    # to `tij`. It is still spelled that way in several places (spinwaves,
    # surface_TI, operators.get_sublattice_hopping_generator, examples),
    # and until now it was silently dropped -- so those built a plain
    # first-neighbor Hamiltonian instead of the one their hopping function
    # describes.
    if "fun" in kwargs:
        if tij is not None:
            raise TypeError("got both 'tij' and its old name 'fun'; pass "
              +"only one of them")
        tij = kwargs.pop("fun")
    # anything not consumed here is only meaningful for the multicell
    # builders below; unknown keywords used to be dropped in silence, so a
    # misspelled has_spin/is_sparse quietly gave the default Hamiltonian
    forwardable = ["cutoff","rcut"] # accepted by the multicell builders
    unknown = [k for k in kwargs if k not in forwardable]
    if len(unknown)>0:
        raise TypeError("get_hamiltonian() got unexpected keyword "
          +"argument(s) "+str(sorted(unknown))+"; the accepted ones are "
          +str(["tij","has_spin","is_sparse","spinful_generator","nc",
                "non_hermitian","is_multicell","mgenerator"]+forwardable))
    ## in case tij is an iterable with hoppings
    from ..checkclass import is_iterable
    if is_iterable(tij): # tij is an iterable
        ts = tij # overwrite the 1st, 2nd etc hoppings
        mgenerator = specialhopping.neighbor_hopping_matrix(self,ts)
        is_multicell = True # overwrite
        nc = 2*len(ts) + 1 # overwrite
        tij = None # set tij to None
    if type(tij)==specialhopping.HoppingGenerator: # for hopping generator
        mgenerator = tij.f # store the function
        tij = None # and set to None
    if self.dimensionality==3: is_multicell=True
    from ..hamiltonians import Hamiltonian
    h = Hamiltonian(self)  # create the object
    h.is_sparse = is_sparse
    h.non_hermitian = non_hermitian # non Hermitian flag
    h.has_spin = has_spin
    h.is_multicell = is_multicell
    if is_multicell:  # workaround for multicell hamiltonians
        from ..multicell import parametric_hopping_hamiltonian
        if mgenerator is not None: # if mgenerator is given
            from ..multicell import parametric_matrix # not implemented
            h = parametric_matrix(h,fm=mgenerator,cutoff=nc,**kwargs)
        else: 
            h = parametric_hopping_hamiltonian(h,fc=tij,**kwargs) # add hopping
#        return h
    else: # non multicell
        if tij is None and mgenerator is None: # no function given
          h.first_neighbors()  # create first neighbor hopping
        else: # function or mgenerator given
          if h.dimensionality<3:
            from ..hamiltonians import generate_parametric_hopping
            h = generate_parametric_hopping(h,f=tij,
                      spinful_generator=spinful_generator,
                      mgenerator=mgenerator) # add hopping
          elif h.dimensionality==3:
            if mgenerator is not None:
                raise NotImplementedError("a matrix generator is not "
                        "supported for 3d Hamiltonians, pass a hopping "
                        "function as tij instead")
            from ..multicell import parametric_hopping_hamiltonian
            h = parametric_hopping_hamiltonian(h,fc=tij,**kwargs) # add hopping
    # ensure right sparsity structure
    if not is_sparse: 
        h = h.get_dense() # dense Hamiltonian
    else:
        h.turn_sparse() # sparse Hamiltonian
    return h # return the object

