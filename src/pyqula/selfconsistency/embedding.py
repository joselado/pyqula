import numpy as np
import types


def hubbard_mf(EB,**kwargs):
    """Wrapper to perform a mean-field Hubbard calculation with an Embedding
    object."""
    ## This is just a workaround
    h = EB.H.copy() # copy Hamiltonian
    h.EB = EB.copy() # copy the object
    def dm(self,**kwargs):
        return {(0,0,0):self.EB.get_density_matrix(**kwargs)}
    def update(self,*args):
        self.EB.set_multihopping(*args)
    def fermi(self,*args,**kwargs): return 0.0
    def shift_fermi(self,*args,**kwargs): return None
    h.get_density_matrix = types.MethodType(dm,h) # overwrite
    h.set_multihopping = types.MethodType(update,h) # overwrite
    h.get_fermi4filling = types.MethodType(fermi,h) # overwrite
    h.shift_fermi = types.MethodType(shift_fermi,h) # overwrite
    h = h.get_mean_field_hamiltonian(**kwargs) # get the mean-field Hamiltonian
    return h.EB.copy() # return the new embedding object


