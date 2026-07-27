import numpy as np
import types


def hubbard_mf(EB,**kwargs):
    """Wrapper to perform a mean-field Hubbard calculation with an Embedding
    object.

    CAVEAT (not currently exercised by any test or example -- EB.
    get_mean_field_hamiltonian, the only caller of this function, is not
    itself called anywhere in this repo): this relies on
    get_mean_field_hamiltonian's SCF loop reading the density matrix via
    h.get_density_matrix(...), which the get_density_matrix override below
    redirects to the Embedding object's own density matrix. For a spinful
    h (has_spin=True) that goes through VJinteraction (this is
    get_mean_field_hamiltonian's default engine), that assumption breaks:
    VJinteraction's SCF core (selfconsistency.spinspin._run_anisotropic_scf)
    computes its density matrix by diagonalizing h.get_hk_gen() directly
    (densitymatrix.full_dm_accumulate_sparse/_with_fermi) rather than
    calling h.get_density_matrix, silently ignoring this override and the
    fermi/shift_fermi ones below with it. A has_spin=False h still goes
    through Vinteraction (get_mean_field_hamiltonian falls back to it for
    spinless Hamiltonians), where this override is honored as before."""
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


