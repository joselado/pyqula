# function to modify the Hamiltonian matrices according to criteria
import numpy as np


def remove_hopping(self,f):
    """Remove hoppings to site according to criteria from the geometry"""
    g = self.geometry
    remove = np.array([f(g.r[i]) for i in range(len(g.r))]) # sites to remove
    if self.has_spin and not self.has_eh: mask = np.repeat(remove,2) # spin-doubled orbitals
    elif not self.has_spin and not self.has_eh: mask = remove
    else: raise
    def fm(m): # function to modify matrices
        m[mask,:] = 0.
        m[:,mask] = 0.
        return m
    self = self.copy()
    self.modify_hamiltonian_matrices(fm)
    return self

