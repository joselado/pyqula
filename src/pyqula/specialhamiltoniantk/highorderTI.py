# Hamiltonians for second order topological insulators
import numpy as np

from .. import geometry 
def square_2OTI(delta=0.):
    """Return the Hamiltonian of a 2nd order TI in the square lattice"""
    d = delta # redefine
    g = geometry.square_lattice()
    g = g.get_supercell((2,2)) # make a supercell
    h = g.get_hamiltonian(has_spin=False)
    h.geometry.center_in_atom() # to get a real gauge
    m = h.intra*(1.-d) # decrease hopping
#    m[0,1] *= -1. # do it by hand
#    m[1,0] *= -1. # do it by hand
    h = h*(1+d) # scale all
    # this is should be written properly
    h.intra = m
    h.add_peierls(0.5)
    return h # return Hamiltonian



def chain_1OTI(delta=0.):
    """Return the Hamiltonian of a 2nd order TI in the square lattice"""
    d = delta # redefine
    g = geometry.chain()
    g = g.get_supercell(2) # make a supercell
    h = g.get_hamiltonian(has_spin=False)
    m = h.intra*(1.-d) # decrease hopping
    h = h*(1+d) # scale all
    # this is should be written properly
    h.intra = m
    return h # return Hamiltonian






