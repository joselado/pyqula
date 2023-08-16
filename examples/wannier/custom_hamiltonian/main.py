# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import hamiltonians
import numpy as np

# create a dictionary with the hopping matrices in each direction
# the key should be the associated unit cell vector (as integer tuple)
# the element should be the hopping matrix
d = {(0,0,0):[[0.0,1.0],[1.0,0.0]],
     (1,0,0):[[1,0.],[0.,1.]],
     (-1,0,0):[[1,0.],[0.,1.]],
     }

# those matrices could be the hoppings obtained from Wannier

# create a dummy Hamiltonian for those hopping matrices
h = hamiltonians.generate_hamiltonian_from_dict(d)


# this Hamiltonian will work as a typical one
h.get_bands()


