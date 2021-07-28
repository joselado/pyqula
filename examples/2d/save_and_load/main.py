# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
import numpy as np
from pyqula import geometry
from pyqula import hamiltonians
g = geometry.honeycomb_lattice()
  
h = g.get_hamiltonian() # create hamiltonian of the system
h.save() # save the Hamiltonian
del h # delete the Hamiltonian
h1 = hamiltonians.load() # load the Hamiltonian
h1.get_bands() # get the bandstructure







