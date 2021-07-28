# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry,potentials

g = geometry.triangular_lattice() # create the geometry of the lattice
g = g.supercell(7) # create a supercell of a certain size
J = potentials.commensurate_potential(g) # get this potential (C3 symmetry)
J = J.normalize() # set the value of the potential between 0 and 1 
Delta = 1.0 - J # define the SC order as 1 - the exchange
J = J*0.3 # reduce the strength of the exchange
Delta = Delta*0.3 # reduce the strength of the SC order
g.write_profile(J,nrep=2,name="EXCHANGE.OUT") # write the exchange in file
g.write_profile(Delta,nrep=2,name="DELTA.OUT") # write the SC in file
h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian
h.add_zeeman(J) # add exchange coupling
h.add_swave(Delta) # add the SC order
h.get_bands() # compute the bands







