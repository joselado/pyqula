# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])


import numpy as np
from pyqula import geometry

# create a supercell
g = geometry.square_lattice() # take a square lattice
g = g.get_supercell(8) # make a supercell
h = g.get_hamiltonian() # generate the Hamiltonian

nk = 60 # number of kpoints (you may want to make it bigger to check convergence)

h.add_rashba(0.12) # add Rashba SOC
from pyqula import parallel ; parallel.cores = 7
from pyqula import potentials
h.add_exchange(potentials.impurity(g.r[0],v=[0.,0., 2.1])) # add impurity
# set a specific filling
h.set_filling(0.14,nk=nk) 
# add superconductivity
h.add_swave(0.06)
# Compute the gap
print("The gap is",h.get_gap()) 
# compute the Chern number
c = h.get_chern(nk=nk)
print("The Chern number is",c) # show the Chern number

