# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
g = geometry.chain(2) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=False) # get the Hamiltonian,spinless
delta = -0.1 # value of the perturbation
h.intra *= (1.+delta) # modify intra-hopping
h.inter *= (1.-delta) # modify inter-hopping
phi = topology.berry_phase(h) # get the berry phase
print(phi)
hf = h.supercell(100) # do a supercell
hf = hf.set_finite_system(periodic=False) # do an open finite system
hf.get_bands()







