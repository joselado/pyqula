# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
import numpy as np
g = geometry.chain() # create a chain
N = 2 # number of sites per unit cell
g = g.get_supercell(N) # create supercell

h = g.get_hamiltonian() # get the Hamiltonian

# add onsite modulation
h.add_onsite(lambda r: 0.3*np.cos(np.pi*2/N*(r[0]-g.r[0][0]))) 

h.add_onsite(1.8) # add chemical potential

h.add_rashba(0.3) # add Rashba SOC
h.add_zeeman([0.,0.,0.3]) # add Zeeman field
h.add_swave(0.1) # add superconducting pairing
from pyqula.topology import berry_phase
# Compute the topological invariant (it only works if you have a gap)
# the invariant is pi for topological, 0 for trivial
print("Berry phase",berry_phase(h))








