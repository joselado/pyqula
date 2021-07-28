# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.honeycomb_armchair_ribbon(40) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h.get_bands() # calculate band structure







