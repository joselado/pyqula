# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



# zigzag ribbon
from pyqula import geometry
g = geometry.chain() # create geometry 
h = g.get_hamiltonian(tij=[1.0,.2]) # create hamiltonian of the system
h.get_bands() # calculate band structure







