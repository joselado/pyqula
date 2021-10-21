# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
g = geometry.chain() # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=10.0,filling=0.2,mf="ferro")
#exit()
h.get_bands(operator="sz")







