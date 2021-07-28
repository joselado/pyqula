# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





# zigzag ribbon
from pyqula import geometry
import ribbon
g = geometry.honeycomb_lattice()
g = ribbon.bulk2ribbon(g,n=5,boundary=[6,1])
h = g.get_hamiltonian(has_spin=False)
h.get_bands()
g = g.supercell(4)
g.write()







