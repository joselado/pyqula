# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.chain() # geometry for a chain
g = g.get_supercell(2)
h = g.get_hamiltonian()
d = h.get_vev()

