# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




#g = geometry.square_lattice()
from pyqula import geometry
g = geometry.triangular_lattice() # create a triangular lattice
g = g.supercell(10) # create a supercell for the moire


print("Reciprocal lattice vectors")
print("B1 = ",g.b1)
print("B2 = ",g.b2)
