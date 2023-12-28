# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
import numpy as np
g = geometry.chain() # generate the geometry
g = g.get_supercell(3)

h = g.get_hamiltonian(has_spin=False)

h.turn_multicell()


print("Direction")
print([0,0,0])
print("Matrix")
print(h.intra.real)


for hop in h.hopping:
    print("Direction")
    print(hop.dir)
    print("Matrix")
    print(hop.m.real)



