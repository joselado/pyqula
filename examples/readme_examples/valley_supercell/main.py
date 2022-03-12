# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice() # get the geometry object
g = g.get_supercell(7) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian object
(k,e,v) = h.get_bands(operator="valley") # compute the band structure
