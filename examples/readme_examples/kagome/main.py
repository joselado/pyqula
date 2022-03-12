# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
g = geometry.kagome_lattice() # get the geometry object
h = g.get_hamiltonian() # get the Hamiltonian object
(k,e) = h.get_bands() # compute the band structure
