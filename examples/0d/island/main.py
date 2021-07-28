# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
import numpy as np
g = islands.get_geometry(name="honeycomb",n=3,nedges=3,rot=0.0) # get an island
# maximum distance to the origin
h = g.get_hamiltonian(has_spin=True) # get the Hamiltonian
h.get_bands(operator=h.get_operator("valley"))







