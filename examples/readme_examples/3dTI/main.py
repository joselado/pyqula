# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import films
import numpy as np
g = geometry.diamond_lattice() # create a diamond lattice
g = films.geometry_film(g,nz=60) # create a thin film
h = g.get_hamiltonian() # generate Hamiltonian
h.add_strain(lambda r: 1.+abs(r[2])*0.8,mode="directional") # add axial strain
h.add_kane_mele(0.1) # add intrinsic spin-orbit coupling
(k,e,c) = h.get_bands(operator="surface") # compute band structure
