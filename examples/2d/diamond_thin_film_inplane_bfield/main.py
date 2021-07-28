# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
g = geometry.diamond_lattice_minimal()
from pyqula import films
g = films.geometry_film(g,nz=2)
h = g.get_hamiltonian(has_spin=False)
h = h.get_multicell()
h.add_inplane_bfield(b=0.05)
h.get_bands(operator="zposition")
h.get_dos(nk=50)







