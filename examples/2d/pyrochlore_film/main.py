# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import films


g = geometry.pyrochlore_lattice()
g = films.geometry_film(g,nz=10)
h = g.get_hamiltonian()
h.add_kane_mele(0.1)
h.turn_dense()
h.get_bands(operator="zposition")







