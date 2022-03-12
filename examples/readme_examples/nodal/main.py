# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula import films
g = geometry.diamond_lattice()
g = films.geometry_film(g,nz=20)
h = g.get_hamiltonian()
(k,e) = h.get_bands()
