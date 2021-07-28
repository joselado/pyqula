# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import specialhamiltonian

g = geometry.honeycomb_lattice()
h = specialhamiltonian.flux2d(g,n=80)
h.geometry.write()
h.get_bands()







