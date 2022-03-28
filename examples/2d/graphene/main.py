# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
(k,e) = h.get_bands()







