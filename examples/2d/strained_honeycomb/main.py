# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import specialhopping
from pyqula import dos
g = geometry.honeycomb_lattice()
g = g.supercell(11)
mgen = specialhopping.strained_hopping_matrix(g,dt=1.0,k=1,v=1.0)
h = g.get_hamiltonian(has_spin=False,mgenerator=mgen)
h.write_hopping()
h.get_bands()







