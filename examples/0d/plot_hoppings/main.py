# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import operators
g = geometry.honeycomb_lattice()
#g = geometry.honeycomb_lattice_C6()
#g = geometry.chain()
g = g.supercell(6)
g.write()
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=True)
h = h.get_multicell()
from pyqula import groundstate
groundstate.hopping(h,nrep=2) # write three replicas







