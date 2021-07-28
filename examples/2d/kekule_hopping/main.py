# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
g = geometry.honeycomb_lattice()
#g = geometry.honeycomb_lattice_C6()
#g = geometry.chain()
g = g.supercell(3)
g.write()
h = g.get_hamiltonian(has_spin=False)
h = h.get_multicell()
h.add_kekule(0.4)
#h.add_haldane(0.3)
#h1 = h.copy() ; h1.clean() ; h1.add_haldane(0.3) ; h.add_hamiltonian(h1)
h.get_bands(operator="valley_x")
from pyqula import groundstate
groundstate.hopping(h,nrep=2) # write three replicas







