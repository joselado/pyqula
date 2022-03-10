# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
from pyqula import operators
g = geometry.honeycomb_lattice()
g = g.get_supercell(6)
g = g.get_supercell(6)
exit()
g = g.remove(0)
h = g.get_hamiltonian(has_spin=True)
h.get_bands()
h.get_ldos(e=0.0,delta=1e-2)
#dos.dos(h,nk=100,use_kpm=True)







