# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import topology
from pyqula import dos
from pyqula import operators
g = geometry.honeycomb_lattice()
g = g.supercell(2)
g = g.remove(0)
h = g.get_hamiltonian(has_spin=True)
f = operators.get_inplane_valley(h)
h.get_bands(operator=f)
#dos.dos(h,nk=100,use_kpm=True)







