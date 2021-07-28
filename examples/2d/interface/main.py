# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
h1 = h.copy()
h2 = h.copy()
h1.add_haldane(0.1)
h2.add_haldane(-0.1)
kdos.interface(h1,h2)







