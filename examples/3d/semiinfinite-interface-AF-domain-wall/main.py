# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
g = geometry.diamond_lattice_minimal()
h = g.get_hamiltonian(has_spin=True)
h1 = h.copy()
h2 = h.copy()
h1.add_antiferromagnetism(0.5)
h2.add_antiferromagnetism(-0.5)
from pyqula import kdos
kdos.interface(h1,h2)







