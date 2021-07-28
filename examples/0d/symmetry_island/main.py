# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import geometry
from pyqula import spectrum
from pyqula import operators

import numpy as np
g = islands.get_geometry(name="honeycomb",n=1.5,nedges=6,rot=0.0) # get an island
g = geometry.chain()
g = g.supercell(6)
#g.write()

h = g.get_hamiltonian(has_spin=False)
print("Operators")
m = h.get_hk_gen()([0.,0.,0.])
print(m)

from pyqula import symmetry
symmetry.commuting_matrices(m)








