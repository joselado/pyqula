# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])

import numpy as np
from pyqula import geometry

g = geometry.triangular_lattice()
g = g.get_supercell((2,2))
h = g.get_hamiltonian()
h0 = h.copy() # copy Hamiltonian
# two ways of adding extended swave
from pyqula.sctk import reorder
reorder.dense = False
d = np.random.random(3)
h1 = h.copy() ; h1.add_pairing(mode="pwave",d=d) ; h1.add_swave(0.2)
reorder.dense = True
h2 = h.copy() ; h2.add_pairing(mode="pwave",d=d) ; h2.add_swave(0.2)
if not (h1-h2).is_zero():
    print("Reordering in Nambu not working")
    raise


