# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.environ["PYQULAROOT"])

import numpy as np
from pyqula import geometry

g = geometry.honeycomb_lattice()
g = g.get_supercell((2,2))
h = g.get_hamiltonian()
h0 = h.copy() # copy Hamiltonian
# two ways of adding extended swave
from pyqula.superconductivity import hopping2deltaud
h1 = h0.copy() ; h1.add_pairing(mode="swave",nn=1,delta=0.1)
h2 = hopping2deltaud(h0,h0*0.1)
if not (h1-h2).is_zero():
    print("H1-H2 is not zero")
    raise

