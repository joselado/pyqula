# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


raise # this does not work yet


import numpy as np
from pyqula import geometry
from pyqula import spectrum
g0 = geometry.honeycomb_lattice()
g = g0.get_supercell(2)
h = g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(.4)

from pyqula import unfolding
op = unfolding.bloch_projector(h,g0)
h.get_qpi(delta=1e-1,mode="pm",operator=op,info=True,nsuper=5,
  nunfold=2)







