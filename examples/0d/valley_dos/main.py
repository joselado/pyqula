# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands
from pyqula import spectrum
from pyqula import operators

import numpy as np
g = islands.get_geometry(name="honeycomb",n=6,nedges=6,rot=0.0) # get an island

i = g.get_central()[0]
g = g.remove(i)

h = g.get_hamiltonian(has_spin = False)
h.add_peierls(0.05)
h.add_sublattice_imbalance(0.1)
#h.add_antihaldane(0.1)
#h.add_zeeman(0.3)

fv = operators.get_valley(h,projector=True)() # valley function
#fv = h.get_operator("sz")
#fv = None
#fv = h.get_operator("sublattice")
#fv = np.abs(fv)
h.get_dos(operator=fv,delta=0.02,use_kpm=False)








