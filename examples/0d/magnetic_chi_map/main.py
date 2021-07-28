# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands

import numpy as np
from pyqula import geometry
g = islands.get_geometry(name="honeycomb",n=4,nedges=3,rot=np.pi/3) 
#g = geometry.bichain()
#g = g.supercell(10)
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=True)
#h.add_rashba(0.3)
from pyqula import susceptibility
from pyqula import parallel
parallel.cores = 5
susceptibility.dominant_correlation(h,write=True)









