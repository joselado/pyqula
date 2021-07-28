# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import islands

import numpy as np
from pyqula import geometry
g = islands.get_geometry(name="honeycomb",n=20,nedges=20,rot=0.) 
#g = geometry.bichain()
#g = g.supercell(100)
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=False)
#h.add_haldane(1.0)
h.add_peierls(0.1)
h.shift_fermi(0.5)
#h.add_sublattice_imbalance(0.2)
#h.get_bands()
#exit()
#h.shift_fermi(0.1)
from pyqula import timeevolution
from pyqula import parallel
parallel.cores = 5
from pyqula import chi
#chi.chargechi_reciprocal(h) ;  exit()
timeevolution.evolve_local_state(h,i=0,ts=np.linspace(0.,100,100),
        mode="green")








