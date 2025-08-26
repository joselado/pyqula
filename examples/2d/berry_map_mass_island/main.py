# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





import numpy as np
from pyqula import geometry
from pyqula import hamiltonians
from pyqula import specialgeometry
from pyqula import topology
#raise # this does not work yet
g = geometry.honeycomb_lattice()
g = g.supercell(12)
h = g.get_hamiltonian(has_spin=False)
rmax = np.sqrt(g.a1.dot(g.a1))
def fm(r): 
#  dr = np.sqrt(r.dot(r)) - rmax/3.
#  return np.tanh(dr)
  if np.sqrt(r.dot(r))<rmax/3: return 1.0
  else: return -1.0
h.add_sublattice_imbalance(fm)
#h.add_haldane(lambda r1,r2: fm((r1+r2)/2))
from pyqula import parallel

# spatially resolved Berry curvature
topology.Omega_rmap(h,k=[0.,0.,0.0],nrep=3,
        integral=True,eps=1e-4,delta=1e-2,operator="valley")







