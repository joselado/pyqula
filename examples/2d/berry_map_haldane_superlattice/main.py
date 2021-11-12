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
g = g.supercell(16)
h = g.get_hamiltonian(has_spin=False)
rmax = np.sqrt(g.a1.dot(g.a1))
def fm(r): 
  if np.sqrt(r.dot(r))<rmax/3: return 0.3
  else: return 0.0
h.add_haldane(fm)
from pyqula import parallel
topology.berry_green_map(h,
        k=[0.,0.,0.0],# do it just at the Gamma point
        nrep=3, # how many supercells to print
        integral=True, # perform the integral to the Fermi energy
        eps=1e-4, # error for the integral
        delta=1e-2 # analytic continuation
        )







