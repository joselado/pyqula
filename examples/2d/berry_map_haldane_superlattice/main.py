# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
g = g.supercell(6)
h = g.get_hamiltonian(has_spin=True)
rmax = np.sqrt(g.a1.dot(g.a1))
def fm(r): 
  if np.sqrt(r.dot(r))<rmax/3: return 0.3
  else: return 0.0
h.add_haldane(fm)
h.add_swave(0.02)
from pyqula import parallel
topology.Omega_rmap(h,
        energy = 0.5,
        k=[0.,0.,0.0],# do it just at the Gamma point
        nrep=3, # how many supercells to print
        integral=True, # perform the integral to the Fermi energy
        eps=1e-4, # error for the integral
        delta=1e-2 # analytic continuation
        )

# the result is print to BERRY_RMAP.OUT
(x,y,d,z) = np.genfromtxt("BERRY_RMAP.OUT").T

import matplotlib.pyplot as plt

plt.scatter(x,y,c=d,cmap="bwr")
plt.colorbar(label="Berry density")
plt.xlabel("x")
plt.ylabel("y")
plt.show()





