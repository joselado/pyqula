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
g = g.get_supercell(16)
h = g.get_hamiltonian(has_spin=False)


omega = np.pi*2./np.sqrt(g.a1.dot(g.a1)) # modulation frequency

from pyqula.strain import graphene_buckling
pot = graphene_buckling(omega=omega,dt=0.2,geometry=g) # special function for buckled lattices
h.add_strain(pot,mode="non_uniform")
h.shift_fermi(0.1) # shift fermi energy

(x,y,d) = h.get_ldos(e=0.,nk=2,nrep=1)

# spatially resolved Berry curvature
b = topology.Omega_rmap(h,k=[0.,0.,0.0],nrep=3,nk=2,
        integral=False,eps=1e-4,delta=1e-2,operator="valley")

import matplotlib.pyplot as plt

plt.subplot(1,2,1) ; plt.title("LDOS")
plt.scatter(x,y,c=d,cmap="inferno",vmax=np.max(d)/3.) ; plt.axis("equal")
plt.colorbar(label="LDOS")

plt.subplot(1,2,2) ; plt.title("Valley Berry curvature")
plt.scatter(g.r[:,0],g.r[:,1],c=b,cmap="bwr") ; plt.axis("equal")

plt.colorbar(label="Valley Berry curvature")
plt.show()





