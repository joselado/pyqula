# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")





from pyqula import geometry
from pyqula import specialhopping
from pyqula import dos
g = geometry.honeycomb_lattice()
g = g.get_supercell(11)
h = g.get_hamiltonian(has_spin=False,is_sparse=True)


import numpy as np

omega = np.pi*2./np.sqrt(g.a1.dot(g.a1)) # modulation frequency

from pyqula.strain import graphene_buckling

pot = graphene_buckling(omega=omega,dt=0.2,geometry=g) # special function for buckled lattices
h.add_strain(pot,mode="non_uniform")
h.write_hopping()
(kb,eb) = h.get_bands(num_bands=20)
(x,y,ld) = h.get_ldos(e=0.,nrep=2)
h.turn_dense()
(e,d) = h.get_dos(energies=np.linspace(-3.5,3.5,500),nk=30,delta=1e-2)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(kb,eb,c="black",s=10)
plt.ylabel("Energy") ; plt.xlabel("Momentum") ; plt.xticks([])


plt.subplot(1,3,2)
plt.scatter(x,y,c=ld,cmap="inferno")
plt.xlabel("x") ; plt.ylabel("y") ; plt.axis("equal")


plt.subplot(1,3,3)
plt.plot(e,d)
plt.xlabel("Energy") ; plt.ylabel("DOS") ; plt.ylim([0,max(d)]) 

plt.tight_layout()

plt.show()





