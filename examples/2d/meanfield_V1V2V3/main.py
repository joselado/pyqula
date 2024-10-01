# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry, supercell
g = geometry.triangular_lattice()
g = supercell.turn_orthorhombic(g) # make a orthorombic cell
g = g.get_supercell((4,4)) # make a supercell
h = g.get_hamiltonian(has_spin=False) # generate Hamiltonian

# perform the mean-field calculation 
hscf = h.get_mean_field_hamiltonian(V1=20,V2=20,V3=1.,mf="random",
        filling=.5,verbose=1,nk=1,mix=0.1) # perform SCF calculation

# now plot the resulting density

import matplotlib.pyplot as plt
plt.title("Interaction-induced density")
d = hscf.get_vev(nk=2) 
x,y = g.x,g.y
plt.scatter(x,y,c=d,s=600,cmap="rainbow",vmin=0.,vmax=1.) 
plt.axis("equal") ; plt.xlabel("x") ; plt.ylabel("y")
plt.colorbar(location="bottom")
plt.tight_layout()

plt.show()



