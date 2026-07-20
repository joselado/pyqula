# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.get_supercell(4)
h = g.get_hamiltonian(has_spin=False)

from pyqula import potentials

f = potentials.commensurate_potential(g,minmax=[-1,1.])
f = f*0.3 # redefine the amplitude of the modulation
g.write_profile(f) # write the profile in a file 
h.add_sublattice_imbalance(f) # add a sublattice imbalance with this profile
(x,y,d) = h.get_ldos(e=0.0,num_bands=20) # compute LDOS
(k,e) = h.get_bands() # compute band structure

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.scatter(x,y,c=d,cmap="inferno")
plt.colorbar(label="LDOS")
plt.xlabel("x") ; plt.ylabel("y") ; plt.axis("equal")

plt.subplot(1,2,2)
plt.scatter(k,e)
plt.xlabel("k-path") ; plt.ylabel("Energy")

plt.show()







