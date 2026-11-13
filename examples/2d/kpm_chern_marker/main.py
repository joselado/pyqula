# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Local Chern marker of a large honeycomb flake, from the KPM projector.
# The left half is a Haldane insulator, with Chern number C=1. The right half
# has a large sublattice mass M=0.9, above 3*sqrt(3)*t2, so it is a trivial
# insulator, with C=0. The marker of each site only needs the projector
# between the sites close to it, so the cost is linear in the number of
# sites, and the map shows where the invariant is carried.
import numpy as np
from pyqula import islands

g = islands.get_geometry(name="honeycomb",n=30,nedges=4,rot=0.0,clean=False)
h = g.get_hamiltonian(has_spin=False) # spinless honeycomb flake
h.add_haldane(0.1) # Haldane coupling, gap of about 1
h.add_sublattice_imbalance(lambda r: 0.9*(r[0]>0.)) # mass only for x>0, trivial there
(r,c) = h.get_kpm_chern_marker(radius=6,npol=200) # local marker, per site
r = np.array(r)
print("number of sites =",len(r))
size = np.max(np.abs(r[:,0])) # half the width of the flake
left = (r[:,0]<-size/3.)*(r[:,0]>-2*size/3.)*(np.abs(r[:,1])<size/3.)
right = (r[:,0]>size/3.)*(r[:,0]<2*size/3.)*(np.abs(r[:,1])<size/3.)
print("marker in the left half =",round(np.mean(c[left]),3)) # about 1
print("marker in the right half =",round(np.mean(c[right]),3)) # about 0

import matplotlib.pyplot as plt
plt.scatter(r[:,0],r[:,1],c=c,s=4,cmap="RdBu_r",vmin=-1.5,vmax=1.5)
plt.colorbar(label="local Chern marker")
plt.axis("equal") ; plt.xlabel("x") ; plt.ylabel("y")
plt.show()
