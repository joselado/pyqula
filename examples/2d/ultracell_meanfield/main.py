# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# A paramagnetic insulator, the honeycomb lattice with onsite energies +1 on
# A and -1 on B, under a slow exchange field with one period over n unit
# cells. The mean field with a Hubbard U runs on the supercell of n cells,
# solved at every iteration in the Bloch states of the unit cell. The U
# keeps the insulator paramagnetic and enhances the moment the field
# induces, by a factor 1.82 for U=1.5t.

import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of the honeycomb lattice
h = g.get_hamiltonian() # spinful, first-neighbor hopping t=1
h.add_sublattice_imbalance(1.0) # +1 on A and -1 on B, a gap of 2t
n = 24 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
B0 = 0.05 # amplitude of the exchange field, in units of t
hs.add_exchange(lambda r: [0.,0.,B0*np.cos(2*np.pi*b1.dot(r)/n)]) # along z
u = h.get_ultracell(hs) # every state kept, so this is the supercell exactly
nk = [2,12] # k-mesh of the ultracell: the ultracell is 24 cells long along a1
m0 = u.get_vev(operator="sz",nk=nk) # induced moment without interaction
u2 = u.get_mean_field_hamiltonian(U=1.5,filling=0.5,mf="ferroZ",nk=nk) # Hubbard U
m = u2.get_vev(operator="sz",nk=nk) # induced moment with the Hubbard U
x = np.array([b1.dot(r) for r in u2.hs.geometry.r]) # coordinate along a1
print("enhancement of the induced moment:",np.max(np.abs(m))/np.max(np.abs(m0)))

import matplotlib.pyplot as plt
o = np.argsort(x)
plt.plot(x[o],m0[o],"o",label="U=0")
plt.plot(x[o],m[o],"o",label="U=1.5t")
plt.xlabel("position along a1 [unit cells]") ; plt.ylabel("induced moment")
plt.legend() ; plt.show()
