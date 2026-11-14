# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Graphene under a slow potential, with one period over n unit cells.
# The ultracell is the supercell of n cells. We solve it in the Bloch
# states of graphene, and read the unfolded spectral function along the
# k-path of graphene, which is what ARPES measures on a superlattice.

import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian() # first-neighbor hopping, t=1
n = 20 # period of the modulation, in unit cells along a1
hs = h.get_supercell([n,1]) # the ultracell, written as a supercell
b1 = np.linalg.inv(np.array([g.a1,g.a2,g.a3]).T)[0] # b1.r is the coordinate along a1
V0 = 0.3 # amplitude of the potential, in units of t
hs.add_onsite(lambda r: V0*np.cos(2*np.pi*b1.dot(r)/n)) # one period over n cells
u = h.get_ultracell(hs) # every state kept, so this is the supercell exactly
(k,e,a) = u.get_kdos_bands(energies=np.linspace(-1.5,1.5,200),nk=200,
                           delta=0.02) # unfolded spectral function

# The band of graphene keeps most of the weight, with weaker replicas
# shifted by the wavevector of the potential. Near M the band spreads
# over t-V0 to t+V0, into a ladder of discrete levels.

import matplotlib.pyplot as plt
plt.scatter(k,e,c=a,cmap="inferno",s=4,vmax=np.percentile(a,99))
plt.colorbar(label="spectral function")
plt.xlabel("k-path of graphene") ; plt.ylabel("Energy [t]")
plt.show()
