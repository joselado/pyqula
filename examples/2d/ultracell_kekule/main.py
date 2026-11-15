# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Graphene with a Kekule modulation of its hoppings. The Kekule pattern
# repeats on the sqrt3 x sqrt3 cell, whose lattice vectors are not along
# those of graphene, A_S = M A with an integer matrix M. Its wavevector
# K-K' couples the two Dirac points, which both fold onto the center of
# the small Brillouin zone, so it opens a gap at the Dirac point. The
# unfolded spectral function along the k-path of graphene, what ARPES
# measures, shows that gap at K.

import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice() # unit cell of graphene
h = g.get_hamiltonian(has_spin=False) # first-neighbor hopping, t=1
M = [[2,1,0],[-1,1,0],[0,0,1]] # the sqrt3 x sqrt3 cell, A_S = M A
hs = h.get_supercell(M) # the ultracell, written as a supercell
hs.add_kekule(0.1) # the Kekule modulation of the hoppings
u = h.get_ultracell(hs) # every state kept, so this is the supercell exactly
(k,e,a) = u.get_kdos_bands(energies=np.linspace(-1.,1.,200),nk=200,
                           delta=0.01) # unfolded spectral function

# The weight follows the bands of graphene, and near K the cone splits:
# the levels closest to zero energy at K give the gap of the Kekule order.
kK = np.array([1./3.,1./3.,0.]) # the K point of graphene
es = np.linalg.eigvalsh(u.get_hk_gen()(u.basis.M@kK)) # folded onto kS
print("levels at K closest to zero:",np.sort(np.abs(es))[0:2])

import matplotlib.pyplot as plt
plt.scatter(k,e,c=a,cmap="inferno",s=4,vmax=np.percentile(a,99))
plt.colorbar(label="spectral function")
plt.xlabel("k-path of graphene") ; plt.ylabel("Energy [t]")
plt.show()
