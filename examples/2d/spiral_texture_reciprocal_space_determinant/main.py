# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")




from pyqula import specialgeometry
import numpy as np
from pyqula import geometry,spintexture

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_sublattice_imbalance(0.3) # opena gap
h.add_rashba(.2) # and add Rashba
#h.set_filling(0.,nk=1)


# now define the two operators you want
ops = [h.get_operator("sx"),h.get_operator("sy")]
# and compute their expactation values
n = [1] # index of the band above the fermi energy you want
spintexture.conduction_texture(h,n=1, # number of valence states to consider
                     nsuper=1,nk=20)

import matplotlib.pyplot as plt

d = np.genfromtxt("DET_TEXTURE.OUT").T
kx,ky,dmx,dmy = d[0],d[1],d[2],d[3]

plt.subplot(1,2,1)
plt.scatter(kx,ky,c=dmx,cmap="bwr")
plt.colorbar(label="det(Mx)")
plt.xlabel("kx") ; plt.ylabel("ky")

plt.subplot(1,2,2)
plt.scatter(kx,ky,c=dmy,cmap="bwr")
plt.colorbar(label="det(My)")
plt.xlabel("kx") ; plt.ylabel("ky")

plt.show()








