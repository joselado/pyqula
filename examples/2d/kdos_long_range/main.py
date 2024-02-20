# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
g = geometry.square_lattice()
#h = g.get_hamiltonian()
h = g.get_hamiltonian(tij=[1.,0.2,1],has_spin=False)
#h = g.get_hamiltonian(tij=[0.5,0.,0.,1.,0.5],has_spin=True)
h.add_rashba(1.)
for t in h.hopping:
    print(t.dir)
    print(t.m)
    print()
#h.add_onsite(3.3)
#h.add_haldane(0.05)
energies = np.linspace(-8.0,8.0,100)
(k,e,ds,db) = h.get_kdos(energies=energies,delta=1e-2,
        nit=100)

import matplotlib.pyplot as plt

plt.subplot(1,3,1)
plt.title("Bulk bands")
plt.ylabel("Energy") ; plt.xticks([]) ; plt.xlabel("k-vector")

for ky in np.linspace(0.,1.0,100):
  kpath = [[kx,ky,0.] for kx in np.linspace(0.,1.0,100)]
  (kb,eb) = h.get_bands(kpath=kpath)
  plt.scatter(kb,eb,c="black")

plt.ylim([np.min(energies),np.max(energies)])

plt.subplot(1,3,2)
plt.title("Edge kdos")
plt.scatter(k,e,c=ds/np.max(ds),vmax=0.1,cmap="inferno")
plt.ylim([np.min(energies),np.max(energies)])
plt.ylabel("Energy") ; plt.xticks([]) ; plt.xlabel("k-vector")

plt.subplot(1,3,3)
plt.title("Bulk kdos")
plt.scatter(k,e,c=db/np.max(db),vmax=0.1,cmap="inferno")
plt.ylim([np.min(energies),np.max(energies)])
plt.ylabel("Energy") ; plt.xticks([]) ; plt.xlabel("k-vector")

plt.tight_layout()

plt.show()






