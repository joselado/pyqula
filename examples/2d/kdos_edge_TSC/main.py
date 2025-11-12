# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

from pyqula import geometry
from pyqula import kdos
g = geometry.triangular_lattice()
h = g.get_hamiltonian()
h.add_onsite(2.)
h.add_rashba(0.5)
h.add_exchange([0.,0.,0.5])
h.add_swave(0.1)
import numpy as np
ng = 100

es0 = np.linspace(-0.2,0.2,ng)
klist = [(k,0,0) for k in np.linspace(-0.5,0.5,ng)]
op = h.get_operator("electron")
(ks,es,db,ds) = h.get_kdos(energies=es0,kpath=klist,operator=op,delta=1./ng)
# normalize 
db = db/np.max(db)
ds = ds/np.max(ds)

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(ks,es,c=db,vmax=0.3,cmap="inferno")
plt.xlabel("k") ; plt.ylabel("E")
plt.colorbar(ticks=[],label="DOS")

adb = db.reshape((ng,ng)) ; adb = np.mean(adb,axis=0)
plt.subplot(2,2,2)
plt.plot(adb,es0)
plt.xlim([0,np.max(adb)])
plt.xlabel("DOS") ; plt.ylabel("E")

plt.subplot(2,2,3)
plt.scatter(ks,es,c=ds,vmax=0.3,cmap="inferno")
plt.xlabel("k") ; plt.ylabel("E")
plt.colorbar(ticks=[],label="DOS")

ads = ds.reshape((ng,ng)) ; ads = np.mean(ads,axis=0)
plt.subplot(2,2,4)
plt.plot(ads,es0)
plt.xlim([0,np.max(ads)])
plt.xlabel("DOS") ; plt.ylabel("E")

plt.tight_layout()

plt.show()


